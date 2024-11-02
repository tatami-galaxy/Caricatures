from dataclasses import dataclass
from typing import Any
from tqdm.auto import tqdm
import collections

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss


@dataclass
class PPOConfig:
    batch_size: int = 256
    mini_batch_size: int = 16
    max_input_length: int = 512
    ignore_index: int = -100
    epsilon: float = 1e-20
    generation_config: Any = None
    gen_kwargs: Any = None

    init_kl_coef: float =  0.2
    target: float = 6.0
    horizon: float = 10000
    ppo_epochs: int = 5
    gamma: float = 1
    lam: float = 0.95
    cliprange_value: float = 0.2
    cliprange: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 1.0


class AdaptiveKLController:
    # https://arxiv.org/pdf/1909.08593.pdf
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class RLTrainer:

    def __init__(
            self,
            config,
            model,
            optimizer,
            lr_scheduler,
            tokenizer,
            accelerator,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer,
        self.lr_scheduler = lr_scheduler,
        self.tokenizer = tokenizer
        self.accelerator = accelerator


    # re-tokenize left padded sequences need for batch generation to right padded sequences
    def re_tokenize(self, token_ids, device):
        tokens = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        tokens = [o.replace(self.tokenizer.pad_token, '') for o in tokens]
        tokens = [o.replace(self.tokenizer.eos_token, '') for o in tokens]
        tokenized_tokens = self.tokenizer(
            tokens,
            padding='max_length',
            max_length=self.config.max_input_length,
            return_tensors='pt',
        ).to(device)
        input_ids = tokenized_tokens['input_ids']
        attention_mask = tokenized_tokens['attention_mask']
        return input_ids, attention_mask
    

    def flatten_dict(self, nested, sep='/'):
        # flatten dictionary and concatenate nested keys with separator
        def rec(nest, prefix, into):
            for k, v in nest.items():
                if sep in k:
                    raise ValueError(
                        f"separator '{sep}' not allowed to be in key '{k}'")
                if isinstance(v, collections.abc.Mapping):
                    rec(v, prefix + k + sep, into)
                else:
                    into[prefix + k] = v
        flat = {}
        rec(nested, '', flat)
        return flat
    

    def pad_and_stack(self, tensor_list, side='right'):
        # get list of all tensors
        all_tensors = [t[i] for t in tensor_list for i in range(t.shape[0])]
        # pad and stack
        tensor_ids = pad_sequence(
            all_tensors,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=side,
        )
        return tensor_ids
    

    def gather_from_acc(self, tensors):
        return self.accelerator.gather(
            self.accelerator.pad_across_processes(
                tensors, dim=1, pad_index=self.tokenizer.pad_token_id)
        )
    

    def logprobs_from_logits(self, logits, labels):
        # https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
        logp = torch.log(F.softmax(logits, dim=2) + self.config.epsilon)
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy


    def entropy_from_logits(self, logits):
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, axis=-1) - \
            torch.sum(pd*logits, axis=-1)
        return entropy


    def zero_out_logits(self, logits, context_ids, attention_mask):
        # zero out context positions in logits
        logits[context_ids != self.config.ignore_index] = 0
        # zero out padding positions in logits
        logits[attention_mask == 0] = 0
        return logits
    

    def padded_mean(self, values, mask):
        return torch.sum(values)/torch.sum(mask)


    def padded_var(self, values, mask):
        p_mean = self.padded_mean(values, mask)
        var_sub = values - p_mean
        var_sub_sq = torch.mul(torch.mul(var_sub, var_sub), mask)
        var_sum_sq = torch.sum(var_sub_sq)
        return var_sum_sq / (torch.sum(mask) - 1)


    def whiten(self, values, mask, shift_mean=True):
        # whiten values
        mean, var = self.padded_mean(
            values, mask), self.padded_var(values, mask)
        whitened = (values - mean) * torch.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return torch.mul(whitened, mask)


    def clip_by_value(self, x, tensor_min, tensor_max):
        # tensor extenstion to torch.clamp
        # https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
        clipped = torch.max(torch.min(x, tensor_max), tensor_min)
        return clipped


class ReinforceTrainer(RLTrainer):
    def __init__(
            self,
            config,
            model,
            optimizer,
            lr_scheduler,
            tokenizer,
            accelerator,
        ):
        super().__init__(
            config,
            model,
            optimizer,
            lr_scheduler,
            tokenizer,
            accelerator,
        )


class PPOTrainer(RLTrainer):

    def __init__(
            self,
            config,
            model,
            ref_model,
            optimizer,
            lr_scheduler,
            tokenizer,
            accelerator,
        ):
        super().__init__(
            config,
            model,
            optimizer,
            lr_scheduler,
            tokenizer,
            accelerator,
        )
        self.ref_model = ref_model
        self.kl_controller = AdaptiveKLController(
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target,
            horizon=self.config.horizon,
        )
        self.ce_loss_fct = CrossEntropyLoss(ignore_index=self.config.ignore_index)


    # sample batch -> input_ids, attention_mask, labels
    def sample_batch(self, batch):

        output_list = []
        batch_size = self.config.batch_size
        mini_batch_size = self.config.mini_batch_size
        num_m_batches = batch_size//mini_batch_size

        # sample batch : need to do iteratively for large batch sizes
        # cant stack them, different sized outptus
        for m in range(num_m_batches):
            with torch.no_grad():
                mini_batch = {k: v[m*mini_batch_size:(m+1)*mini_batch_size] for k, v in batch.items()}
                output_ids = self.accelerator.unwrap_model(self.model).generate(
                    **mini_batch,
                    generation_config=self.config.generation_config,
                    **self.config.gen_kwargs
                )
            # gather from accelerator
            output_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
            )
            output_list.append(output_ids)

        label_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)
            )
        
        # stack output_list -> tensors of different length
        output_ids = self.pad_and_stack(output_list, side='left')

        return output_ids, label_ids
    

    # re-tokenize, set padding
    def prepare_input_for_ppo_step(self, output_ids, gen_label_ids, device):

        output_ids, attention_mask = self.re_tokenize(output_ids, device)
        # context labels needed for ce loss for context -> TODO: use gen ids instead of output ids?
        # get only context label tokens -> model always generates context first
        all_tokens = self.tokenizer.batch_decode(output_ids)
        context_tokens = [t.split(self.tokenizer.sep_token)[0] for t in all_tokens]
        tokenized_context = self.tokenizer(
            [c+self.tokenizer.sep_token for c in context_tokens],
            padding='max_length',
            max_length=self.config.max_input_length,
            return_tensors='pt',
        ).to(device)
        context_label_ids = tokenized_context['input_ids']
        # set context label padding to -100
        context_label_ids = [
            [
                (l if l != self.tokenizer.pad_token_id else self.config.ignore_index) for l in label
            ] for label in context_label_ids.tolist()
        ]
        context_label_ids = torch.tensor(context_label_ids).to(device)

        # collect into dict
        # output_ids -> context ids + generated action ids
        # attention mask -> attention mask for output_ids
        # gen_label_ids -> context_ids + label action ids
        # context_label_ids -> context ids, needed to compute ce loss for context
        rl_inputs = {}
        rl_inputs['output_ids'] = output_ids
        rl_inputs['attention_mask'] = attention_mask
        rl_inputs['context_label_ids'] = context_label_ids
        rl_inputs['gen_label_ids'] = gen_label_ids

        return rl_inputs


    # forward with generated samples to get logtis, values
    def forward_with_gen_samples(self, output_ids, label_ids, low_mem=False):

        logit_list = []
        ref_logit_list = []
        values_list = []

        batch_size = self.config.batch_size
        mini_batch_size = self.config.mini_batch_size
        num_m_batches = batch_size//mini_batch_size
        device = 'cpu' if low_mem else self.accelerator.device

        # change padding from left to right
        # output_ids -> context ids + generated action ids
        # attention mask -> attention mask for output_ids
        # gen_label_ids -> context_ids + label action ids
        # context_label_ids -> context ids, needed to compute ce loss for context
        # TODO: involves gpu -> cpu -> gpu: can we speed this up?
        rl_inputs = self.prepare_input_for_ppo_step(
            output_ids,
            label_ids,
            device,
        )

        # model forward
        output_ids = rl_inputs['output_ids'].to(device)
        attention_mask = rl_inputs['attention_mask'].to(device)
        output_ids_list = [
            output_ids[m*mini_batch_size:(m+1)*mini_batch_size] for m in range(num_m_batches)
        ]
        attention_mask_list = [
            attention_mask[m*mini_batch_size:(m+1)*mini_batch_size] for m in range(num_m_batches)
        ]
        # can be on cpu
        gen_label_ids = rl_inputs['gen_label_ids'].to(device)
        context_label_ids = rl_inputs['context_label_ids'].to(device)

        # need to do iteratively for large batch sizes
        # output = (lm_logits, loss=None, value)
        for m in range(num_m_batches):
            with torch.no_grad():
                logits, _, values = self.model(
                    input_ids=output_ids_list[m].to(self.accelerator.device),
                    attention_mask=attention_mask_list[m].to(self.accelerator.device)
                )
                ref_logits, _, _ = self.ref_model(
                    input_ids=output_ids_list[m].to(self.accelerator.device),
                    attention_mask=attention_mask_list[m].to(self.accelerator.device)
                )
            # gather from accelerator
            logits = self.gather_from_acc(logits)
            ref_logits = self.gather_from_acc(ref_logits)
            values = self.gather_from_acc(values)

            # append to list
            logit_list.append(logits)
            ref_logit_list.append(ref_logits)
            values_list.append(values)

        # stack lists
        # make sure same device
        logits = self.pad_and_stack(logit_list).to(device)
        ref_logits = self.pad_and_stack(ref_logit_list).to(device)
        values = self.pad_and_stack(values_list).to(device)
        # logprobs
        logprobs = self.logprobs_from_logits(logits, gen_label_ids)
        ref_logprobs = self.logprobs_from_logits(ref_logits, gen_label_ids)

        # zero out
        logprobs = self.zero_out_logits(logprobs, context_label_ids, attention_mask)
        ref_logprobs = self.zero_out_logits(ref_logprobs, context_label_ids, attention_mask)
        values = self.zero_out_logits(values, context_label_ids, attention_mask)

        # scores
        score, score_mask = self.score_function(
            output_ids,
            gen_label_ids,
            context_label_ids,
            metric='acc'
        )

        forward_dict = {
            'output_ids': output_ids,
            'attention_mask': attention_mask,
            'gen_label_ids': gen_label_ids,
            'context_label_ids': context_label_ids,
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'values': values,
            'score': score,
            'score_mask': score_mask,
        }

        return forward_dict


    def score_function(self, output_ids, gen_label_ids, context_label_ids, metric='acc'):
        # calculate score per time step
        if metric == 'acc':
            score = (output_ids == gen_label_ids).type(torch.float)
            # zero out context and padding positions
            score[context_label_ids != self.config.ignore_index] = 0
            score[gen_label_ids == self.tokenizer.pad_token_id] = 0
            
            # score mask
            score_mask = torch.ones_like(score).to(score.device)
            score_mask[context_label_ids != self.config.ignore_index] = 0
            score_mask[gen_label_ids == self.tokenizer.pad_token_id] = 0

        elif metric == 'incr_acc':
            pass

        else:
            raise ValueError('Incorrect metric passed to score function')
        
        return score, score_mask
    

    def compute_rewards(self, forward_dict, kl_penalty=True):
        # https://arxiv.org/pdf/1909.08593 -> equation 2
        logprobs = forward_dict['logprobs']
        ref_logprobs = forward_dict['ref_logprobs']
        score = forward_dict['score']

        if kl_penalty:
            kl = logprobs - ref_logprobs  # will be zero initially
            rewards = -self.kl_controller.value * kl
            rewards = rewards + score
        else:
            rewards = score
            
        return rewards


    def compute_advantages(self, values, rewards, mask):
        lastgaelam = 0
        # reversed since delta_t depends on delta_t+1, delta_t+2, ...
        advantages_reversed = []

        # eq 11 and eq 12 ppo paper : https://arxiv.org/pdf/1707.06347
        with torch.no_grad():
            nextvalues = values.roll(-1, dims=-1)
            nextvalues[:, -1] = 0
            delta = rewards + self.config.gamma * nextvalues - values
            
            for t in reversed(range(delta.shape[1])):
                # advantage estimate for each timestep (revresed)
                lastgaelam = delta[:, t] + self.config.gamma * self.config.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

            # mask out context and padding positions
            advantages = torch.mul(advantages, mask)
            # whiten
            advantages = self.whiten(advantages, mask)

            return advantages


    def run_minibatch(self, mini_batch, rewards, low_mem=False):

        device = 'cpu' if low_mem else self.accelerator.device

        output_ids = mini_batch['output_ids']
        attention_mask = mini_batch['attention_mask']
        gen_label_ids = mini_batch['gen_label_ids']
        # context_ids padded with self.config.ignore_index
        context_label_ids = mini_batch['context_label_ids']
        old_logprobs = mini_batch['logprobs']
        values = mini_batch['values']
        score_mask = mini_batch['score_mask']

        # compute advantages from values and rewards
        advantages = self.compute_advantages(values, rewards, score_mask)

        # model forward
        # output = (lm_logits, loss=None, value)
        logits, _, vpred = self.model(
            input_ids=output_ids.to(self.accelerator.device),
            attention_mask=attention_mask.to(self.accelerator.device)
        )
        # gather from accelerator
        # make sure same device
        logits = self.gather_from_acc(logits).to(device)
        vpred = self.gather_from_acc(vpred).to(device)

        # logprobs
        logprobs = self.logprobs_from_logits(logits, gen_label_ids)
        # zero out context and padding positions
        logprobs = self.zero_out_logits(logprobs, context_label_ids, attention_mask)
        vpred = self.zero_out_logits(vpred, context_label_ids, attention_mask)

        # value function fitting 
        # calculate value function loss with clipping
        vpred_clipped = self.clip_by_value(
            vpred,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value
        )
        # y values for value function fitting
        # instead of using the single sample return,
        # use bootstrapped estimate of the value
        returns = advantages + values
        # equation 9 ppo paper : https://arxiv.org/pdf/1707.06347
        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpred_clipped - returns)**2
        # clamped loss following DQL
        # https://discuss.pytorch.org/t/creating-a-clipped-loss-function/12022/4
        vf_loss = .5 * torch.mean(
            torch.clamp(torch.max(vf_losses1, vf_losses2), min=-1, max=1)
        )
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        # calculate policy gradient loss
        # importance sampling ratio
        ratio = torch.exp(logprobs - old_logprobs)
        # clipping surrogate, section 6.1 ppo paper
        # https://arxiv.org/pdf/1707.06347
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.cliprange,
            1.0 + self.config.cliprange
        )
        # clamped loss following DQL
        # https://discuss.pytorch.org/t/creating-a-clipped-loss-function/12022/4
        pg_loss = torch.clamp(torch.max(pg_losses, pg_losses2), min=-1, max=1)
        # TODO: zero out context positions and padding positions in pg_loss?
        # compute avg pg_loss to get a scalar loss
        pg_loss = torch.sum(pg_loss) / torch.sum(score_mask)

        # cross entropy loss for context
        # shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_context_labels = context_label_ids[..., 1:].contiguous()
        ce_loss = self.ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_context_labels.view(-1)
        )

        # total loss
        loss =  pg_loss + self.config.vf_coef * vf_loss + ce_loss

        # get stats
        with torch.no_grad():
            pg_clipfrac = self.padded_mean(torch.gt(pg_losses2, pg_losses).double(), score_mask)
            entropy = self.padded_mean(self.entropy_from_logits(logits), score_mask)
            approxkl = .5 * self.padded_mean((logprobs - old_logprobs)**2, score_mask)
            policykl = self.padded_mean(logprobs - old_logprobs, score_mask)

            return_mean, return_var = self.padded_mean(
                returns, score_mask), self.padded_var(returns, score_mask)
            value_mean, value_var = self.padded_mean(
                values, score_mask), self.padded_var(values, score_mask)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, ce_loss=ce_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=self.padded_mean(advantages, score_mask), ratio=ratio
            ),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )

        return loss, self.flatten_dict(stats)


    def step(self, batch, low_mem=False):

        batch_size = self.config.batch_size
        mini_batch_size = self.config.mini_batch_size
        num_m_batches = batch_size//mini_batch_size

        ## sample batch ##
        # output_ids -> context ids + generated action ids
        # gen_label_ids -> context_ids + label action ids
        output_ids, label_ids = self.sample_batch(batch)

        ## forward pass with generated ids (+context) ##
        # output_ids, attention_mask
        # gen_label_ids, context_label_ids,
        # logprobs, ref_logprobs, values, 
        # score, score_mask
        forward_dict = self.forward_with_gen_samples(output_ids, label_ids, low_mem)
        
        ## compute rewards ##
        rewards = self.compute_rewards(forward_dict)
        
        ## run minibatches and update policy ##
        ppo_bar = tqdm(range(self.config.ppo_epochs), disable=not self.accelerator.is_main_process, position=1)
        stats = []
        for _ in range(self.config.ppo_epochs):
            for m in range(num_m_batches):
                mini_batch = {
                    k: v[m*mini_batch_size:(m+1)*mini_batch_size] for k, v in forward_dict.items()
                }
                mini_batch_rewards = rewards[m*mini_batch_size:(m+1)*mini_batch_size]
                loss, train_stats = self.run_minibatch(mini_batch, mini_batch_rewards, low_mem)
                print(train_stats)
                quit()
                stats.append(train_stats)

                # backprop
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            ppo_bar.update(1)
        
        ## housekeeping ##
        # TODO: process and update stats
        self.kl_ctl.update(stats['objective/kl'], self.args.batch_size)
