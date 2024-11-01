from dataclasses import dataclass
from typing import Any

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from trl import AutoModelForCausalLMWithValueHead


@dataclass
class PPOConfig:
    batch_size: int = 256
    mini_batch_size: int = 16
    max_input_length: int = 512
    ignore_index: int = -100
    generation_config: Any = None
    gen_kwargs: Any = None
    init_kl_coef: float =  0.2
    target: float = 6.0
    horizon: float = 10000
    ppo_epoch: int = 5


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
            tokenizer,
            accelerator,
    ):
        
        self.EPSILON = 1e-20

        self.config = config
        self.model = model
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


class ReinforceTrainer(RLTrainer):
    def __init__(
            self,
            config,
            model,
            tokenizer,
            accelerator,
        ):
        super().__init__(
            config,
            model,
            tokenizer,
            accelerator,
        )



class PPOTrainer(RLTrainer):

    def __init__(
            self,
            config,
            model,
            ref_model,
            tokenizer,
            accelerator,
        ):
        super().__init__(
            config,
            model,
            tokenizer,
            accelerator,
        )
        self.ref_model = ref_model
        self.kl_controller = AdaptiveKLController(
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target,
            horizon=self.config.horizon,
        )


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
    

    def gather_from_acc(self, tensors):
        return self.accelerator.gather(
            self.accelerator.pad_across_processes(
                tensors, dim=1, pad_index=self.tokenizer.pad_token_id)
            )


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

        # needs to be on gpu for forward
        # TODO: convert to list and put onto accelerator one at a time

        output_ids = rl_inputs['output_ids']
        attention_mask = rl_inputs['attention_mask']
        output_ids_list = [
            output_ids[m*mini_batch_size:(m+1)*mini_batch_size] for m in range(num_m_batches)
        ]
        attention_mask_list = [
            attention_mask[m*mini_batch_size:(m+1)*mini_batch_size] for m in range(num_m_batches)
        ]

        #output_ids = rl_inputs['output_ids'].to(self.accelerator.device)
        #attention_mask = rl_inputs['attention_mask'].to(self.accelerator.device)
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
        logits = self.pad_and_stack(logit_list)
        ref_logits = self.pad_and_stack(ref_logit_list)
        values = self.pad_and_stack(values_list)

        # make sure same device
        output_ids = output_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = logits.to(device)
        ref_logits = ref_logits.to(device)
        values = values.to(device)

        logprobs = self.logprobs_from_logits(logits, gen_label_ids)
        ref_logprobs = self.logprobs_from_logits(ref_logits, gen_label_ids)

        # zero out
        logprobs = self.zero_out_logits(logprobs, context_label_ids, attention_mask)
        ref_logprobs = self.zero_out_logits(ref_logprobs, context_label_ids, attention_mask)
        logits = self.zero_out_logits(logits, context_label_ids, attention_mask)
        values = self.zero_out_logits(values, context_label_ids, attention_mask)

        # scores
        score, score_mask = self.score_function(
            output_ids,
            gen_label_ids,
            context_label_ids,
            metric='acc'
        )

        forward_dict = {
            'logits': logits,
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'values': values,
            'score': score,
            'score_mask': score_mask,
        }

        return forward_dict
    

    def logprobs_from_logits(self, logits, labels):
        # https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
        logp = torch.log(F.softmax(logits, dim=2) + self.EPSILON)
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy
    

    def zero_out_logits(self, logits, context_ids, attention_mask):
        # zero out context positions in logits
        logits[context_ids != self.config.ignore_index] = 0
        # zero out padding positions in logits
        logits[attention_mask == 0] = 0
        return logits


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
        # logits, logprobs, ref_logprobs
        # values, score, score_mask

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
    

    def train_minibatch(self, forward_dict, rewards):
        lastgaelam = 0
        advantages_reversed = []

        values = forward_dict['values']
        print(values[0])
        print(values.shape)
        print(rewards.shape)
        quit()

        # eq 11 and eq 12 from https://arxiv.org/pdf/1707.06347
        with torch.no_grad():
            nextvalues = values.roll(-1, dims=-1)
            nextvalues[:, -1] = 0
            delta = rewards + self.args.gamma * nextvalues - values
            for t in reversed(range(gen_len)):
                # nextvalues = values[:, t + 1] if (t < gen_len) else 0.0
                # delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
                if t < num_ce_tokens:
                    # Handle CE from Mixer
                    advantages_reversed.append(torch.zeros(
                        values.shape[0], dtype=torch.float, device=values.device))
                else:
                    lastgaelam = delta[:, t] + \
                        self.args.gamma * self.args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)


    def step(self, batch, low_mem=False):

        ## sample batch ##
        # output_ids -> context ids + generated action ids
        # gen_label_ids -> context_ids + label action ids
        output_ids, label_ids = self.sample_batch(batch)

        ## forward pass with generated ids (+context) ##
        # lists with minibatch outputs
        # logits, logprobs, ref_logprobs, values, score, score_mask
        forward_dict = self.forward_with_gen_samples(output_ids, label_ids, low_mem)
        
        ## compute rewards ##
        rewards = self.compute_rewards(forward_dict)
        
        ## run minibatches and update policy ##
        for _ in range(self.config.ppo_epochs):
            self.train_minibatch(forward_dict, rewards)