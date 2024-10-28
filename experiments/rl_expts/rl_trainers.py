import torch
import torch.nn.functional as F
from trl import AutoModelForCausalLMWithValueHead
from dataclasses import dataclass
from typing import Any


@dataclass
class PPOConfig:
    batch_size: int = 256
    mini_batch_size: int = 16
    max_input_length: int = 512
    ignore_index: int = -100
    generation_config: Any = None
    gen_kwargs: Any = None


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


    # sample batch -> input_ids, attention_mask, labels
    def sample_batch(self, batch):

        output_list = []
        #logit_list = []
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
                    #return_dict_in_generate=True,
                    #output_logits=True,
                    **self.config.gen_kwargs
                )
                #output_ids = output.sequences
                #logits = output.logits
            # gather from accelerator
            output_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
            )
            #logits = self.accelerator.gather(self.accelerator.pad_across_processes(logits))
            output_list.append(output_ids)
            #logit_list.append(logits)

        label_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)
            )
        return output_list, label_ids
    
    # re-tokenize, set padding

    def prepare_input_for_ppo_step(self, output_list, gen_label_ids, device):
        # generated_ids -> context ids + generated action ids
        # attention mask -> attention mask for generated_ids
        # gen_label_ids -> generated action ids
        # context_label_ids -> context ids, needed to compute ce loss for context
        rl_inputs = {
            'generated_ids_list': [],
            'attention_mask_list': [],
            'gen_label_ids': [],
            'context_label_ids_list': [],
        }
        gen_label_ids, _ = self.re_tokenize(gen_label_ids, device)
        for l in range(len(output_list)):
            generated_ids, attention_mask = self.re_tokenize(output_list[l], device)
            # context labels needed for ce loss for context
            # get only context labels
            all_tokens = self.tokenizer.batch_decode(generated_ids)
            context_tokens = [t.split(self.tokenizer.sep_token)[
                0] for t in all_tokens]
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
            rl_inputs['generated_ids_list'].append(generated_ids)
            rl_inputs['attention_mask_list'].append(attention_mask)
            rl_inputs['gen_label_ids'] = gen_label_ids
            rl_inputs['context_label_ids_list'].append(context_label_ids)

        return rl_inputs


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



    # forward with generated samples to get logtis, values
    def forward_with_gen_samples(self, output_list, label_ids, low_mem=False):

        logit_list = []
        logprob_list = []
        ref_logprob_list = []
        value_list = []
        batch_size = self.config.batch_size
        mini_batch_size = self.config.mini_batch_size
        num_m_batches = batch_size//mini_batch_size

        device = 'cpu' if low_mem else self.accelerator.device

        # generated_ids -> context ids + generated action ids
        # attention mask -> attention mask for generated_ids
        # gen_label_ids -> generated action ids
        # context_label_ids -> context ids, needed to compute ce loss for context
        # TODO: involves gpu -> cpu -> gpu: can we speed this up?
        rl_inputs = self.prepare_input_for_ppo_step(
            output_list,
            label_ids,
            device,
        )

        for m in range(num_m_batches):
            # needs to be on gpu for forward
            generated_ids = rl_inputs['generated_ids_list'][m].to(self.accelerator.device)
            attention_mask = rl_inputs['attention_mask_list'][m].to(self.accelerator.device)
            # can be on cpu
            gen_label_ids = rl_inputs['gen_label_ids'].to(device)
            context_label_ids = rl_inputs['context_label_ids_list'][m].to(device)

            # output = (lm_logits, loss=None, value)
            with torch.no_grad():
                logits, _, values = self.model(input_ids=generated_ids, attention_mask=attention_mask)
                ref_logits, _, _ = self.ref_model(input_ids=generated_ids, attention_mask=attention_mask)

            # make sure same device
            logits = logits.to(device)
            values = values.to(device)
            ref_logits = ref_logits.to(device)

            gen_label_ids_m = gen_label_ids[m*mini_batch_size:(m+1)*mini_batch_size]
            logprob = self.logprobs_from_logits(logits, gen_label_ids_m)
            ref_logprob = self.logprobs_from_logits(ref_logits, gen_label_ids_m)

            # zero out
            logprob = self.zero_out_logits(logprob, context_label_ids, attention_mask)
            ref_logprob = self.zero_out_logits(ref_logprob, context_label_ids, attention_mask)
            logits = self.zero_out_logits(logits, context_label_ids, attention_mask)
            values = self.zero_out_logits(values, context_label_ids, attention_mask)

            # append to list
            logprob_list.append(logprob)
            ref_logprob_list.append(ref_logprob)
            logit_list.append(logits)
            value_list.append(values)

        forward_dict = {
            'logit_list': logit_list,
            'logprob_list': logprob_list,
            'ref_logprob_list': ref_logprob_list,
            'value_list': value_list,
        }

        return forward_dict
    

    def compute_rewards(self):
        #scores = score_function(generated_ids, gen_label_ids)
        # get score earlier
        pass


    def step(self, batch, low_mem=False):

        ## sample batch ##
        # outputs are padded per minibatch
        # label_ids -> single tensor
        output_list, label_ids = self.sample_batch(batch)

        ## forward pass with generated ids (+context) ##
        # lists with minibatch outputs
        # logit_list, logprob_list, ref_logprob_list, value_list
        # TODO: get scores here
        forward_dict = self.forward_with_gen_samples(output_list, label_ids, low_mem)
        
        ## compute rewards ##
        self.compute_rewards()

        # loop
        
        #   sample minibatch

        #   update policy