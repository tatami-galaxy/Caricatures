import torch
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
            ref_model=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.ref_model = ref_model


    # re-tokenize left padded sequences need for batch generation to right padded sequences
    def re_tokenize(self, token_ids, device='cpu'):
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


    # re-tokenize, set padding
    def prepare_input_for_rl_step(self, output_list, gen_label_list, device='cpu'):
        # generated_ids -> context ids + generated action ids
        # attention mask -> attention mask for generated_ids
        # gen_label_ids -> generated action ids
        # context_label_ids -> context ids, needed to compute ce loss for context
        rl_inputs = {
            'generated_ids_list': [],
            'attention_mask_list': [],
            'gen_label_ids_list': [],
            'context_label_ids_list': [],
        }
        for l in range(len(output_list)):
            generated_ids, attention_mask = self.re_tokenize(output_list[l], device)
            gen_label_ids, _ = self.re_tokenize(gen_label_list[l]) 
            # context labels needed for ce loss for context
            # get only context labels
            all_tokens = self.tokenizer.batch_decode(generated_ids)
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
            rl_inputs['generated_ids_list'].append(generated_ids)
            rl_inputs['attention_mask_list'].append(attention_mask)
            rl_inputs['gen_label_ids_list'].append(gen_label_ids)
            rl_inputs['context_label_ids_list'].append(context_label_ids)
        return rl_inputs


class ReinforceTrainer(RLTrainer):
    pass


class PPOTrainer(RLTrainer):

    def __init__(
            self,
            config,
            model,
            tokenizer,
            accelerator,
            ref_model=None,
        ):
        super().__init__(
            config,
            model,
            tokenizer,
            accelerator,
            ref_model=None,
        )


    # sample batch -> input_ids, attention_mask, labels
    def sample_batch(self, batch):

        output_list = []
        label_list = []
        batch_size = self.config.batch_size
        mini_batch_size = self.config.mini_batch_size
        num_m_batches = batch_size/mini_batch_size

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
                print(output_ids)
                print(output_ids.shape)
                quit()
                
            # gather from accelerator
            output_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
            )
            label_ids = self.accelerator.gather(
                self.accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)
            )
            output_list.append(output_ids)
            label_list.append(label_ids)


    # forward with generated samples ti get logtis, values
    def forward_with_gen_samples(self):
        with torch.no_grad():
                output = self.model(model_input, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
                ref_logits = self.ref_model(model_input, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)["logits"]
        logits, v = output["logits"], output["values"]   
        if decoder_input_ids is None:
            value = v[:, -gen_len-1:-1]
            logprob = logprobs_from_logits(logits[:,-gen_len-1:-1,:], model_input[:,-gen_len:])
            ref_logprob = logprobs_from_logits(ref_logits[:,-gen_len-1:-1,:], model_input[:,-gen_len:])

            value[attention_mask[:, -gen_len-1:-1] == 0] = 0 # Zero out
            logprob[attention_mask[:, -gen_len-1:-1] == 0] = 0 # Zero out
            ref_logprob[attention_mask[:, -gen_len-1:-1] == 0] = 0 # Zero out
        else:
            value = v[:, :-1]
            logprob = logprobs_from_logits(logits[:,:-1,:], decoder_input_ids[:,1:])
            ref_logprob = logprobs_from_logits(ref_logits[:,:-1,:], decoder_input_ids[:,1:])

            value[decoder_attention_mask[:,:-1] == 0] = 0 # Zero out
            logprob[decoder_attention_mask[:,:-1] == 0] = 0 # Zero out
            ref_logprob[decoder_attention_mask[:,:-1] == 0] = 0 # Zero out
            
        # Handle CE for MIXER
        if num_ce_tokens > 0:
            value[:,:num_ce_tokens] = 0
            logprob[:,:num_ce_tokens] = 0
            ref_logprob[:,:num_ce_tokens] = 0        
        
        return logprob, ref_logprob, value


    def step(self):
        # sample batch
        self.generate_samples()


        # calculate rewards
        # loop:
        #   sample minibatch
        #   update policy
        pass