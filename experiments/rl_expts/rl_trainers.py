import torch
from trl import AutoModelForCausalLMWithValueHead


class RLTrainer:

    def __init__(
            self,
            model,
            tokenizer,
            max_input_length,
            ref_model=None,
            ignore_index=-100,
    ):
        self.ref_model = ref_model
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.ignore_index = ignore_index


    # re-tokenize left padded sequences need for batch generation to right padded sequences
    def re_tokenize(self, token_ids, device='cpu'):
        tokens = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        tokens = [o.replace(self.tokenizer.pad_token, '') for o in tokens]
        tokens = [o.replace(self.tokenizer.eos_token, '') for o in tokens]
        tokenized_tokens = self.tokenizer(
            tokens,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt',
        ).to(device)
        input_ids = tokenized_tokens['input_ids']
        attention_mask = tokenized_tokens['attention_mask']
        return input_ids, attention_mask


    # re-tokenize, set padding
    def prepare_input_for_rl_step(self, output_ids, gen_label_ids):
        generated_ids, attention_mask = self.re_tokenize(output_ids)
        gen_label_ids, _ = self.re_tokenize(gen_label_ids) 
        # context labels needed for ce loss for context
        # get only context labels
        all_tokens = self.tokenizer.batch_decode(generated_ids)
        context_tokens = [t.split(self.tokenizer.sep_token)[0] for t in all_tokens]
        tokenized_context = self.tokenizer(
            [c+self.tokenizer.sep_token for c in context_tokens],
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt',
        ).to(self.model.device)
        context_label_ids = tokenized_context['input_ids']
        # set context label padding to -100 
        context_label_ids = [
            [
                (l if l != self.tokenizer.pad_token_id else self.ignore_index) for l in label
            ] for label in context_label_ids.tolist()
        ]
        context_label_ids = torch.tensor(context_label_ids).to(self.model.device)
        return generated_ids, attention_mask, gen_label_ids, context_label_ids


class ReinforceTrainer(RLTrainer):
    pass


class PPOTrainer(RLTrainer):

    def __init__(
            self,
            model,
            tokenizer,
            accelerator,
            max_length,
            ref_model=None,
            ignore_index=-100,
        ):
        super().__init__(
            model,
            tokenizer,
            max_length,
            ref_model=None,
            ignore_index=-100,
        )
        self.optimizer = None
        self.scheduler = None


    # sample batch
    def generate_samples(self):
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