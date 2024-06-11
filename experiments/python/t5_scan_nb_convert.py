#!/usr/bin/env python
# coding: utf-8

# #### Installations

# In[2]:




# In[3]:


# #### Imports

# In[4]:


import math
from tqdm.auto import tqdm
import numpy as np
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader

import evaluate

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig,
    DataCollatorForSeq2Seq, get_scheduler, T5ForConditionalGeneration,
)

from datasets import load_dataset


# #### Hyperparamters

# In[20]:


SEED = 42

dataset_name = 'scan'
dataset_config_name = 'simple'
trust_remote_code = True
preprocessing_num_workers = None
overwrite_cache = False

gradient_accumulation_steps = 1

model_name = 'google-t5/t5-base'
#model_name = 'google/flan-t5-base'
# gpt2
# gemma

source_prefix = ""
max_target_length = 1024
max_source_length = 1024
padding = False
ignore_pad_token_for_loss = True
per_device_train_batch_size = 16
per_device_eval_batch_size = 16

weight_decay = 0.0
learning_rate = 5e-5
train_steps = 100000
eval_steps = 4000
lr_scheduler_type = 'linear'
num_warmup_steps = 0
checkpointing_steps = None
num_beams = 1

#output_dir = './'
output_dir = '/users/ujan/caricatures/models/scan_t5_base/'


# #### Setup accelerator

# In[21]:


accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
set_seed(SEED)


# #### Get dataset

# In[22]:


raw_datasets = load_dataset(dataset_name, dataset_config_name, trust_remote_code=trust_remote_code)


# #### Split train set into train and validation

# In[23]:


train_val_split = raw_datasets['train'].train_test_split(test_size=0.1, seed=SEED)
raw_datasets['train'] = train_val_split['train']
raw_datasets['validation'] = train_val_split['test']


# #### Load pretrained model or initialize from scratch. Load tokenizer

# In[24]:


config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

# pretrained model
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, trust_remote_code=trust_remote_code)

# from scratch
model = T5ForConditionalGeneration(config=config)
generation_config = GenerationConfig.from_pretrained(model_name)


# #### Resize the embeddings when necessary to avoid index errors

# In[25]:


embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
  model.resize_token_embeddings(len(tokenizer))
if model.config.decoder_start_token_id is None:
  raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

prefix = source_prefix


# #### Preprocess dataset

# In[26]:


column_names = raw_datasets["train"].column_names
input_column = column_names[0]
output_column = column_names[1]


def preprocess_function(examples):
  inputs = examples[input_column]
  targets = examples[output_column]
  inputs = [prefix + inp for inp in inputs]
  model_inputs = tokenizer(
      inputs, max_length=max_source_length, padding=padding, truncation=True)

  # tokenize targets with the `text_target` keyword argument
  labels = tokenizer(text_target=targets, max_length=max_target_length,
                     padding=padding, truncation=True)

  # if we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
  # padding in the loss.
  if padding == "max_length" and ignore_pad_token_for_loss:
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


# In[27]:


with accelerator.main_process_first():
  train_dataset = raw_datasets["train"].map(
      preprocess_function,
      batched=True,
      num_proc=preprocessing_num_workers,
      remove_columns=column_names,
      load_from_cache_file=not overwrite_cache,
      desc="Running tokenizer on dataset",
  )

  eval_dataset = raw_datasets["validation"].map(
      preprocess_function,
      batched=True,
      num_proc=preprocessing_num_workers,
      remove_columns=column_names,
      load_from_cache_file=not overwrite_cache,
      desc="Running tokenizer on dataset",
  )


# #### Data collator

# In[28]:


label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)


# #### Optimizer

# In[29]:


# split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)


# #### Scheduler

# In[30]:


lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps * accelerator.num_processes,
    num_training_steps=train_steps * accelerator.num_processes,
)


# #### Prepare everything with Accelerate

# In[31]:


model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


# # Train

# In[ ]:


progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process)
eval_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

total_loss = 0
completed_steps = 0
model.train()

while True:
  for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(model):
      outputs = model(**batch)
      loss = outputs.loss
      total_loss += loss.detach().float()
      accelerator.backward(loss)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()

      # checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        completed_steps += 1

      if completed_steps % eval_steps == 0:
        model.eval()

        # metric
        accuracy = 0.0

        gen_kwargs = {
            "max_length": max_target_length,
            "num_beams": num_beams,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
              generated_tokens = accelerator.unwrap_model(model).generate(
                  batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  generation_config=generation_config,
                  **gen_kwargs,
              )

              generated_tokens = accelerator.pad_across_processes(
                  generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
              labels = batch["labels"]
              # we did not pad to max length, we need to pad the labels too
              labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

              generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
              generated_tokens = generated_tokens.cpu().numpy()
              labels = labels.cpu().numpy()

              if ignore_pad_token_for_loss:
                # replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
              if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
              decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
              decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

              accuracy += sum([decoded_preds[i] == decoded_labels[i] for i in range(len(decoded_preds))])

              eval_bar.update(1)

        eval_bar.refresh()
        eval_bar.reset()

        accelerator.print('steps : {}'.format(completed_steps))
        accelerator.print('accuracy : {}'.format(accuracy / len(raw_datasets['validation'])))
        accelerator.print('')

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # save
        new_path = output_dir+'checkpoint_'+str(completed_steps)
        if not os.path.isdir(new_path):
          os.makedirs(new_path)
        unwrapped_model.save_pretrained(
            new_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

        model.train()

    if completed_steps == train_steps:
      quit()


# #### Test

# In[9]:






