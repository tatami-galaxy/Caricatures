import math
from tqdm.auto import tqdm
import numpy as np
import os
import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader

import evaluate

from transformers import (
    AutoTokenizer, default_data_collator, DataCollatorWithPadding,
    get_scheduler, AutoModelForCausalLM
)

from datasets import load_dataset


def train(args, accelerator):
    
    # `trust_remote_code` is to be used with Auto classes

    # get dataset
    raw_datasets = load_dataset(args.dataset, args.dataset_config, trust_remote_code=True)


    # split train set into train and validation
    train_val_split = raw_datasets['train'].train_test_split(test_size=args.validation_split, seed=args.seed)
    raw_datasets['train'] = train_val_split['train']
    raw_datasets['validation'] = train_val_split['test']


    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token="<pad>")

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,)

    # resize the embeddings when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # preprocess dataset
    column_names = raw_datasets["train"].column_names
    input_column = column_names[0]
    output_column = column_names[1]

    """mlen = 0
    for sample in raw_datasets['validation']:
        model_inputs = tokenizer(sample[input_column], sample[output_column])
        l = len(model_inputs['input_ids'])
        if l > mlen:
            mlen = l
    print(mlen)
    quit()"""

    def preprocess_function(examples):
        # commands, actions
        inputs = examples[input_column]
        targets = examples[output_column]

        # tokenize as single sequence separated by special token (<bos>)
        # padding = False by default
        model_inputs = tokenizer(inputs, targets, padding=True)
        # labels same as inputs. labels shifted right in the model forward by default
        model_inputs['labels'] = model_inputs['input_ids'].copy()
        # set label padding to -100 
        model_inputs['labels'] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs['labels']
        ]

        return model_inputs


    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            #batch_size=4, # for testing
            num_proc=args.num_workers,  # set as 1 for testing, otherwise args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # data collator and loaders

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.train_steps * accelerator.num_processes,
    )

    # prepare everything for accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval

    accelerator.log({
        "train_batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "gpus": accelerator.state.num_processes
    },
        step=global_step + 1,
    )

    # load from checkpoint
    ## loading checkpoint changing CER. val loss behaviour same. not sure why. ##
    # check if checkpoint directory passed in
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # if resumed from checkpoint
        # we need to skip steps until we reach the current step
        # ../checkpoint-123 -> int(123)
        steps_completed = int(
            args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
        global_step = steps_completed
        if args.skip_steps:
            train_dataloader = accelerator.skip_first_batches(
                train_dataloader, steps_completed)  # consider dataset len


    num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": num_beams,
    }


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
    # eval bar
    eval_bar = tqdm(range(len(eval_dataloader)), position=1)

    while True:

        model.train()

        for batch in train_dataloader:
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

            if (global_step + 1) % args.eval_steps == 0:
                model.eval()
                val_loss = 0
                
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    eval_bar.update(1)

                eval_bar.refresh()
                eval_bar.reset()

                accelerator.print('step : {}, val loss  : {}'.format(global_step + 1, val_loss/len(eval_dataloader)))
                accelerator.log({
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.per_device_train_batch_size),
                    "val_loss": val_loss / len(eval_dataloader)
                },
                    step=global_step + 1,
                )

                # save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # saved to folders named `checkpoint-{global_step}`
                # will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # if mixed precision was used, will also save a "scalar.bin" file
                output_dir = f"checkpoint-{global_step + 1}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # save config
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    # model.config.save_pretrained(output_dir)
                    unwrapped_model.config.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                model.train()
                total_loss = 0

            global_step += 1

            if global_step >= args.train_steps:
                return




def run():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='openai-community/gpt2',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
    )
    parser.add_argument(
        "--padding",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        default="scan",
        type=str,
        help="Dataset",
    )
    # 'simple', 'addprim_jump', 'addprim_turn_left', 'filler_num0', 
    # 'filler_num1', 'filler_num2', 'filler_num3', 'length', 
    # 'template_around_right', 'template_jump_around_right', 
    # 'template_opposite_right', 'template_right'
    parser.add_argument(
        "--dataset_config",
        default="simple",
        type=str,
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        '--max_source_length',
        type=int,
        default=512
    )
    parser.add_argument(
        '--max_target_length',
        type=int,
        default=512
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="checkpoint directory to load model from",
    )
    parser.add_argument(
        "--skip_steps",
        action="store_true",
        help="whether to skip steps in dataloader (checkpoint)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=os.cpu_count(),  # 1, None, 32
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=100000,
        type=int,
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=5e-5, 
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default='linear',
        type=str,
    )
    parser.add_argument(
        "--mixed_precision", # choose from no, fp16, bf16 or fp8
        default='no',
        type=str,
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=1
    )

    # parse args
    args = parser.parse_args()

    # set seed 
    set_seed(args.seed)

    if args.output_dir is None:
        raise ValueError(f"Pass in output directory to save checkpoints")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    print('output directory set to : {}'.format(args.output_dir))

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.per_device_train_batch_size,
    }
    # run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()



if __name__ == '__main__':
    run()