from tqdm.auto import tqdm
import os
import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, default_data_collator,
    get_scheduler, AutoModelForCausalLM,
    AutoConfig, GenerationConfig,
)

from datasets import load_dataset

from rouge_score import rouge_scorer

import scan_constants

def train(args, accelerator):
    
    # dataset
    raw_datasets = load_dataset(args.dataset, args.dataset_config, trust_remote_code=True)

    # split train set into train and validation
    train_val_split = raw_datasets['train'].train_test_split(test_size=args.validation_split, seed=args.seed)
    raw_datasets['train'] = train_val_split['train']
    raw_datasets['validation'] = train_val_split['test']

    column_names = raw_datasets["train"].column_names
    input_column = column_names[0]
    output_column = column_names[1]

    # format dataset with dummy tokens
    scan_constants.special_tokens_dict["additional_special_tokens"] = [scan_constants.dummy_token]

    def add_empty_token(x):
        command_str = x[input_column]
        command = command_str.split()
        padded_command = []
        index = 0
        c = 0
        while index < scan_constants.command_max_len:
            expected_cs = scan_constants.command_structure[index]
            if c < len(command) and command[c] in expected_cs:
                padded_command.append(command[c])
                c += 1
            else:
                padded_command.append(scan_constants.dummy_token)
            index += 1
        
        x[input_column] = ' '.join(padded_command)
        return x

    with accelerator.main_process_first():
        raw_datasets["train"] = raw_datasets["train"].map(
            add_empty_token,
            batched=False,
            num_proc=args.num_workers, 
            desc="Running tokenizer on dataset",
        )
        raw_datasets["validation"] = raw_datasets["validation"].map(
            add_empty_token,
            batched=False,
            num_proc=args.num_workers,
            desc="Running tokenizer on dataset",
    )
        

    # model and tokenizer
    config = AutoConfig.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, trust_remote_code=True)
    tokenizer.add_special_tokens(scan_constants.special_tokens_dict)

    # LEFT PADDING FOR BATCH GENARATION
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
                args.model_checkpoint,
                config=config,
                trust_remote_code=True,
            )

    # Resize the embeddings only when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Generation config
    generation_config = GenerationConfig.from_pretrained(args.model_checkpoint)
    #gen_dict = generation_config.to_dict()
    #gen_dict["language"] = model_lang
    # reload with new attributes
    #generation_config = GenerationConfig.from_dict(gen_dict)
    #max_gen_length = model.config.max_length
    #num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    generation_config.pad_token_id = tokenizer.pad_token_id
    gen_kwargs = {"max_new_tokens": args.max_gen_length, "num_beams": args.num_beams}


    # preprocess dataset
    def preprocess_function(examples):
        # commands, actions
        inputs = examples[input_column]
        targets = examples[output_column]

        # tokenize as single sequence separated by special token
        model_inputs = tokenizer(
            [i+tokenizer.sep_token for i in inputs],
            padding='max_length', max_length=args.max_input_length
        )
        # labels same as inputs. labels shifted right in the model forward by default
        model_inputs['labels'] = tokenizer(
            [t+tokenizer.eos_token for t in targets],
            padding='max_length', max_length=args.max_input_length
        )['input_ids']

        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )


    # main dataloaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size
    ) 
    # prepare main dataloaders
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # TODO: define ppo trainer


    # train
    global_step = 0  # tracks total steps
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
    # eval bar
    eval_bar = tqdm(range(len(eval_dataloader)), position=1)

    while True:
        for batch in train_dataloader:
            pass



def run():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='google-t5/t5-base', # gp2, gemma
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
        "--source_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        '--max_source_length',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--max_target_length',
        type=int,
        default=1024
    )
    parser.add_argument(
        "--output_dir",
        default='/users/ujan/caricatures/models/scan_t5-base/',
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
        "--from_scratch",
        action="store_true",
    )
    parser.add_argument(
        "--lr",
        default=5e-5, # 1e-5, 2e-3, 2e-5
        type=float,
        help="Learning rate to use. From scratch training is quite sensitive to the learning rate."
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
        default='fp16',
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

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    print('output directory set to : {}'.format(args.output_dir))

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=args.output_dir
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "max_input_length": args.max_input_length,
        "max_gen_length": args.max_gen_length,
    }
    # run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


if __name__ == "__main__":

    run()