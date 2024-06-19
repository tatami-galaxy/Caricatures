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
    GemmaConfig, GemmaTokenizer, default_data_collator, 
    get_scheduler, GemmaForCausalLM,
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


    # load pretrained model or initialize from scratch. Load tokenizer
    config = GemmaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = GemmaTokenizer.from_pretrained(args.model_name_or_path)

    #print(raw_datasets['train'][0])
    ex = raw_datasets['train'][0]
    input_ids = tokenizer(ex['commands'], ex['actions'])['input_ids']
    print(tokenizer.decode(input_ids))
    quit()


    model = GemmaForCausalLM.from_pretrained(args.model_name_or_path, config=config)

    # resize the embeddings when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))




def run():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='google/gemma-2b',
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
        default=1024
    )
    parser.add_argument(
        '--max_target_length',
        type=int,
        default=1024
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
        "--from_scratch",
        action="store_true",
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