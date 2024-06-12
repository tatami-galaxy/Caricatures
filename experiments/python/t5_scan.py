import math
from tqdm.auto import tqdm
import numpy as np
import os
import argparse

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

# torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))


def train(args, accelerator):

    # get dataset
    raw_datasets = load_dataset(args.dataset, args.dataset_config, trust_remote_code=True)


    # split train set into train and validation
    train_val_split = raw_datasets['train'].train_test_split(test_size=0.1, seed=args.seed)
    raw_datasets['train'] = train_val_split['train']
    raw_datasets['validation'] = train_val_split['test']


    # load pretrained model or initialize from scratch. Load tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.from_scratch:
        model = T5ForConditionalGeneration(config=config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)

    generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)


    # resize the embeddings when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix

    # preprocess dataset
    column_names = raw_datasets["train"].column_names
    input_column = column_names[0]
    output_column = column_names[1]


    def preprocess_function(examples):
        inputs = examples[input_column]
        targets = examples[output_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding=args.padding, truncation=True)

        # tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length,
                            padding=args.padding, truncation=True)

        # if we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
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


    # data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


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
                accuracy = 0.0
                val_loss = 0
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # generate and compute metric
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

                accuracy = accuracy / len(raw_datasets['validation'])
                # add wer for hindi
                accelerator.print('step : {}, accuracy : {}'.format(global_step + 1, accuracy))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "accuracy": accuracy,
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
                    generation_config.save_pretrained(output_dir)

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
    parser.add_argument(
        "--dataset_config",
        default="simple",
        type=str,
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
        default='/users/ujan/caricatures/models/scan_t5_base/',
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
        "--mixed_precision",
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


if __name__ == "__main__":

    run()
