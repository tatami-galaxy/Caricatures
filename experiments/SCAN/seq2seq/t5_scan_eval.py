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
    raw_datasets = load_dataset(
        args.dataset, args.dataset_config, trust_remote_code=True)

    # load pretrained model or initialize from scratch. Load tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, config=config)

    # resize the embeddings when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

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
        test_dataset = raw_datasets["test"].map(
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

    # dataloader
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": num_beams,
    }

    # prepare with accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # test bar
    test_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)

    # metric
    accuracy = 0.0

    for step, batch in enumerate(test_dataloader):
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
        labels = accelerator.pad_across_processes(
            batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

        generated_tokens, labels = accelerator.gather_for_metrics(
            (generated_tokens, labels))
        generated_tokens = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()

        # replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        accuracy += sum([decoded_preds[i] == decoded_labels[i]
                        for i in range(len(decoded_preds))])

        test_bar.update(1)

    accelerator.print('accuracy : {}'.format(accuracy / len(raw_datasets['test'])))


def run():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='google-t5/t5-base', 
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
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
        '--num_workers',
        type=int,
        default=os.cpu_count(),  # 1, None, 32
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=4000,
        type=int,
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


    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


if __name__ == "__main__":

    run()
