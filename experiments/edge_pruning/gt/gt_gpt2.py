from dataclasses import dataclass, field
import os

from datasets import(
    load_from_disk,
    load_dataset,
    DatasetDict,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import HfArgumentParser, AutoTokenizer

from modeling_fpt2 import FPT2LMHeadModel

import torch

#import lovely_tensors as lt
#lt.monkey_patch()


def load_datasets(dataset_path, max_train_samples, max_eval_samples, train_split="train"):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    if "validation" not in dataset:
        assert max_eval_samples is not None, "Validation set is missing! (val)"
        assert max_train_samples is not None, "Validation set is missing! (train)"
        dataset = DatasetDict({
            train_split: dataset[train_split].select(range(max_train_samples)),
            "validation": dataset[train_split].select(range(max_train_samples, max_train_samples+max_eval_samples)),
        })
    else:
        if max_train_samples is not None and max_train_samples < len(dataset[train_split]):
            dataset[train_split] = dataset[train_split].select(range(max_train_samples))
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]): 
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset


def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


@dataclass
class CommonArguments:

    seed: int = field(default=2)
    repo_dir: str = field(default='/users/ujan/caricatures')

    def __post_init__(self):
        pass


@dataclass
class TrainingArguments:

    mixed_precision: str = field(
        default="no",
        metadata={"help": "choose from : ['no', 'fp16', 'bf16', 'fp8']"}
    )
    gradient_accumulation_steps: int = field(default=1)

    def __post_init__(self):
        pass


@dataclass
class ModelArguments:

    model_name_or_path: str = field(default='openai-community/gpt2')
    with_embedding_nodes: bool = field(default=False)
    disable_linear_reg_term: bool = field(default=False)

    def __post_init__(self):
        pass


@dataclass
class DataArguments:

    dataset_path: str = field(
        default='/users/ujan/caricatures/data/processed/edge_pruning/gt',
    )
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)
    train_split: str = field(default="train")
    

    def __post_init__(self):
        pass


def train(common_args, training_args, model_args, data_args, accelerator):

    accelerator.print('loading dataset from {}'.formar(data_args.dataset_path))
    # load dataset
    raw_datasets = load_datasets(
        data_args.dataset_path,
        data_args.max_train_samples,
        data_args.max_eval_samples,
        data_args.train_split
    )
    accelerator.print('dataset loaded')
    n_train = len(raw_datasets["train"])

    accelerator.print('loading gpt2 models and tokenizer')
    # load gpt2 model
    # one model is to obtain behaviour(gt)
    # the other model is to prune and match the behaviour
    prune_model = FPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        with_embedding_nodes=model_args.with_embedding_nodes,
        disable_linear_regularization_term=model_args.disable_linear_reg_term,
    )
    gpt2_model = FPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        with_embedding_nodes=data_args.with_embedding_nodes,
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    freeze_all_except_pruning_params(prune_model)

    for name, param in prune_model.named_parameters():
        if param.requires_grad:
            print(name)


def run():

    # parse cl arguments
    parser = HfArgumentParser((CommonArguments, TrainingArguments, ModelArguments, DataArguments))
    common_args, training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(common_args.seed)

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=common_args.repo_dir+'/experiments/edge_pruning/gt',
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        #"lr": training_args.lr,
        #"train_steps": training_args.train_steps,
        "seed": common_args.seed,
        #"train_batch_size": training_args.per_device_train_batch_size,
    }
    # run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(common_args, training_args, model_args, data_args, accelerator)

    # end logging
    accelerator.end_training()


if __name__ == "__main__":
    run()