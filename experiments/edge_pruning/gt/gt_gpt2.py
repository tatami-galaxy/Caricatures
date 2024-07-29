from dataclasses import dataclass, field
import os, math

from datasets import(
    load_from_disk,
    load_dataset,
    DatasetDict,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import(
    HfArgumentParser,
    AutoTokenizer,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
)

from modeling_fpt2 import FPT2LMHeadModel

import torch
import torch.nn as nn
from torch.optim import AdamW

#import lovely_tensors as lt
#lt.monkey_patch()

class DataCollatorYear:
    def __init__(
        self, 
        tokenizer,
        max_length,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def __call__(self, examples):
        input_ids = []
        corr_input_ids = []
        labels = []         # need to pass something otherwise compute_metrics will not be called
        indices = []
        digits = []
        
        for example in examples:
            text = example["prefix"]
            corr_text = example["corr_prefix"]
            
            input_ids_example = self.tokenizer(text, return_tensors="pt").input_ids[0]
            corr_input_ids_example = self.tokenizer(corr_text, return_tensors="pt").input_ids[0]
            indices.append(input_ids_example.shape[0] - 1)
            input_ids_example = torch.nn.functional.pad(
                input_ids_example, 
                (0, self.max_length - input_ids_example.shape[0]), 
                value=self.tokenizer.pad_token_id
            )
            corr_input_ids_example = torch.nn.functional.pad(
                corr_input_ids_example,
                (0, self.max_length - corr_input_ids_example.shape[0]),
                value=self.tokenizer.pad_token_id
            )
            
            input_ids.append(input_ids_example)
            corr_input_ids.append(corr_input_ids_example)
            labels.append(torch.ones_like(input_ids_example) * -100)
            digits.append(int(example["digits"]))
        
        return {
            "input_ids": torch.stack(input_ids),
            "corr_input_ids": torch.stack(corr_input_ids),
            "labels": torch.stack(labels),
            "indices": torch.LongTensor(indices),
            "digits": torch.LongTensor(digits),
        }      


class FPT2InfoTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.0)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.gpt2_model = kwargs.pop('gpt2_model', None)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        
        self.digits = None
                
        super().__init__(*args, **kwargs)
        
        self.tokenizer = kwargs.pop('tokenizer', None)

    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) * 
                    global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (math.log(1 - self.target_edge_sparsity) - 
                    math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity
        
    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) * 
                    global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (math.log(1 - self.target_layer_sparsity) - 
                    math.log(1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.digits is None:
            self.digits = torch.LongTensor([self.tokenizer.encode("{:02d}".format(i))[0] for i in range(100)]).to(self.args.device)

        indices = inputs.pop("indices", None)
        digits_ = inputs.pop("digits", None)
        corr_input_ids = inputs.pop("corr_input_ids")
        input_ids = inputs.pop("input_ids")
        
        with torch.no_grad():
            # First get the logits from the GPT-2 model
            gpt2_logits = self.gpt2_model(input_ids=input_ids, **inputs).logits
            gpt2_logits = torch.gather(
                gpt2_logits, 
                1, 
                indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, gpt2_logits.shape[-1])
            ).squeeze()
            gpt2_logits = torch.gather(gpt2_logits, 1, self.digits.unsqueeze(0).repeat(gpt2_logits.shape[0], 1))
            gpt2_logits = torch.nn.functional.log_softmax(gpt2_logits, dim=-1)
            
            # Now run the corrupted inputs through it, and retain the activations
            corr_x = self.gpt2_model(input_ids=corr_input_ids, **inputs, output_writer_states=True).writer_states
        
        outputs = model(
            input_ids=input_ids,
            **inputs, 
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_layer_target_sparsity(self.state.global_step),
            corr_x=corr_x
        )
        
        # print(torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.memory_allocated() / 1024**3)
        
        reg_edge_loss = outputs["edge_loss"]
        if self.skip_layer_loss_if_higher_sparsity and outputs["model_node_sparsity"] > outputs["target_node_sparsity"]:
            reg_layer_loss = 0
        else:
            reg_layer_loss = outputs["node_loss"]
        reg_loss = reg_edge_loss + reg_layer_loss
        
        ## Restricting to 01-99 for now
        # Only the last position
        logits = torch.gather(outputs.logits, 1, indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs.logits.shape[-1])).squeeze()
        logits = torch.gather(logits, 1, self.digits.unsqueeze(0).repeat(logits.shape[0], 1))
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        kl_loss = nn.functional.kl_div(logits, gpt2_logits, reduction="batchmean", log_target=True)
        
        loss = kl_loss + reg_loss
        outputs["loss"] = loss
        outputs["kl_loss"] = kl_loss
        outputs["prob_digits"] = torch.nn.functional.softmax(logits, dim=-1)
        outputs["digits"] = digits_

        return (loss, outputs) if return_outputs else loss
    

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


def eval_fn(eval_pred):         
    (
        _, logits, reg_edge_loss, reg_layer_loss, target_edge_sparsity, target_layer_sparsity, model_edge_sparsity, model_layer_sparsity, 
        kl_loss, prob_digits, digits
    ) = eval_pred.predictions
    
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_layer_sparsity = model_layer_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_layer_sparsity = target_layer_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_layer_sparsity = model_layer_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_layer_sparsity = target_layer_sparsity.item()
    
    probability_difference = 0
    for i in range(digits.shape[0]):
        probability_difference += prob_digits[i, digits[i]+1:].sum() - prob_digits[i, :digits[i]].sum()
    probability_difference /= digits.shape[0]
    
    probability_difference_10 = 0
    for i in range(digits.shape[0]):
        probability_difference_10 += prob_digits[i, digits[i]+1:digits[i]+10].sum() - prob_digits[i, digits[i]-10:digits[i]].sum()
    probability_difference_10 /= digits.shape[0]
    
    kl_loss = kl_loss.mean().item()
    reg_edge_loss = reg_edge_loss.mean().item()
    reg_layer_loss = reg_layer_loss.mean().item()
    
    return {
        "eval_probability_difference": probability_difference,
        "eval_probability_difference_10": probability_difference_10,
        "model_edge_sparsity": model_edge_sparsity,
        "model_layer_sparsity": model_layer_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_layer_sparsity": target_layer_sparsity,
        "eval_kl_loss": kl_loss,
        "eval_reg_edge_loss": reg_edge_loss,
        "eval_reg_layer_loss": reg_layer_loss,
    }


def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0, disable_node_loss=False):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif ('sparsity_lambda_node' in n) and (not disable_node_loss):
            optimizer_4_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
                'lr': edges_lr,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,
                'lr': reg_edges_lr,
            },
            {
                'params': optimizer_3_group,
                'lr': layers_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_layers_lr,
            } 
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler


@dataclass
class CommonArguments:

    seed: int = field(default=2)
    repo_dir: str = field(default='/home/drdo/Caricatures')

    def __post_init__(self):
        pass


@dataclass
class TrainingArguments:  # inherit #

    mixed_precision: str = field(
        default="no",
        metadata={"help": "choose from : ['no', 'fp16', 'bf16', 'fp8']"}
    )
    gradient_accumulation_steps: int = field(default=1)
    max_steps: int = field(default=3000)
    warmup_steps: int = field(default=200)
    stop_optimizing_layer_if_higher_sparsity: bool = field(default=False)
    num_sparsity_warmup_steps: int = field(default=0)
    warmup_type: str = field(default="linear")
      
    def __post_init__(self):
        pass


@dataclass
class ModelArguments:

    model_name_or_path: str = field(default='openai-community/gpt2')
    with_embedding_nodes: bool = field(default=False)
    disable_linear_reg_term: bool = field(default=False)
    edge_learning_rate: float = field(default=1e-2)
    layer_learning_rate: float = field(default=1.0)
    reg_edge_learning_rate: float = field(default=1e-2,)
    reg_layer_learning_rate: float = field(default=1.0)
    disable_node_loss: bool = field(default=False)
    start_edge_sparsity: float = field(default=0.0)
    target_edge_sparsity: float = field(default=0.98)
    start_layer_sparsity: float = field(default=0.0)
    target_layer_sparsity: float = field(default=0.68)

    def __post_init__(self):
        pass


@dataclass
class DataArguments:

    dataset_path: str = field(
        default='/home/drdo/Caricatures/data/processed/edge_pruning/gt',
    )
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)
    train_split: str = field(default="train")
    max_seq_length:  int = field(
        default=64,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    
    def __post_init__(self):
        pass


def train(common_args, training_args, model_args, data_args, accelerator):

    accelerator.print('loading dataset from {}'.format(data_args.dataset_path))
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
        with_embedding_nodes=model_args.with_embedding_nodes,
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    freeze_all_except_pruning_params(prune_model)

    # check train and eval splits
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]

    # we don't have a validation dataset, so we'll just use the test dataset.
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]

    # data collator
    collator = DataCollatorYear(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
    
    # optimizers
    optimizers = get_optimizers(
        prune_model, 
        edges_lr=model_args.edge_learning_rate,
        layers_lr=model_args.layer_learning_rate,
        reg_edges_lr=model_args.reg_edge_learning_rate,
        reg_layers_lr=model_args.reg_layer_learning_rate,
        disable_node_loss=model_args.disable_node_loss,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
    )

    ## fix training_args ##

    # initialize Trainer
    trainer = FPT2InfoTrainer(
        model=prune_model,
        tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=model_args.start_edge_sparsity,
        target_edge_sparsity=model_args.target_edge_sparsity,
        start_layer_sparsity=model_args.start_layer_sparsity,
        target_layer_sparsity=model_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=training_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=training_args.num_sparsity_warmup_steps,
        warmup_type=training_args.warmup_type,
    )



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