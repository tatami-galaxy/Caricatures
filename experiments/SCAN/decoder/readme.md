#### 4090
```bash
accelerate launch gemma_scan.py --output_dir /path/to/output_dir --dataset_config length --model_name_or_path google/gemma-2-2b --train_steps 200000 --lr 1e-3 --mixed_precision no --per_device_train_batch_size 8 --per_device_eval_batch_size 8

accelerate launch gemma_scan.py --output_dir /home/drdo/Caricatures/models/scan_gemma2-2b --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --lr 1e-4

accelerate launch gpt_scan.py --output_dir /home/drdo/Caricatures/models/scan_gpt2 --lr 1e-3

```

##### gpt2
```bash
accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2 --output_dir /root/Caricatures/models/scan_gpt2 --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8

accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2 --output_dir /home/drdo/Caricatures/models/scan_gpt2 --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2 --output_dir /home/drdo/Caricatures/models/scan_dummy_tokens_gpt2 --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --add_dummy_tokens

```

##### distilgpt2
```bash
accelerate launch gpt_scan.py --model_name_or_path distilbert/distilgpt2 --output_dir /root/Caricatures/models/scan_distilgpt2 --train_steps 200000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --add_dummy_tokens

```

##### gpt2 large, medium
```bash
accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2-large --output_dir /home/drdo/Caricatures/models/scan_gpt2-large --train_steps 200000 --eval_steps 10000 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --lr 1e-3
accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2-medium --output_dir /home/drdo/Caricatures/models/scan_gpt2-medium --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

```

#### A100
```bash
CUDA_VISIBLE_DEVICES=4 accelerate launch gpt_scan.py --model_name_or_path /home/ujan/LLMs/gpt2-large --local_dataset --dataset /home/ujan/Datasets/scan/scan_simple --output_dir /home/ujan/Caricatures/models/scan_gpt2-large --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

CUDA_VISIBLE_DEVICES=5 accelerate launch gpt_scan.py --model_name_or_path /home/ujan/LLMs/gpt2-medium --local_dataset --dataset /home/ujan/Datasets/scan/scan_simple --output_dir /home/ujan/Caricatures/models/scan_gpt2-medium --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

CUDA_VISIBLE_DEVICES=6 accelerate launch gpt_scan.py --model_name_or_path /home/ujan/LLMs/gpt2-large --local_dataset --dataset /home/ujan/Datasets/scan/scan_simple --add_action_tokens --output_dir /home/ujan/Caricatures/models/scan_gpt2-large --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

CUDA_VISIBLE_DEVICES=7 accelerate launch gpt_scan.py --model_name_or_path /home/ujan/LLMs/gpt2-medium --local_dataset --dataset /home/ujan/Datasets/scan/scan_simple --add_action_tokens --output_dir /home/ujan/Caricatures/models/scan_gpt2-medium --train_steps 100000 --eval_steps 5000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 

```