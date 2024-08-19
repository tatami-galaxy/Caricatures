#### train
```bash
accelerate launch gemma_scan.py --output_dir /path/to/output_dir --dataset_config length --model_name_or_path google/gemma-2-2b --train_steps 200000 --lr 1e-3 --mixed_precision no --per_device_train_batch_size 8 --per_device_eval_batch_size 8

accelerate launch gemma_scan.py --output_dir /home/drdo/Caricatures/models/scan_gemma2-2b --per_device_train_batch_size 2 --per_device_eval_batch_size 2 

accelerate launch gpt_scan.py --output_dir /home/drdo/Caricatures/models/scan_gpt2 --lr 1e-3

accelerate launch gpt_scan.py --model_name_or_path openai-community/gpt2 --output_dir /root/Caricatures/models/scan_gpt2 --train_steps 50000 --eval_steps 1000 --per_device_train_batch_size 8 --per_device_eval_batch_size 8

accelerate launch gpt_scan.py --model_name_or_path distilbert/distilgpt2 --output_dir /root/Caricatures/models/scan_distilgpt2 --train_steps 50000 --eval_steps 1000 

```

