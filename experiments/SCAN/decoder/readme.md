#### train
```bash
accelerate launch gemma2_scan.py --output_dir /path/to/output_dir --dataset_config length --model_name_or_path google/gemma-2-2b --train_steps 200000 --lr 1e-3 --mixed_precision no --per_device_train_batch_size 8 --per_device_eval_batch_size 8

accelerate launch gemma2_scan.py --output_dir /home/drdo/Caricatures/models/scan_gemma2-2b --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --eval_steps 50

accelerate launch gpt2_scan.py --output_dir /home/drdo/Caricatures/models/scan_gpt2 --lr 1e-3

```

