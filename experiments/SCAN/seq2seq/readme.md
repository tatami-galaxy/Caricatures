#### train
```bash
accelerate launch t5_scan.py --from_scratch --output_dir /path/to/output_dir --dataset_config length --model_name_or_path google/flan-t5-large --train_steps 100000 --eval_steps 5000 --mixed_precision no --per_device_train_batch_size 8 --per_device_eval_batch_size 8
```

#### eval
```bash
accelerate launch t5_scan_eval.py --model_name_or_path pretrained_model_name --checkpoint /path/to/checkpoint --dataset_config length
