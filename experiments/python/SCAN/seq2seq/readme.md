##### train
accelerate launch t5_scan.py --from_scratch --output_dir `output_dir` --dataset_config length --model_name_or_path google/flan-t5-large --train_steps 200000 --lr 1e-5, `1e-3` --mixed_precision no --per_device_train_batch_size 8 --per_device_eval_batch_size 8

##### eval
accelerate launch t5_scan_eval.py --model_name_or_path `trained model name` --checkpoint `checkpoint` --dataset_config length
