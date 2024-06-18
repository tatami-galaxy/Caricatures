##### train
accelerate launch t5_scan.py --from_scratch --output_dir `output_dir` --dataset_config length --model_name_or_path google-t5/t5-base --train_steps 200000 --lr 1e-5, `1e-3`

##### eval
accelerate launch t5_scan_eval.py --model_name_or_path `trained model name` --checkpoint `checkpoint` --dataset_config length
