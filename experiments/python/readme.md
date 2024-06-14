##### train
accelerate launch t5_scan.py --from_scratch --output_dir `output_dir` --dataset_config length --model_name_or_path google-t5/t5-large

##### eval
accelerate launch t5_scan_eval.py --trust_remote_code --ignore_pad_token_for_loss --checkpoint `checkpoint` --dataset_config length
