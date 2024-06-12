##### train
python t5_scan.py --from_scratch --output_dir `output_dir` --dataset_config length

##### eval
python t5_scan_eval.py --trust_remote_code --ignore_pad_token_for_loss --checkpoint `checkpoint` --dataset_config length
