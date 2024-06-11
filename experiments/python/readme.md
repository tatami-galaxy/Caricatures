##### train
python t5_scan.py --trust_remote_code --ignore_pad_token_for_loss --from_scratch --output_dir `output_dir`

##### eval
python t5_scan_eval.py --trust_remote_code --ignore_pad_token_for_loss --model_name_or_path `checkpoint`
