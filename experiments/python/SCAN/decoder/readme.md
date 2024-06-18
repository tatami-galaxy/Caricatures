##### train
accelerate launch gemma_scan.py --output_dir /home/drdo/Caricatures/models/scan_gemma-2b/ --dataset_config length --model_name_or_path google/gemma-2b --train_steps 200000 --lr 1e-5, `1e-3`

##### eval
accelerate launch t5_scan_eval.py --model_name_or_path `trained model name` --checkpoint `checkpoint` --dataset_config length
