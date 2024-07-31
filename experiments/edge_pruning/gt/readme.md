accelerate launch gt_gpt2.py --output_dir `default is current dir` --resume_from_checkpoint `checkpoint dir` --load_last_checkpoint

example : 

accelerate launch gt_gpt2.py --output_dir /home/drdo/Caricatures/models/gpt2_gt --with_embedding_nodes
python gt_eval.py -m /home/drdo/Caricatures/models/gpt2_gt/checkpoint-3000 -w