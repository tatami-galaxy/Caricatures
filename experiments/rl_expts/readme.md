```bash
accelerate launch gpt2_scan_ppo.py --output_dir /path/to/output_dir --model_checkpoint /path/to/checkpoint --batch_size 64

accelerate launch gpt2_scan_ppo.py --output_dir /home/drdo/Caricatures/models/scan_distilgpt2_ppo --model_checkpoint /home/drdo/Caricatures/models/scan_distilgpt2/checkpoint-40000 --batch_size 64
accelerate launch gpt2_scan_ppo.py --output_dir /home/drdo/Caricatures/models/scan_distilgpt2_ppo --model_checkpoint /home/drdo/Caricatures/models/scan_distilgpt2_new/checkpoint-6000 --batch_size 64
```