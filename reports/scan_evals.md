### SCAN

All trained on a single RTX 4090: 

| Split | Model | Pre-Trained | Eval at Step | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| simple | t5-base | Yes | 18k | 16 | 5e-5 | - | 0.979 |
| simple | t5-base | No | 18k | 16 | 5e-5 | - | 0.987 |
| simple | distilgpt2 | Yes | 19k | 16 | 5e-5 | - | 0.65 |
| simple | gpt2 | Yes | 60k | 8 | 5e-5 | - | 0.82 |
| simple | gpt2-large | Yes | 160k | 2 | 5e-5 | - | 0.66 |
| simple | gemma-2b | Yes | - | - | - | - | - |

| Split | Model | Pre-Trained | Eval at Step | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| length | t5-base | Yes | 25k | 16 | 5e-5 | 0.973 | 0.125 |
| length | t5-base | No | 60k | 16 | 1e-3 | 0.978 | 0.005 |
| length | flan-t5-base | Yes | 70k | 16 | 5e-5 | 0.991 | 0.138 |
| length | flan-t5-large | Yes | 75k | 8 | 5e-5 | 0.995 | 0.136 |
| length | gpt2 | Yes | 50k | 8 | 5e-5 | - | 0.01 |
| length | gemma-2b | Yes | - | - | - | - | - |

| Split | Model | Pre-Trained | Eval at Step | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| addprim_jump | t5-base | Yes | 50k | 16 | 5e-5 | 0.993 | 0.922 |
| addprim_jump | t5-base | No | 90k | 16 | 1e-5 | 0.994 | 0.001 |
| addprim_turn_left | t5-base | Yes | 35k | 16 | 5e-5 | 0.982 | 0.971 |
| addprim_turn_left | t5-base | No | 100k | 16 | 1e-5 | 0.992 | 0.858 |

