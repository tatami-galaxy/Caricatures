### SCAN

All trained on a single RTX 4090: 

| Split | Model | Pre-Trained | Steps | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| simple | t5-base | Yes | 18k | 16 | 5e-5 | - | 0.979 |
| simple | t5-base | No | 18k | 16 | 5e-5 | - | 0.987 |

| Split | Model | Pre-Trained | Steps | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| length | t5-base | Yes | 25k | 16 | 5e-5 | 0.973 | 0.125 |
| length | t5-base | No | 60k | 16 | 1e-3 | 0.978 | 0.005 |
| length | flan-t5-base | Yes | 70k | 16 | 5e-5 | 0.991 | 0.138 |
| length | flan-t5-large | Yes | 75k | 8 | 5e-5 | 0.995 | 0.136 |

| Split | Model | Pre-Trained | Steps | Batch Size | Learning Rate | Eval Accuracy ☨ | Test Accuracy ☨ |
| ------- | ------- | ------ | -------- | -------- | --------- | ------ | ----- |
| addprim_jump | t5-base | Yes | 50k | 16 | 5e-5 | 0.993 | 0.922 |
| addprim_jump | t5-base | No | - | 16 | 5e-5 | - | - |
| addprim_turn_left | t5-base | Yes | - | 16 | 5e-5 | - | - |

