#### SCAN finetuning : 
- models : t5, flan-t5, gpt2, distilgpt2
- code : 
    - `gpt_scan_test_gen_with_empty.py` under `/experiments/SCAN/decoder/`
    - `t5_scan.py` under `/experiments/SCAN/seq2seq/`


#### Causal model for SCAN : 
- `causal_model_v1_pyvene.py` under `/experiments/SCAN/causal_model/`
- `causal_model_v2_TD.py` under `/experiments/SCAN/causal_model/`

Each example in the SCAN dataset is aimed at converting a natural language command to a sequence of actions. 

$$ InputCommand \longrightarrow OutputSequence$$

Phrase Structure Grammar :

The input commands can be generated with a basic PS grammar starting from C and ending with U: 

1. C $\longrightarrow$ S and S
2. C $\longrightarrow$ S after S
3. C $\longrightarrow$ S
4. S $\longrightarrow$ V twice
5. S $\longrightarrow$ V thrice
6. S $\longrightarrow$ V
7. V $\longrightarrow$ D[1] opposite D[2]
8. V $\longrightarrow$ D[1] around D[2]
9. V $\longrightarrow$ D
10. V $\longrightarrow$ U
11. D $\longrightarrow$ U left
12. D $\longrightarrow$ U right
13. D $\longrightarrow$ turn left
14. D $\longrightarrow$ turn right
15. U $\longrightarrow$ walk
16. U $\longrightarrow$ run
17. U $\longrightarrow$ jump
18. U $\longrightarrow$ look

Where C=Full Command, S= Sentence Phrase, V= Verb Phrase, D= Direction Phrase, U= Verb

Compositional Abstraction Modelling :

Compositionality refers to the ability of compositional generalization i.e the ability to recognize the abstract underlying data structure, recovering the rules of abstraction and productively applying those rules in new contexts.

In the context of SCAN, a CAM (compositional abstraction model) should recover the phrase structure abstraction of the given input and apply it to parse new input and convert it into action sequences. Given such a form of abstraction where the model recovers phrase structure grammar, there can be two possible abstraction models: 

1. Top Down: The highest nodes like C are resolved and interpreted first
2. Bottom Up: The lowest nodes like U are resolved and interpreted first.

If the model has really been able to understand the compositional abstraction of the data in the form of PSG here, it should follow the sequence of PSG to resolve nodes and that can only be accomplished in one of two ways listed above. 


#### Intervention experiment :
- `gpt2_scan_interventions_v1.ipynb` under `/experiments/SCAN/interventions/`

Intervention experiments aim to test aligment between the causal or algorithmic solution and the network being analyzed. This is done using pairs or inputs and counterfactual inputs. To test if a network component is a causal abstraction of a casual variable, that is if they are aligned, the variable and network component values are replaced with their counterfactual values, all else being kept unchanged. For perfect alignment, the two interventions should always produce identical results. 

When the network is an encoder, the computation is a single forward pass through the model at t = 0. If the intervention occurs at a component at layer i, it can only influence computation at layer j > i.

In case when the network is an autoregressive decoder, the network has to generate one token per time step t, after the intervention on the initial input or prompt at layer i. Therefore at time t > 0 and layer j < i token representations that are causally influenced by the intervention also attends over input representations prior to the intervention. 