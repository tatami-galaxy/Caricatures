### SCAN finetuning : 
- models : t5, flan-t5, gpt2, distilgpt2

### Causal model for SCAN : 

Removed examples with 'turn' used as a verb to the simply causal model

Each example in the SCAN dataset is aimed at converting a natural language command to a sequence of actions. 

$ InputCommand \longrightarrow OutputSequence$

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


### Intervention experiments :

Intervention experiments aim to test aligment between the causal or algorithmic solution and the network being analyzed. This is done using pairs or inputs and counterfactual inputs. To test if a network component is a causal abstraction of a casual variable, that is if they are aligned, the variable and network component values are replaced with their counterfactual values, all else being kept unchanged. For perfect alignment, the two interventions should always produce identical results. 

#### Encoder : 
- When the network is an encoder, the computation is a single forward pass through the model at t = 0. If the intervention occurs at a component at layer i, it only influences computation at layer j > i. Here we need to assume that the computation we are interested in is localized in the intervened network component. In reality this assumption might not be true and the computation might be distributed in other components in the same or adjacent layers (Aside: Does the type of positional embedding have an impact on this?). The hypothesis is that the more the computation "leaks out" of the intervened component, the worse the alignment will be.  

#### Decoder : 
- In case when the network is an autoregressive decoder, it has to generate one token per time step, after the intervention on the initial input or prompt at layer i. Therefore the computation can not only leak into other components or layers, but also across time steps. Also if the computation occurs at t = t1 > 0, then intervention at t = 0 will not lead to any alignment. To search for this, for a given causal variable we could generate (input, counterfactual input) pairs where the computation corresponding to the causal variable can realistically occur at t > 0. For example : 

    `base input = turn right and jump twice`

    `source input = turn right and jump thrice`

    Here the computation corresponding to resolving `twice/thrice` can in theory happen after the decoder has generated the actions for `jump`. In that case the intervention needs to be after that generation time step.
    
    If we are only interested in top-down or bottom-up parses, we can rule out interventions at certain timesteps. For example in the top-down parse, the resolution of `and/after` is the first computation. Therefore if we're testing for top-down alignment, we should only search for this computation at `t = 0`.
    
    Can we also rule out interventions based on auto-regression? For example for the same `and\after`, if the model is doing auto-regressive generation it must resolve this before it emits the first token. Otherwise if its an `after` the model will not be able to recover if it starts by emitting action tokens for the first part of the command. Therefore this resolution must happen at `t = 0`.
    
    Can we do this systematically for all the variables we are interested in for both top-down and bottom-up parses? Given a variable/computation can we algorithmically generate the potential time steps this computation, or a part of it, can occur? The constraints are auto-regression and the fixed ordering of top-down and bottom-up parses. 

    ##### Bottom-Up :
    - For bottom-up parse `and\after` resolution is the last computation. Therefore all other computations must happen at `t = 0`. **We only compute loss for generation, not for context.** This is because we are not modeling how the context (command) output should change given an intervention on the context. Our causal model only predicts how the output actions change with an intervention. Since we need to compute a language modeling loss we use the expected output actions after the intervention to get a cross entropy loss. Can we use PPO or REINFORCE here to optimize a non differentiable objective between the `generated actions after intervention` and the `expected generated actions after intervention`?
    * `base_inputs` = `base_command` + `expected generated actions after intervention`
    * `source_inputs` = `source_command` + `expected generated actions after intervention`
    * `labels` = `expected generated actions after intervention`

    ##### Top-Down : 
    - `and\after` resolution is the first computation which much happen at `t = 0`. What other time steps do we search for the other computations? The last computation is `verb` resolution. In the output actions, if we ignore the turn verb (since turn as a verb is not explicitly represented), the first verb is at most the `3rd action` (ex : jump opposite). Therefore if the model indeed implements the top-down parse then all the resolutions should be done before emitting the `3rd action` where the exact value of `t` depends on the tokenizer. Let's say we want to intervene at `t = n`. **We need to make sure that the action output for** `0 < t < n` **is identical even after the intervention.** Otherwise during intervention we need to "correct" the internal representations computed for `0 < t < n`.  
    * `base_inputs` = `base_command` + `expected generated actions after intervention`
    * `source_inputs` = `source_command` + `expected generated actions after intervention`
    * `labels` = `expected generated actions after intervention for t >= n`

    If there is no alignment with both top-down or bottom-up parses can we claim that the model learns an algorithm which is not strictly compositional? Can we force the model to learn either the top-down or the bottom-up parse perhaps with IIT? Does that help in for example length generalization? Does RASP help?

    ##### Causal Model
    The causal model starts its computation at the leaves, goes up the tree computing each node and produces the final output at the root. When the network is an encoder, the causal model and the computation graph can be quite similar. The network also computes representations bottom up from the lowest layer to produce the final layer embeddings. The output will be in this final layer, perhaps trained to be in a special token such as the CLS token.

    When model is a decoder however, the final output is not produced after a single forward pass. Its generated one token at a time with each forward pass through the model. Therefore the final layer at `t = 0` or at `t << N` will not contain the final output representation. 

    ##### and/after resolution
    How to represent this in a node in case of the top-down parse?


    ##### Steps:

    - Train network to solve task / Prompt tune for LLM 
    - Select parse
    - Identify variables to align
    - Select network component to align
    - Select timesteps
    - Generate (input, counterfactual input) pairs
    - Train boundless DAS
    - Validate

- Does the "leaking" of the computation into other layers have a greater impact on alignment in case of decoders? This might be true because if there is a leak across time steps then that might imply that the leak across layers or components gets compounded. 

- Can circuit discovery effectively deal with this "leaking" by pruning the search space enough such that the alignment values are still useful to draw conclusions?