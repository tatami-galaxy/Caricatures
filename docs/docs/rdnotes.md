03/07/2024

- create and clone branch locally : git clone --branch ruchiradhar-work https://github.com/tatami-galaxy/Caricatures
- create environment caricaturesenv
    - conda create --name caricaturesenv -y
    - conda install jupyter notebook
    - conda install ipykernel
    - python -m ipykernel install --user --name=caricaturesenv
    - conda install pytorch::pytorch torchvision torchaudio -c pytorch
    - conda install --file requirements.txt
- create requirements.yaml
    - conda env export > requirements.yaml
- push changes
    - git add .
    - git commit -m ""
    - git push 
- readings: [Inner workings of a Transformer](https://levelup.gitconnected.com/understanding-transformers-from-start-to-end-a-step-by-step-math-example-16d4e64e6eb1), [Key-Value Memories]()


13/07/2024
- Check environments: conda env list
- Created: [Pyvene Exploration Colab](https://colab.research.google.com/drive/1QZlOqEKFd334qKwzHyxCDmQywGHcCF_4#scrollTo=e08304ea)

22/07/2024
- Read all Geiger et al papers from 2018 to 2022. 
- Experiments with building causal model for SCAN

Todo: 
- Do: Adapt ReaSCAN work for SCAN
- Readings: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), [LMs Explain Neurons](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- Code: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

24/07/2024
- Merge everything from main: " git fetch origin"  gets you up to date with origin and then "git merge origin/master" to finally pull from master

29/07/2024
- Install pip in conda: !conda install pip -y
- Install pyvene in env: pip install git+https://github.com/stanfordnlp/pyvene.git ()
- Checked install path in python with: print(pyvene.__file__)

30/07/2024
- Creating Causal Model for SCAN in Experiements>Notebooks