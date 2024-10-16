# compositional-reasoning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Compositionality And Reasoning in Cognitive ArchitecTURES 

Collaboration: Ruchira Dhar and Ujan Deb

#### Setup

``` 
git clone https://github.com/tatami-galaxy/Caricatures.git
cd Caricatures
python repo_setup.py

cd ..
git clone https://github.com/tatami-galaxy/pyvene.git
cd pyvene
pip install .

accelerate config

wandb login
hugginface-cli login

curl https://rclone.org/install.sh
bash
rclone config
```
