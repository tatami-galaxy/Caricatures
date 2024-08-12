import os
import subprocess


def main():

    req = ['transformers', 'datasets', 'accelerate', 'evaluate', 'tensorboard']


    # directorries
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/processed')

    os.mkdir('models')

    # install packages
    for package in req:
        subprocess.run(["pip", "install", package])

    # huggingface-cli login
    # accelerate config


if __name__ == "__main__":
    main()
