conda create -n ytsa python=3.11 -y

conda activate ytsa

pip install -r requirements.txt

## DVC

dvc init 

dvc repro

dvc dag

## AWS

aws configure