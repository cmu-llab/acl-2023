# Transformed Protoform Reconstruction


## Installation
```pip install -r requirements.txt```
Then separately install torch (https://pytorch.org/get-started/locally/)

### Phylogeny probe dependencies

To run the phylogeny probe, you may need to manually install tqDist and PHYLIP.
* tqDist - used to calculate generalized quartet distance (GQD)

On a Linux machine,
```conda install -c bioconda tqdist```
or 
Follow the instructions on https://www.birc.au.dk/~cstorm/software/tqdist/

* PHYLIP consense - used to get a consensus tree
Install PHYLIP from https://evolution.genetics.washington.edu/phylip/ (use wget if needed)
Follow instructions at https://evolution.genetics.washington.edu/phylip/install.html
Copy ./consense to src/phylogeny/: ```cp PATH_TO_PHYLIP/phylip-3.695/exe/consense.app/Contents/MacOS/consense src/phylogeny```


## Running our code
Note: Do not run the code on Mac M1 silicon - the results will not hold. 

### Training
```
export WORK_DIR=.           
export SRC_DIR=src
export DATA_DIR=data
export CONF_DIR=conf

python src/train.py --conf chinese_baxter.json
python src/train.py --conf romance_orto_meloni.json
python src/train.py --conf romance_ipa_meloni.json
```

### Evaluation
```
export WORK_DIR=.           
export SRC_DIR=src
export DATA_DIR=data
export CONF_DIR=conf

python src/evaluate.py --conf romance_ipa_meloni
python src/evaluate.py --conf romance_orto_meloni
python src/evaluate.py --conf chinese_baxter
```
If you would like to run the evaluation with romance_ipa and romance_orto (the full datasets we use in our main results), please contact Ciobanu and Dinu (2014) for their data. 


Phylogeny probe:
```
python src/phylogeny_probe.py --model transformer --dataset romance_orto
python src/phylogeny_probe.py --model transformer --dataset romance_ipa
python src/phylogeny_probe.py --model transformer --dataset chinese_baxter
. ./src/phylogeny/phylogeny.sh
```
(don't forget the dot)


### RNN baseline (re-implementation of Meloni et al)
This PyTorch re-implementation of Meloni et al 2021 supersedes our previous re-implementation at https://github.com/cmu-llab/meloni-2021-reimplementation.


Training:
```
wandb login

export WORK_DIR=.
export DATA_DIR=data

python rnn/train_encoder_decoder_rnn.py \
    --dataset DATASET --save_predictions true \
    --lr \
    --beta1 \
    --beta2 \
    --encoder_layers 1 \
    --decoder_layers 1 \
    --embedding_size \
    --dim_feedforward \
    --dropout \
    --epochs \
    --lang_separators \
    --warmup_epochs \
    --batch_size \
    --wandb_name YOUR_WANDB_PROJECT \
    --wandb_entity YOUR_WANDB_ORG \
    --sweeping true
```
(note - do not set the batch size to anything besides 1 for *decoding*)


Evaluation:
```
python rnn/evaluate.py --model rnn --dataset romance_orto_meloni
python rnn/evaluate.py --model rnn --dataset romance_ipa_meloni
python rnn/evaluate.py --model rnn --dataset chinese_baxter
```
If you would like to run the evaluation with romance_ipa and romance_orto (the full datasets we use in our main results), please contact Ciobanu and Dinu (2014) for their data. 



Phylogeny probe:
```
python rnn/phylogeny_probe.py --model rnn --dataset romance_orto
python rnn/phylogeny_probe.py --model rnn --dataset romance_ipa
python rnn/phylogeny_probe.py --model rnn --dataset chinese_baxter
```


# Citing our paper

Please cite our paper as follows:

Young Min Kim, Kalvin Chang, Chenxuan Cui, and David Mortensen (forthcoming). Transformed Protoform Reconstruction. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*, Toronto, Canada.


```
@InProceedings{Kim-et-al:2023,
  author = {Kim, Young Min and Chang, Kalvin and Cui, Chenxuan and Mortensen, David R.},
  title = {Transformed Protoform Reconstruction},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)},
  year = {forthcoming},
  month = {July},
  date = {9--14},
  location = {Toronto, Canada},
}
```
