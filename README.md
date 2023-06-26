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
