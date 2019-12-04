
# Stochastic Wasserstein Autoencoder for Probabilistic Sentence Generation

![](https://img.shields.io/badge/python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)

This is the official codebase for the following paper, implemented in tensorflow:

Hareesh Bahuleyan, Lili Mou, Hao Zhou, Olga Vechtomova. **Stochastic Wasserstein Autoencoder for Probabilistic Sentence Generation.** NAACL 2019. https://arxiv.org/pdf/1806.08462.pdf

## Overview
This package contains the code for two tasks
- SNLI Generation (`snli` : autoencoder models) 
- Dialog Generation (`dialog` : encoder-decoder models)

For the above tasks, the code for the following models have been made available:
1. Variational autoencoder (`vae`) /  Variational encoder-decoder (`ved`)
2. Deterministic Wasserstein autoencoder (`wae-det`) /  Deterministic Wasserstein encoder-decoder (`wed-det`)
3. Stochastic Wasserstein autoencoder (`wae-stochastic`) /  Stochastic Wasserstein encoder-decoder (`wed-stochastic`)

## Datasets
The models mentioned in the paper have been evaluated on two datasets:
 - [SNLI Sentences](https://nlp.stanford.edu/projects/snli/) 
 - [Daily Dialog](http://yanran.li/dailydialog.html) dataset

Additionally, the following dataset is also available to run dialog generation experiments:
 - [Cornell Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset

The data has been preprocessed and the train-val-test split is provided in the `data/` directory of the respective task.

## Requirements
- numpy==1.16.0
- pandas==0.22.0
- gensim==3.7.0
- nltk==3.2.3
- Keras==2.0.8
- tqdm==4.19.1
- tensorflow-gpu==1.3.0
- sklearn
- matplotlib

## Instructions
1. Create a virtual environment using `conda`
```
conda create -n nlg python=3.6.1
```
2. Activate virtual environment and install the required packages. 
```
source activate nlg
cd probabilistic_nlg/
pip install -r requirements.txt
```
3. Generate word2vec, required for initializing word embeddings (you would need to specify the dataset as argument for `dialog` generation task) :
```
cd snli/
python w2v_generator.py
```
4. Train the desired model, set configurations in the `model_config.py` file. For example,
```
cd wae-det
vim model_config.py # Make necessary edits or specify the hyperparams as command line arguments as below
python train.py --lstm_hidden_units=100 --vocab_size=30000 --latent_dim=100 --batch_size=128 --n_epochs=20 --kernel=IMQ --lambda_val=3.0
``` 
- The model checkpoints are stored in `models/` directory, the summaries for Tensorboard are stored in `summary_logs/` directory. As training progresses, the metrics on the validation set are dumped into`bleu_log.txt`  and `bleu/` directory. The model configuration and outputs generated during training are written to a text file within `runs/` 
5. Run`predict.py` specifying the desired checkpoint (`--ckpt`) to (1) generate sentences given test set inputs; (2) generate sentences by randomly sampling from the latent space; (3) linear interpolation between sentence in the latent space. 
By default for `vae` and `wae-stochastic`, sampling from latent space is carried out within one standard deviation from the mean <img src="https://latex.codecogs.com/svg.latex?\Large&space;z=\mu+\sigma\otimes\epsilon"/>. *Note* that `predict.py` also outputs the BLEU scores. Hence, when computing BLEU scores, it is ideal to simply use the mean <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mu"/> (i.e., no sampling) - for this, set the argument `--z_temp=0.0`.
The `random_sample_save(checkpoint, num_batches=3)` function call within `predict.py` automatically saves sentences generated by latent space sampling into `samples/sample.txt`

6. To compute the metrics for evaluating the latent space (AvgLen, UnigramKL, Entropy) as proposed in the paper, run `evaluate_latent_space.py` specifying reference sentence set path (i.e., training corpus) and generated sentence samples path (~100k samples is recommended). For example:
```
python evaluate_latent_space.py -ref='snli/data/snli_sentences_all.txt' -gen='snli/wae-det/samples/sample.txt'
```
## Citation
If you found this code useful in your research, please cite:
```
@inproceedings{probabilisticNLG2019,
  title={Stochastic Wasserstein Autoencoder for Probabilistic Sentence Generation},
  author={Bahuleyan, Hareesh and Mou, Lili and Zhou, Hao and Vechtomova, Olga},
  booktitle={Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2019}
}
```

This is added by Shiva!
