# struct-learning-with-flow

This is PyTorch implementation of the [paper](https://arxiv.org/abs/1808.09111):
```
Unsupervised Learning of Syntactic Structure with Invertible Neural Projections
Junxian He, Graham Neubig, Taylor Berg-Kirkpatrick
EMNLP 2018
```

The code performs unsupervised structure learning with flow, specifically on Markov structure and dependency structure.

Please concact junxianh@cs.cmu.edu if you have any questions.

## Requirements

- Python 3
- PyTorch >=0.4
- [scikit-learn](http://scikit-learn.org/stable/) (for tagging task only)
- [NLTK](https://www.nltk.org/) (for parsing task only)

## Data
We provide the pre-trained word vector file we used in the paper and a small subset of Penn Treebank data for testing the tagging code. This dataset contains 10% samples of Penn Treebank and is public in [NLTK corpus](http://www.nltk.org/howto/corpus.html). Full Penn Treebank dataset requires a LDC license.

To download the sample data, run:
```shell
python prepare_data.py
```
The downloaded data is located in `sample_data`.

Throughout two tasks we use simplified CoNLL format as data input that contains four columns:
```
ID Token Tag Head
```
At training time only `Token` is used, `Head` represents the dependency head index (for evaluation of parsing task). `Tag` is used for evaluation of tagging task. As observations in our generative model, pre-trained word vectors are required. The input word2vec map should be a pickled representation of Python dict object.

We also provide script to preprocess full Penn Treebank dataset for parsing (e.g. converting parse trees, removing punctuations, etc.), the `wsj` directory should look like:
```
wsj
+-- 00
|   +-- wsj_0001.mrg
|   +-- ...
+-- 01
+-- ...

```
run:
```shell
python preprocess_ptb.py --ptbdir /path/to/wsj
```
This command would generate train/test files in `ptb_parse_data`. Note that the generated data files contain gold POS tags in the `Tag` column, thus are not the files we used in the paper, where the tags are induced from the Markov model. 

**TODO**: Simpify the pipline to generate train/test files without gold POS tags for parsing to reproduce the parsing results.

## Markov Structure for Tagging

### Training

Train a Gaussian HMM baseline: 

```shell
python markov_flow_train.py --model gaussian --train_file /path/to/train --word_vec /path/to/word_vec_file
```

By default we evaluate on the training data (this is not cheating in unsupervised learning case),  different test dataset can be specified by `--test_file` option. Training uses GPU when there is GPU available,  and CPU otherwise, but running on CPU can be extremely slow. Full configuration options can be found in `markov_flow_train.py`. After training the trained model will be saved in `dump_models/markov/`.

Unsupervised learning is usually very sensitive to initializations, for this task we run multiple random restarts and pick the one with the highest training data likelihood as described in paper. It is generally sufficient to run 10 random restarts. When running with multiple random restarts, it is necessary to specify the `--jobid` or `--taskid` options to avoid model overwriting.

After training the Gaussian HMM, train a projection model with Markov prior:

```shell
python markov_flow_train.py \
        --model nice \
        --lr 0.01 \
        --train_file /path/to/train \
        --word_vec /path/to/word_vec_file \
        --load_gaussian /path/to/gaussian_model 
```

Initializing the prior with pre-trained Gaussian baseline would make the training much more stable. By default 4 coupling layers are used in NICE projection. 

### Results

On the provided subset of Penn Treebank that contains 3914 sentences, the Gaussian HMM is able to achieve ~76.5% M1 accuracy and ~0.692 VM score, and the projection model (4 layers) achieves ~79.2% M1 accuracy and ~0.718 VM score.

### Prediction

After training, prediction can be performed with :

```shell
python markov_flow_train.py --model nice --train_file /path/to/tag_file --tag_from /path/to/pretrained_model
```

Here `--train_file` represents the file to be tagged, the output file is located in the current directory. 




## DMV Structure for Parsing
### Training

First train a vanilla DMV model with viterbi EM (this only runs on CPU):

```shell
python dmv_viterbi_train.py --train_file /path/to/train_data --test_file /path/to/test_data
```

Trained model is saved in `dump_models/dmv/viterbi_dmv.pickle`. Implementation of this basic DMV training is partially based on [this repo](https://github.com/davidswelt/dmvccm).



Then use the pre-trained DMV to initialize the syntax model in flow/Gaussian model:

```shell
python dmv_flow_train.py \
        --model nice \
        --train_file /path/to/train_data \
        --test_file /path/to/test_data \
        --word_vec /path/to/word_vec_file \
        --load_viterbi_dmv dump_models/dmv/viterbi_dmv.pickle
```

The script trains a Gaussian baseline when `--model` is specified as `gaussian`. Training uses GPU when there is GPU available,  and CPU otherwise. Trained model is saved in `dump_models/dmv/`.

## Acknowledgement
The awesome `nlp_commons` package (for preprocessing the Penn Treebank) in this repo was originally developed by Franco M. Luque and can be found in this [repo](https://github.com/davidswelt/dmvccm). 


## Reference
```
@inproceedings{he2018unsupervised,
    title = {Unsupervised Learning of Syntactic Structure with Invertible Neural Projections},
    author = {Junxian He and Graham Neubig and Taylor Berg-Kirkpatrick},
    booktitle = {Proceedings of EMNLP},
    year = {2018}
}
```

