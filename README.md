# struct-learning-with-flow

This is PyTorch implementation of the [paper](https://arxiv.org/abs/1808.09111):
```
Unsupervised Learning of Syntactic Structure with Invertible Neural Projections
Junxian He, Graham Neubig, Taylor Berg-Kirkpatrick
EMNLP 2018
```

The code performs unsupervised structure learning with flow, specifically on Markov structure and dependency structure.

## Requirements

- Python 3.6
- PyTorch >=0.4
- [scikit-learn](http://scikit-learn.org/stable/) (for tagging task only)
- [NLTK](https://www.nltk.org/) (for parsing task only)

## Data
Throughout two tasks we use simplified CoNLL format as data input that contains four columns:
```
ID Token Tag Head
```
At training time only `Token` is used, `Head` represents the dependency head index (for evaluation of parsing task). `Tag` is used for evaluation of tagging task.

As observations in our generative model, pre-trained word vectors are required. The input word2vec map should be a pickled representation of Python dict object.

We provide the pre-trained word vector file we used in the paper and a small subset of Penn Treebank data ([HERE](https://drive.google.com/open?id=18f61nN7l-Dvzqys7BypCsaCcj8gay7ip)) for testing the tagging code. This dataset contains 10% samples of Penn Treebank and is public in [NLTK corpus](http://www.nltk.org/howto/corpus.html). Full Penn Treebank dataset requires a LDC license.

## Markov Structure for Tagging

### Training

Train a Gaussian HMM baseline: 

```shell
python markov_train.py --model gaussian --train_file /path/to/train --word_vec /path/to/word_vec_file
```

By default we evaluate on the training data (this is not cheating in unsupervised learning case),  different test dataset can be specified by `--test_file` option. Training uses GPU when there is GPU available,  and CPU otherwise, but running on CPU can be extremely slow. Full configuration options can be found in `markov_flow_train.py`. After training the trained model will be saved in `dump_models/markov` directory.

Unsupervised learning is usually very sensitive to initializations, for this task we run multiple random restarts and pick the one with the highest training data likelihood as described in paper. It is generally sufficient to run 10 random restarts. When running with multiple random restarts, it is necessary to specify the `--jobid` or `--taskid` options to avoid model overwriting.

After training the Gaussian HMM, train a projection model with Markov prior:

```shell
python markov_train.py --model nice --train_file /path/to/train --word_vec /path/to/word_vec_file --load_gaussian /path/to/gaussian_model 
```

Initializing the prior with pre-trained Gaussian baseline would make the training much more stable. By default 4 coupling layers are used in NICE projection. 

### Results

On the provided subset of Penn Treebank that contains 3914 sentences, the Gaussian HMM is able to achieve ~76.5% M1 accuracy and ~0.692 VM score, and the projection model (4 layers) achieves ~79.2% M1 accuracy and ~0.718 VM score.

### Prediction

After training, prediction can be performed with :

```shell
python markov_train.py --model nice --train_file /path/to/tag_file --tag_from /path/to/pretrained_model
```

Here `--train_file` represents the file to be tagged, the output file is located in the current directory. 




## DMV Structure for Parsing
### Training

First train a vanilla DMV model with viterbi EM (this only runs on CPU):

```shell
python dmv_viterbi_train.py --train_file /path/to/train_data --test_file /path/to/test_data
```

Saved model is located in `dump_models/dmv/viterbi_dmv.pickle`.



Then use the pre-trained DMV to initialize the syntax model in flow/Gaussian model:

```shell
python dmv_fow_train.py --model nice --train_file /path/to/train_data --test_file /path/to/test_data --word_vec /path/to/word_vec_file --load_viterbi_dmv dump_models/dmv/viterbi_dmv.pickle
```

The script trains a Gaussian baseline when `--model` is specified as `gaussian`. Training uses GPU when there is GPU available,  and CPU otherwise. 



## References
```
@inproceedings{he2018unsupervised,
    title = {Unsupervised Learning of Syntactic Structure with Invertible Neural Projections},
    author = {Junxian He and Graham Neubig and Taylor Berg-Kirkpatrick},
    booktitle = {Proceedings of EMNLP},
    year = {2018}
}
```

