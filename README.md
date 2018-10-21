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
- PyTorch 0.4
- [scikit-learn](http://scikit-learn.org/stable/)

## Markov Structure for Tagging

### Data
We use simplified CoNLL format as data input that contains four columns:
```
Token GoldTag PredTag Head
```
At training time only `Token` is used, `GoldTag` is used at testing time for evaluation. `PredTag` is the predicted unsupervised tags (for prediction), and `Head` represents the dependency head index (for evaluation of parsing task). Both `PredTag` and `Head` are not read in tagging task, we include them for consistency only.

As observations in our generative model, pre-trained word vectors are required. The input word vector map should be a pickled representation of Python dict object, but other formats of word vectors can also be read by easily modifying `markov_train.py`. 

We provide the pre-trained word vector file we used in the paper and a small subset of Penn Treebank data ([HERE](https://drive.google.com/open?id=1EXkzGjKnbIVUhVvI9wquSvp5B8gK4frX)) for testing the code. This dataset contains 10% samples of Penn Treebank and is public in [NLTK corpus](http://www.nltk.org/howto/corpus.html). Full Penn Treebank dataset requires a LDC license.

### Training

Train a Gaussian HMM baseline: 

```shell
python markov_train.py --model gaussian --train_file /path/to/train --word_vec /path/to/word_vec_file
```

By default we evaluate on the training data (this is not cheating in unsupervised learning case),  different test dataset can be specified by `--test_file` option. Training uses GPU when there is GPU available,  and CPU otherwise, but running on CPU can be extremely slow. Full configuration options can be found in `markov_train.py`.

Unsupervised learning is usually very sensitive to initializations, for this task we run multiple random restarts and pick the one with the highest training data likelihood as described in paper. It is generally sufficient to run 10 random restarts. The saved models are located in `./dump_models/markov`. When running with multiple random restarts, it is necessary to specify the `--jobid` or `--taskid` options to avoid model overwriting.

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
Coming soon.

## References
```
@inproceedings{he2018unsupervised,
    title = {Unsupervised Learning of Syntactic Structure with Invertible Neural Projections},
    author = {Junxian He and Graham Neubig and Taylor Berg-Kirkpatrick},
    booktitle = {Proceedings of EMNLP},
    year = {2018}
}
```

