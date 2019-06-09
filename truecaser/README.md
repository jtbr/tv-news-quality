From https://github.com/nreimers/truecaser and modified to run in python 3.
Only needed files are retained here. To run you must download
[distribution.obj](https://github.com/nreimers/truecaser/releases/download/v1.0/english_distributions.obj.zip)

# Language Independent Truecaser for Python
This is an implementation of a trainable Truecaser for Python.

A truecaser converts a sentence where the casing was lost to the most probable casing. Use cases are sentences that are in all-upper case, in all-lower case or in title case.

A model for English is provided, achieving an accuracy of 98.39% on a small test set of random sentences from Wikipedia.

# Model
The model was inspired by the paper of [Lucian Vlad Lita  et al., tRuEcasIng](https://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf) but with some simplifications.

The model applies a greed strategy. For each token, from left to right, it computes for all possible casing:

`score(w_0) = P(w_0) * P(w_0 | w_{-1}) * P(w_0 | w_1) * P(w_0 | w_{-1}, w_1)`

with `w_0` the word at the current position, `w_{-1}` the previous word, `w_1` the next word in the sentence.

All observed casings for `w_0` are tested and the casing with the highest score is selected.

The probabilities `P(...)` are computed based on a large training corpus.

# Requirements
The Code was written for Python 2.7 and requires NLTK 3.0.

From NLTK, it uses the functions to spilt sentences into tokens and the FreqDist(). These parts of the code can easily be replaced, so that the code can be used without NLTK install.

# Run the Code
You need a `distributions.obj` that contains information on the frequencies of unigrams, bigrams, and trigrams. One large `distributions.obj` for English is provided in the download section of github.

You can train your own `distributions.obj` using the `TrainTruecaser.py` script.

To run the code, have a look at `EvaluateTruecaser.py`. 

# Train your own Truecaser
You can retrain the Truecaser easily. Simply change the `train.txt` file with a large sample of sentences, change the `TrainTruecaser.py` such that is uses the `train.txt` and run the script. You can also it for other languages than English like German, Spanish, or French.
