## Source code for analyzing various dimensions of quality in broadcast and cable news

As used in "The content of television and radio news"

Written in a combination of python 2.7 and 3.7 (generally that which uses
the classifiers, CUDA, and deep learning is in python 3.7, and the rest can
be run in either, but in certain cases runs better in 2.7).

The mechanical turk portion is in html with javascript using bootstrap and
jquery.

Full implementation requires mongodb (used v2.6.10), although this might not
be necessary for everything.

Requires python packages:
 - numpy
 - pandas
 - pymongo
 - gensim
 - python-dateutil
 - scikit-learn
 - unidecode
 - bz2file
 - [fastai](https://docs.fast.ai) v1.0.42
 - matplotlib
 - seaborn

It provides modified versions of the [gsdmm](https://github.com/rwalk/gsdmm) and
[truecaser](https://github.com/nreimers/truecaser) libraries (as also
found under my account); the original [distributions.obj](https://github.com/nreimers/truecaser/releases/download/v1.0/english_distributions.obj.zip) file is needed to run the
truecaser. It can make use of the [libshorttext library (v1.1)](https://www.csie.ntu.edu.tw/~cjlin/libshorttext/).

Some of my data and results are available upon request.

If you plan to use or cite this repository in academic or commercial work, I 
would be interested to hear about it.