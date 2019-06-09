#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Use fastai to train text classifiers for each label group in each fold of the specified directory'''

from fastai.text import *
bs = 128 # size of minibatch (32 on iceberg)
# written with fastai v1.0.48

from collections import defaultdict
model_path = Path('/data/fastai-models')
lang_model_path = model_path / 'language_model'
vocab = Vocab(pickle.load(open(lang_model_path/'itos.pkl','rb')))

path = Path('../labeled_data')

# load our labeled data into a TextClasDataBunch
# dir structure: labeled_data/fold_i/groupname/groupname_[train|valid].tsv for i in 0..4
fold_accuracies = defaultdict(list)
category_t3_accs = []
category_t5_accs = []
for fold in range(5):
    fold_name = 'fold_' + str(fold)
    for groupfilepath in (path/fold_name).ls():
        groupname = str(groupfilepath.parts[-1])
        print('Processing ' + fold_name + ' ' + groupname + '\n\n')
        
        # load the training and validation data for this group name and fold
        train_df = pd.read_csv(path/fold_name/groupname/(groupname+'_train.tsv'), header=None, delimiter='\t', names=['label','text'])
        valid_df = pd.read_csv(path/fold_name/groupname/(groupname+'_valid.tsv'), header=None, delimiter='\t', names=['label','text'])

        # settings for single vs multi-label learners
        top3accuracy = partial(top_k_accuracy, k=3)
        top5accuracy = partial(top_k_accuracy, k=5)
        metrics = [accuracy]
        label_delim = None
        initial_lr = 1e-2
        if groupname == 'multilabel':
            metrics = [accuracy_thresh] # accuracy per label at default threshold of 0.5
            label_delim = ','
            initial_lr *= 4
        elif groupname == 'label_category':
            metrics += [top3accuracy, top5accuracy]
        
        data_clas = TextClasDataBunch.from_df(model_path/fold_name, train_df, valid_df, vocab=vocab, 
                                              text_cols='text', label_cols='label', bs=bs,
                                              label_delim=label_delim)
        #data_clas.save('fold_'+fold+'-'+groupname+'_clas')
        
        learn = text_classifier_learner(data_clas, AWD_LSTM, metrics=metrics, drop_mult=0.6)
        learn.load_encoder(lang_model_path/'lm_fine_tuned3_enc')
        learn.freeze()
        learn.fit_one_cycle(2, initial_lr, moms=(0.8,0.7))
        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(initial_lr/(2.6**4), initial_lr), moms=(0.85,0.75))
        learn.freeze_to(-3)
        learn.fit_one_cycle(1, slice(initial_lr*0.5/(2.6**4), initial_lr*0.5), moms=(0.85,0.75))
        learn.unfreeze()
        learn.fit_one_cycle(2, slice(initial_lr*0.1/(2.6**4), initial_lr*0.1), moms=(0.8,0.7))
        learn.export(groupname+'_clas_fine_tuned.pkl')
        
        # save accuracies for later use
        fold_accuracies[groupname].append(float(learn.recorder.metrics[-1][0]))
        if groupname == 'label_category':
            category_t3_accs.append(float(learn.recorder.metrics[-1][1]))
            category_t5_accs.append(float(learn.recorder.metrics[-1][2]))

print(fold_accuracies)
print(category_t3_accs)
print(category_t5_accs)

print("Cross-fold cross-validation statistics:")
for labelgroup, accuracies in fold_accuracies.items():
    print("For %s" % labelgroup)
    print(accuracies)
    accuracies = np.array(accuracies)
    print("means %02f, SDs %02f\n" % (100.*accuracies.mean(axis=0), 100.*accuracies.std(axis=0)))