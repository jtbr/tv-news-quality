#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Jan 2019
@author: Justin

#
# Loads a label data file produced by adjudicate.py 
#
# Creates train/test folds in a reproducibly random way (with a fixed random seed) for each label
# Generates ML-framework-compatible labeled data files for each fold's train and test sets
#
"""
import random as rand
import numpy as np
import json
import sys, os, re
from collections import defaultdict

from common import Label_DbFields, Synthetic_Category_Groups, Synthetic_Category_Group_Names, Other_Synthetic_Group_Names, MultiLabel_Group_Name

USE_TRUECASER = True
if USE_TRUECASER:
    from truecaser.Truecaser import getTrueCase
    try:
        import cPickle as pickle
    except:
        import pickle
    import nltk
    import nltk.data
    from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    detokenizer = Detok()


def getFolds(k, nSamples, rand_seed = None):
    '''Assign each sample to one of k folds. These can be used to create k independant (but overlapping) train-test sets,
       with a proportion of ~ (k-1)/k training data to 1/k test data'''

    indices = range(nSamples)

    if rand_seed:
        rand.seed(rand_seed)
    rand.shuffle(indices)

    return np.array([i % k for i in indices])


def trainTestSets(folds):
    '''Yields a tuple with the train set indices and test set indices for each of the folds'''
    k = np.max(folds) + 1

    for iFold in range(k):
        yield (np.where(folds != iFold)[0],   # train set for fold iFold
               np.where(folds == iFold)[0] )  # test set for fold iFold


def loadTrueCaserModel(model_filename):
    print('Loading truecaser model ...')
    if not os.path.exists(model_filename):
        model_filename = '../' + model_filename
    with open(model_filename, 'rb') as f:
        uniDist = pickle.load(f)
        backwardBiDist = pickle.load(f)
        forwardBiDist = pickle.load(f)
        trigramDist = pickle.load(f)
        wordCasingLookup = pickle.load(f)
    return (wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)


def trueCaseSnippets(snippets, model):
    '''true-case a list of (usually 3) snippets, and remove control characters within them'''
    truecase_snippets = []
    for part in snippets:
        part = re.sub(r"[\x00-\x0c]+", '', part) # remove control chars
        sentences = sent_detector.tokenize(part)
        trueCaseSentences = []
        for sentence in sentences:
            speech_start = False
            if len(sentence) > 2 and sentence[:2] == "> ":
                speech_start = True
                s = sentence[2:]
            else:
                s = sentence
            s = re.sub(r"&#[xX][\da-fA-F]+;", 'xxbell', s)
            tokens = nltk.word_tokenize(s.lower())
            tokens = getTrueCase(tokens, 'lower', *model)
            trueCaseSentence = detokenizer.detokenize(tokens)
            if speech_start:
                trueCaseSentence = "> " + trueCaseSentence          
            trueCaseSentences.append(trueCaseSentence)
        truecase_snippets.append(' '.join(trueCaseSentences))
    return truecase_snippets


def getRecAsLSTLabelData(label_pair, label_field):
    '''return as <label>\t<text>\n for libshorttext-style dataset'''
    snippets, labels = label_pair # snippets as list, labels as dict
    if (label_field in labels):
        text = " ".join(snippets) # " \\\\ " no longer put \\ between snippets
        return labels[label_field] + '\t' + text + '\n'
    else:
        return None


def main(args):

    if len(args) < 2:
        print("genLabelSets [label_data file]")
        sys.exit(1)
    
    datafile_path = args[1]
    if not os.path.exists(datafile_path):
        print("%s not found" % datafile_path)
        sys.exit(2)
    
    datafile_dir = os.path.dirname(datafile_path)
    datafile_name = os.path.basename(datafile_path)
    datafile_basename, ext = os.path.splitext(datafile_name)

    tc_model = None
    if USE_TRUECASER:
        tc_model = loadTrueCaserModel('truecaser/distributions.obj')

    with open(datafile_path) as f:
        label_data = json.load(f)
    

    # replace snippets with true-cased versions, and add multi-label field from original labels
    for (snippets, labels) in label_data:
        # replace snippets in-place with truecased snippets
        if USE_TRUECASER:
            snippets[:] = trueCaseSnippets(snippets, tc_model)

        # add a group with all the labels, comma separated
        labels[MultiLabel_Group_Name] = ','.join([labels[field] for field in labels.keys()])

    ## Augment labels with synthetic category labels and save them to disk
    # construct synthetic categories:
    for (snippets, labels) in label_data:
         # snippets as list, labels as dict (keys for each label group, values are the labels)
        try:
            labeled_category = labels['label_category']
        except:
            continue # skip record if there was no agreement on label_category and so it was excluded

        for synthetic_category, synthetic_category_labeldict in Synthetic_Category_Groups.iteritems():
            synthetic_label = None
            for (label, categories) in synthetic_category_labeldict.iteritems():
                if labeled_category in categories:
                    synthetic_label = label
                    break
            assert(synthetic_label) # otherwise some error in synthetic_category_groups
            # add synthetic label category to existing label_data set
            labels[synthetic_category] = synthetic_label
    
    for (snippets, labels) in label_data:
        # create a final special synthetic category for investigative/non-investigative/opinion/other
        factinvestigative_label = labels.get('label_factopinion', '')
        if factinvestigative_label == 'fact':
            factinvestigative_label = labels.get('label_investigative', '')

        if factinvestigative_label == '':
            continue # skip record if label_factopinion or label_investigative isn't there for some reason

        # add synthetic label category to existing label_data set
        labels[Other_Synthetic_Group_Names[0]] = factinvestigative_label

    # save augmented files (with synthetic category labels)
    augmented_datafile_name = datafile_basename + '-augmented.json'
    augmented_datafile_path = datafile_dir + '/' + augmented_datafile_name if datafile_dir else augmented_datafile_name
    with open(augmented_datafile_path, 'w') as f:
        json.dump(label_data, f)

    if datafile_name.find("holdout") >= 0:
        # for holdout data, don't generate folds, just a single data file.
        folds = np.zeros(len(label_data), dtype=np.int8)
    else:
        ## Rather than save the folds for future use with this label set, we use a fixed random seed to ensure we always get the same folds
        folds = getFolds(5, len(label_data), rand_seed = 42)  # for first run, rand_seed was 1

    print("Writing training and (training) validation data ... ")
    for iFold, sets in enumerate(trainTestSets(folds)):
        
        fold_path = "fold_%d" % iFold
        if datafile_dir:
            # write the train/test data in subdirs of the directory containing the label_data
            fold_path = datafile_dir + '/' + fold_path

        # iterate across label groups and synthetic label groups (which have been added to the label_data)
        for field in Label_DbFields + Synthetic_Category_Group_Names + Other_Synthetic_Group_Names + [MultiLabel_Group_Name]:
            field_path = fold_path + '/' + field
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            if (True):
                # libshorttext format files: <label><TAB><text>
                for train_test_idx, set_type in enumerate(['train', 'valid']):
                    with open(field_path + '/' + field + '_' + set_type + ".tsv", 'w') as f:
                        count = 0
                        for label_index in sets[train_test_idx]:
                            labeled_data_line = getRecAsLSTLabelData(label_data[label_index], field)
                            if labeled_data_line:
                                f.write(labeled_data_line)
                                count += 1
                        print("Fold %d, %s %s set has %d exemplars" % (iFold, field, set_type, count))

    print("Done.")


if __name__ == "__main__":
    main(sys.argv)