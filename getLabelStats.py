#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Jan 2019
@author: Justin

#
# Loads a label data file produced by adjudicate.py (optionally as augmented by genLabelSets.py for synthetic categories)
#
# Outputs the number and percent of exemplars for each class in each label type
# Outputs the expected accuracies of a classifier that guesses among classes randomly or by using the class prevalence (only)
#
"""

import numpy as np
import json
import sys, os
from collections import defaultdict
from common import Label_DbFields, Synthetic_Category_Group_Names, Other_Synthetic_Group_Names


def expectedAccuracy(classProbs):
    '''return the accuracy obtained by choosing classes randomly with weights equal to the classes empirical probabilities'''
    # expected accuracy (classifier based only upon empirical distribution):
    # E(a) = sum(chance of choosing*chance of being correct for all cases/classes)
    # E(a) = P(a)*P(a) + P(b)*P(b) ; for equal prob 2 class: = 0.25 + 0.25 = 0.5
    # E(a) = P(C1)*P(C1) + P(C2)+P(C2) + ... + P(CN)*P(CN) ; for 3 classes with .8, .1, .1 probs: 0.64 + 0.01 + 0.01 = 0.66%
    return np.dot(classProbs, classProbs)


def printLabelStats(label_data):
    # gather stats about how many of each label there are
    counts = [defaultdict(int) for i in Label_DbFields + Synthetic_Category_Group_Names + Other_Synthetic_Group_Names]
    for label in label_data:
        labels = label[1]
        for field_idx, field in enumerate(Label_DbFields + Synthetic_Category_Group_Names + Other_Synthetic_Group_Names):
            if field in labels:
                counts[field_idx][labels[field]] += 1
    
    have_augmented_data = sum(counts[len(Label_DbFields)].values()) > 0

    # Present statistics for the label distribution for each db field
    label_counts = []
    precisions_preponderance = []
    fields_to_iterate = (Label_DbFields + Synthetic_Category_Group_Names + Other_Synthetic_Group_Names) if have_augmented_data else Label_DbFields
    for field_idx, field in enumerate(fields_to_iterate):
        total_count = sum(counts[field_idx].values())
        print("\n\nFor %s (%d labeled snippets):" % (field, total_count))
        label_probs = []
        for label in counts[field_idx].iterkeys():
            label_count = counts[field_idx][label]
            label_prob = label_count / float(total_count)
            label_probs.append( label_prob )
            print("    %20s: %4d exemplars (%2.2f)" % (label, label_count, (100.0 * label_prob)))
        precision_random = 1.0 / len(label_probs)
        precision_preponderance = expectedAccuracy(label_probs)
        label_counts.append(total_count)
        precisions_preponderance.append(precision_preponderance)
        print("Random-guess precision: %0.4f;  Naive field-preponderance precision: %0.4f\n" % (precision_random, precision_preponderance))

    print("# labeled")
    print(label_counts)
    print("field preponderence precisions")
    print(precisions_preponderance)


#TODO: fact-investigative, fact-non, opinion, other

def main(args):

    if len(args) < 2:
        print("getLabelStats [label_data file]")
        sys.exit(1)
    
    datafile_path = args[1]
    if not os.path.exists(datafile_path):
        print("%s not found" % datafile_path)
        sys.exit(2)
    
    with open(datafile_path) as f:
        label_data = json.load(f)

    print("Based upon %s" % datafile_path)
    printLabelStats(label_data)
    

if __name__ == "__main__":
    main(sys.argv)