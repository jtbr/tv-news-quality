#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Oct 2018
@author: Justin

#
# Adjudicate.py can be used to do various things with the labeled snippets for given batches in the mongo database snippetlabels
# (you choose by commenting/uncommenting in main())
#
# 1) Show label pairs (highlighting differences), to vet Mturkers' work
# 2) Adjudicate labeled pairs by adding a tiebreaker record with your preferred answers
# 3) Label missing snippets (for if there are snippets from a batch file that aren't yet in the database; as would happen if you use MTurk sandbox)
# 4) Show confusion matrices and "agreement accuracies" from pairs of labels for snippets in the database
# 5) Output the "best" label for each label type for each snippet to a label file for the use of downstream algorithms.
#
# TODO: could compute cohen's kappa to evaluate trustworthiness of mturkers,
"""

import future
from builtins import input
import pymongo
import textwrap as tw
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from common import *

def request_corrected_label(field, label1, label2):
    val = input('? ')
    corrected_label = None

    if val == '1' or val == '':
        corrected_label = label1  # default
    elif val == '2':
        corrected_label = label2
    else:
        possible_labels = Labels[field]
        for l in possible_labels:
            if l.startswith(val):
                corrected_label = l
                break

        if val == 'na':
            corrected_label = 'na'

        if not corrected_label:
            # try again
            return request_corrected_label(field, label1, label2)
    
    return corrected_label


def request_label(field):
    label = None
    while not label:
        val = input('? ')
                        
        possible_labels = Labels[field]
        if val == 'na':
            label = 'na'
        elif len(val) == 0:
            continue
        else:
            try:
                i = int(val)
                if i< len(Labels[field]):
                    label = Labels[field][i]
            except:
                pass
            
            for l in possible_labels:
                if l.startswith(val):
                    label = l
                    break
        # try again if not yet labeled
    
    return label


def show_get_diffs(rec1, rec2, db_table = None):
    '''show records and differences in labels; if db_table is set, elicit corrections and insert new tiebreaker record'''
    #rec1 should be from workerId  A2FBHZUEK35JBP (me on worker sandbox)
    if rec2["WorkerId"] == JTB_WorkerID:
        rec1, rec2 = rec2, rec1  # swap so JTB is first

    w1Id = rec1['WorkerId']
    if w1Id == JTB_WorkerID:
        w1Id = "JTB"
        # OPTIONAL: don't re-label items JTB has labeled
        return

    assert(rec1['ccfn'] == rec2['ccfn'] and rec1['ccsampnum'] == rec2['ccsampnum'])
    print("\n\n\n\n%s sample %d\n\n" % (rec1['ccfn'], rec2['ccsampnum']))
    print("  %s\n\n  %s\n\n  %s\n" % 
        ('\n'.join(tw.wrap(rec1['priorsample'],50)), 
         '\n'.join(tw.wrap(rec1['sample'],50)), 
         '\n'.join(tw.wrap(rec1['nextsample'],50))))

    print('1: {:42} 2: {}\n'.format(' '.join([w1Id, str(rec1['_id'])]), ' '.join([rec2['WorkerId'], str(rec2['_id'])])))
    
    tiebreak_rec = {}
    tiebreak_rec['WorkerId'] = TIEBREAK_WorkerID
    tiebreak_rec['batch'] = rec1['batch']
    tiebreak_rec['holdout'] = rec1['holdout']
    tiebreak_rec['idx'] = rec1['idx']
    tiebreak_rec['ccfn'] = rec1['ccfn']
    tiebreak_rec['ccsampnum'] = rec1['ccsampnum']
    tiebreak_rec['station'] = rec1['station']
    tiebreak_rec['airdate'] = rec1['airdate']
    tiebreak_rec['priorsample'] = rec1['priorsample']
    tiebreak_rec['sample'] = rec1['sample']
    tiebreak_rec['nextsample'] = rec1['nextsample']

    print('  {:45} {}'.format(rec1.get('label_comment', ''), rec2.get('label_comment', '')))

    no_diffs = True
    # show fields, and (if updating) allow for changes
    for field in Label_DbFields:
        if field in rec1 or field in rec2:
            label1 = rec1.get(field, 'na')
            label2 = rec2.get(field, 'na')

            print('{} {:45} {}'.format("=" if label1 == label2 else " ", label1, label2))
            if db_table: # update
                if label1 == label2:
                    corrected_label = label1
                else:
                    no_diffs = False
                    corrected_label = request_corrected_label(field, label1, label2)

                if corrected_label != "na":
                    tiebreak_rec[field] = corrected_label
                # otherwise, leave the field unpopulated

    # if changing, show the tiebreaker record and request confirmation before adding it
    if db_table:
        print("")
        for field in Label_DbFields:
            print(tiebreak_rec.get(field, 'na'))
        val = 'y' if no_diffs else ''     # if no differences, tiebreaker auto-added from consensus values
        while val not in ['y','n','c']:
            val = input('update? (y)es/(n)o/(c)hange ')
        if val == 'y':
            db_table.insert(tiebreak_rec)
            print("added tiebreaker record\n")
        elif val == 'c':
            for field in Label_DbFields:
                label = tiebreak_rec.get(field, 'na')
                print(label)
                corrected_label = request_corrected_label(field, label, label)
                if corrected_label != "na":
                    tiebreak_rec[field] = corrected_label
                # otherwise, leave the field unpopulated
            db_table.insert(tiebreak_rec)
            print("added tiebreaker record\n")

    else:
        input('press enter to continue\n')


def show_get_labels(rec):
    '''show a record and elicit labels to put in the db'''
    print("\n\n\n\n%s sample %d; batch idx %d\n\n" % (rec['ccfn'], rec['ccsampnum'], rec['idx']))
    print("  %s\n\n  %s\n\n  %s\n" % 
        ('\n'.join(tw.wrap(rec['priorsample'],50)), 
         '\n'.join(tw.wrap(rec['sample'],50)), 
         '\n'.join(tw.wrap(rec['nextsample'],50))))

    new_rec = {}
    new_rec['WorkerId'] = JTB_WorkerID
    new_rec['batch'] = rec['batch']
    new_rec['holdout'] = rec['holdout']
    new_rec['idx'] = rec['idx']
    new_rec['ccfn'] = rec['ccfn']
    new_rec['ccsampnum'] = rec['ccsampnum']
    new_rec['station'] = rec['station']
    new_rec['airdate'] = rec['airdate']
    new_rec['priorsample'] = rec['priorsample']
    new_rec['sample'] = rec['sample']
    new_rec['nextsample'] = rec['nextsample']

    while True:
        # show fields, and (if updating) allow for changes
        for field in Label_DbFields:
            # in case we're retrying remove our old answers
            try:
                del new_rec[field]
            except:
                pass

            # ignore certain labels for certain categories
            if field != 'label_category' and new_rec['label_category'] not in Categories[field]:
                if field == 'label_usforeign' and new_rec['label_category'] in ['elections_soft', 'elections_hard']:
                    # implied field
                    new_rec['label_usforeign'] = 'domestic'
                continue
            if field == 'label_investigative' and 'label_factopinion' in new_rec and new_rec['label_factopinion'] != 'fact':
                # implied field (currently not used)
                #new_rec['label_investigative'] = 'noninvestigative'
                continue

            labelvals = dict(zip(range(len(Labels[field])), Labels[field]))
            print("%s: %s" % (field, labelvals))
            label = request_label(field)

            if label != "na":
                new_rec[field] = label
            # otherwise, leave the field unpopulated

        comment = input("any comment? ")
        if comment:
            new_rec['label_comment'] = comment
        #NOTE: leaving comment unpopulated in this case (as in tiebreaker)
        #else:
        #    new_rec['label_comment'] = '{}'

        # if changing, show the tiebreaker record and request confirmation before adding it
        print("")
        for field in Label_DbFields:
            print(new_rec.get(field, 'na'))
        print(new_rec.get('label_comment',''))
        val = None
        val = input('update ok? [y]es/(n)o ')
        if val != 'n':
            print("adding labeled record\n")
            return new_rec
        
        # otherwise try again


def getSnippetLabels(batch = None, test_data = False):
    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.labeldb
        msnippetlabels = mdb.snippetlabels
        # ensure we have an index on this table
        msnippetlabels.create_index([('ccfn', pymongo.ASCENDING), 
                                     ('ccsampnum', pymongo.ASCENDING),
                                     ('WorkerId', pymongo.ASCENDING)], unique=True)
        query = []
        
        ###To generate for all station-random labeled data (and comment out the next two ifs)
        #query.append({"$match": { 'batch': { "$in": [0, 1, 2]}}})

        if batch != None:
            query.append({"$match": { 'batch': batch }})

        if test_data:
            query.append({"$match": { 'holdout': True }})
        else:
            query.append({"$match": { 'holdout': False}})

        query.append({"$group": { "_id": { "ccfn": "$ccfn", "ccsampnum": "$ccsampnum" }}})

        snippets = msnippetlabels.aggregate(query)

        for snippet in snippets:
            #print(snippet)
            ccfn = snippet['_id']['ccfn']
            ccsampnum = snippet['_id']['ccsampnum']

            # could request count, but just getting them seems quicker
            label_cursor = msnippetlabels.find({ "ccfn": ccfn, "ccsampnum": ccsampnum })
            label_recs = []
            for label in label_cursor:
                label_recs.append(label)

            yield label_recs


def showLabelPairs(snippet_batch = None, test_data = False):
    print("Showing pairs in batch %s" % snippet_batch)
    for label_group in getSnippetLabels(snippet_batch, test_data):
        if len(label_group) == 2:           
            show_get_diffs(label_group[0], label_group[1])
        else:
            ccfn = label_group[0]['ccfn']
            ccsampnum = label_group[0]['ccsampnum']
            print("%s sample %d had %d label(s)" % (ccfn, ccsampnum, len(label_group)))


def adjudicateLabelPairs(snippet_batch = None, test_data = False):
    print("Assessing pairs in batch %s" % snippet_batch)
    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.labeldb
        msnippetlabels = mdb.snippetlabels
        
        for label_group in getSnippetLabels(snippet_batch, test_data):
            if len(label_group) == 2:
                #TEMP: only adjudicate where NONE of the labels match:
                #if not [labelname for labelname in Label_DbFields if labelname in label_group[0] and labelname in label_group[1] and label_group[0][labelname] == label_group[1][labelname]]:
                #TEMP: only adjudicate where the groupname's label doesn't match:
                groupname = "label_tone" #"label_category" 
                if groupname in label_group[0] and groupname in label_group[1] and label_group[0][groupname] != label_group[1][groupname]:
                    show_get_diffs(label_group[0], label_group[1], msnippetlabels)


# requires matplotlib
def displayConfusionMat(cm, category_names = None):
    rows, cols = np.shape(cm)
    if category_names == None and rows == 2:
        category_names = ['Negative','Positive']
    n = np.sum(cm)
    assert(category_names and len(category_names) == rows and rows == cols)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    #plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(rows)
    #TODO: add col and row sums
    plt.xticks(tick_marks, category_names, rotation=45)
    plt.yticks(tick_marks, category_names)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(rows):
        for j in range(rows):
            if rows == 2:
                plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
            elif rows < 6:
                plt.text(j, i, "%d (%2.0f%%)" % (cm[i][j], 100.*cm[i][j]/float(n)))
            else:
                plt.text(j, i, "%d" % cm[i][j])
    plt.show()


def getLabelStatistics(snippet_batch, test_data = False):
    confusion_mats = \
        dict(zip(Label_DbFields + Synthetic_Category_Group_Names, 
                 [np.zeros((15,15),'int'), np.zeros((3,3), 'int'), np.zeros((3,3), 'int'), np.zeros((2,2), 'int'), np.zeros((3,3), 'int'), np.zeros((3,3), 'int'), np.zeros((13, 13), 'int')] 
                 + [np.zeros((5,5), 'int'), np.zeros((3,3), 'int'), np.zeros((2,2), 'int'), np.zeros((2,2), 'int'), np.zeros((2,2), 'int')]))

    # Generate confusion matrices for each question type
    snippet_count = 0
    for label_group in getSnippetLabels(snippet_batch, test_data):
        # exclude workerId "TIEBREAKER" from statistics:
        label_group = [labelrec for labelrec in label_group if labelrec["WorkerId"] not in (TIEBREAK_WorkerID, JTB_WorkerID)]

        if len(label_group) >= 2: 
            snippet_count += 1
            for label in Label_DbFields:
                if label in label_group[0] and label in label_group[1]:
                    # only add to confusion matrix if both labellers have labeled this field
                    possible_labels = Labels[label]
                    label_A_idx = possible_labels.index(label_group[0][label])
                    label_B_idx = possible_labels.index(label_group[1][label])

                    confusion_mat = confusion_mats[label]
                    confusion_mat[(label_A_idx, label_B_idx)] += 1
                
                if label == 'label_category':
                    category_label_A = label_group[0]['label_category']
                    category_label_B = label_group[1]['label_category']
                    for synthetic_category_name, synthetic_category in Synthetic_Category_Groups.iteritems():
                        synthetic_label_A_idx = None
                        synthetic_label_B_idx = None
                        for synthetic_category_index, actual_labels in enumerate(synthetic_category.itervalues()):
                            if category_label_A in actual_labels:
                                synthetic_label_A_idx = synthetic_category_index
                            if category_label_B in actual_labels:
                                synthetic_label_B_idx = synthetic_category_index
                        assert(synthetic_label_A_idx >= 0 and synthetic_label_B_idx >= 0) # otherwise error in synthetic_category_groups

                        confusion_mat = confusion_mats[synthetic_category_name]
                        confusion_mat[(synthetic_label_A_idx, synthetic_label_B_idx)] += 1


    if snippet_count == 0:
        print("No snippets found. Batch correct?")
        return

    print("Assessed results pairs for %d snippets in batch %s" % (snippet_count, snippet_batch))

    # Display the confusion matrices
    label_accuracies = []
    for label in Label_DbFields + Synthetic_Category_Group_Names:
        confusion_mat = confusion_mats[label]
        n = np.sum(confusion_mat)
        t = np.sum([ confusion_mat[i][i] for i in range(np.shape(confusion_mat)[0]) ])
        #confusion_mat /= n
        
        if label in Labels:
            possible_labels = Labels[label]
        else:
            possible_labels = Synthetic_Category_Groups[label].keys()
        
        accuracy = t/float(n)
        label_accuracies.append(accuracy)
        print("\nConfusion matrix for %s, from %d snippets, %2.1f%% accurate" % (label, n, 100.*accuracy))
        
        confusion_df = pd.DataFrame(confusion_mat, columns = possible_labels, index = possible_labels)
        print(confusion_df)
        displayConfusionMat(confusion_mat, possible_labels)

    print('\nlabel accuracies')
    print(label_accuracies)

def snippetAndLabels(labelrec):
    snippets = [labelrec['priorsample'], labelrec['sample'], labelrec['nextsample']]
    labels = {labelname: labelrec[labelname] for labelname in Label_DbFields if labelname in labelrec}
    return (snippets, labels)


def generateLabeledData(test_data = False):
    """Returns all labeled data as tuples: list of snippets, along with dict of labels"""

    # Generate a consensus label for each label set (across all labeled data)
    labeled_data = []
    for label_group in getSnippetLabels(None, test_data): 
        if len(label_group) == 1:
            labeled_data.append(snippetAndLabels(label_group[0]))

        elif len(label_group) == 2:
            if label_group[0]["WorkerId"] in [JTB_WorkerID, TIEBREAK_WorkerID]:
                labeled_data.append(snippetAndLabels(label_group[0]))
            elif label_group[1]["WorkerId"] in [JTB_WorkerID, TIEBREAK_WorkerID]:
                labeled_data.append(snippetAndLabels(label_group[1]))
            else:
                # include BOTH labels (?) - NO; will only increase our error rate. Instead include only the matching labels
                #labeled_data.append(snippetAndLabels(label_group[0]))
                #labeled_data.append(snippetAndLabels(label_group[1]))

                # instead, only include labels that match between labelers:
                # BUT this may not have a label_category at all (which must be handled downstream)
                labelrec = {labelname : label_group[0][labelname] for labelname in Label_DbFields if labelname in label_group[0] and labelname in label_group[1] and label_group[0][labelname] == label_group[1][labelname]}
                if labelrec:
                    print("Saving unanimous portions of record from batch: %d" % label_group[0]['batch'])
                    labelrec['priorsample'] = label_group[0]['priorsample']
                    labelrec['sample']      = label_group[0]['sample']
                    labelrec['nextsample']  = label_group[0]['nextsample']
                    labeled_data.append(snippetAndLabels(labelrec))
                else:
                    print("Twice-labeled record with no unanimity!")

        elif len(label_group) == 3:
            tiebreaker_rec = next((rec for rec in label_group if rec["WorkerId"] in [TIEBREAK_WorkerID, JTB_WorkerID]), None)
            if tiebreaker_rec:
                labeled_data.append(snippetAndLabels(tiebreaker_rec))
            else:
                print("UNEXPECTED: 3 labeled records, none of which is JTB or Tiebreaker!!!")
        else:
            print("UNEXPECTED: more than 3 labeled records")

    if len(labeled_data) == 0:
        print("No snippets found!")
        
    return labeled_data


def labelMissingSnippets(snippet_batch, batch_file):
    '''AMTurk sandbox doesn't allow you to label a whole set; this second method of labelling allows you to fill in the gaps.
       Note: It only checks that a given snippet hasn't been labeled _by JTB_.
       
       Can also be used to do labelling separately from AMTurk.'''

    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.labeldb
        msnippetlabels = mdb.snippetlabels
        
        batch_idxs = msnippetlabels.find({'batch': snippet_batch, 'WorkerId': JTB_WorkerID}, ['idx'])
        #For adding missing records to a holdout set (missing from anybody):
        #batch_idxs = msnippetlabels.find({'batch': snippet_batch, 'holdout': True}, ['idx'])
        batch_idxs = set([rec['idx'] for rec in batch_idxs])
        
        batch = pd.read_csv(batch_file)
        assert(batch['batch'].iloc[0] == snippet_batch) # check we're modifying the right batch
        for row_idx, snippet in batch.iterrows():
            if snippet['idx'] not in batch_idxs:
                rec = show_get_labels(snippet)
                msnippetlabels.insert_one(rec)


def labelSelectedSnippets(batch_file, snippet_indices):
    '''Label snippets with the given snippet indices from the given batch file'''

    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.labeldb
        msnippetlabels = mdb.snippetlabels

        batch = pd.read_csv(batch_file)
        for row_idx, snippet in batch.iterrows():
            if not snippet_indices or snippet['idx'] in snippet_indices:
                rec = show_get_labels(snippet)
                msnippetlabels.insert_one(rec)


def main(args):
    SNIPPET_BATCH = 2
    TEST_DATA = False
    ## this file can perform various functions; just uncomment the one you want:

    # showLabelPairs(SNIPPET_BATCH, TEST_DATA)

    # adjudicateLabelPairs(SNIPPET_BATCH, TEST_DATA)
    
    # getLabelStatistics(SNIPPET_BATCH, TEST_DATA)

    # labelMissingSnippets(SNIPPET_BATCH, 'mturk_batches/train_set_2.csv')

    # labelSelectedSnippets('mturk_batches/train_set_2.csv', set([353, 297, 97, 513, 131, 470, 467, 218, 272, 160, 258, 93, 398, 165, 393, 115, 430, 471, 118, 261, 324, 215, 412, 418, 192, 121, 193, 216, 18, 320, 291, 187]))
    
    # data = generateLabeledData(TEST_DATA)
    # fn = 'labeled_data.json'
    # print("Total of %d labeled adjudicated snippets. Writing to %s." % (len(data), fn))
    # with open(fn, 'w') as f:
    #    json.dump(data, f)
   

if __name__ == "__main__":
    main(sys.argv)