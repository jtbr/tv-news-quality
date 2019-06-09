### This is a second run, compiling cross-class tendencies
# NOTE: There is some issue with memory/instability when running this way (there seems to be a torch/fastai bug)
#       May want to just load each classifier from disk
from collections import defaultdict, deque
from datetime import datetime
import pandas as pd
import random
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from common import Label_DbFields, Synthetic_Category_Groups, Synthetic_Category_Group_Names, Other_Synthetic_Group_Names, MultiLabel_Group_Name, Labels, Stations

# this contains the labels for each classifier
label_set = Labels.copy()
label_set['factinvestigative'] = ['investigative', 'noninvestigative', 'opinion', 'other']

from fastai.text import *
bs = 256 # size of minibatch
# written with fastai v1.0.48


modeldir = '/data/fastai-models/selected' # the second best auto-trained model among folds

hard_categories = Synthetic_Category_Groups['hardsoft']['hard']
us_foreign_categories = ['domestic', 'foreign', 'unknown']

# write headers for each output file
# (only needed the first time we run, in case this gets aborted midway)
#for name in ['hard_x_factopinion', 'hard_x_negative', 'hard_x_usforeign_x_scary']:
def writeCsvHeader(name, groups, secondary_groups = None):
    with open(name + '_stats-counts.csv', 'w') as stats_f:
        header = ["quarter_begin_date"]
        
        if secondary_groups:
            for group in groups:
                for station in ['overall'] + Stations:
                    header += [station+'-'+group+'-'+label for label in (['total'] + secondary_groups)]
        else:
            for station in ['overall'] + Stations:
                header += [station+'-'+label for label in (['total'] + groups)]

        stats_f.write(",".join(header) + "\n")

# actually write the headers
writeCsvHeader('hard_x_usforeign_x_scary', hard_categories, us_foreign_categories)
writeCsvHeader('hard_x_tone', hard_categories, label_set['label_tone'])
writeCsvHeader('hard_x_factinvestigative', hard_categories, label_set['factinvestigative'])


# learner_class_names = learn.data.train_ds.classes
# output_class_names = label_set[clas_name]
# k = 3 if clas_name in ['label_category', 'station', 'supergroups'] else 2
# #Stations defined globally
# station_col = df['station'] #for labeled snippets

def getPerStationStats(preds, learner_class_names, output_class_names, station_col, k):      
    yhat = np.argmax(preds, axis=1)
    labelcounts = [defaultdict(int) for s in ['overall'] + Stations]
    for i, station in enumerate(Stations):
        station_yhat = yhat[station_col == station]
        for label in station_yhat:
            labelcounts[i+1][label] += 1   # leave index 0 for the overall counts to calculate next

    # add 'overall' counts:
    for label_idx, _ in enumerate(learner_class_names):
        labelcounts[0][label_idx] = sum([labelcounts[i+1][label_idx] for i, __ in enumerate(Stations)])

    # translate from NN class index into class name, and make a full list of counts
    all_counts_ordered = []
    for i, station in enumerate(['overall'] + Stations):
        namedlabelcounts = defaultdict(int)
        total = 0
        for key,v in labelcounts[i].items():
            namedlabelcounts[learner_class_names[key]] = v
            total += v

        print("    ", station, namedlabelcounts)

        # label counts in order
        counts_ordered = [str(total)] + [str(namedlabelcounts[cn]) for cn in output_class_names]

        all_counts_ordered += counts_ordered
        
    all_summed_likelihoods_ordered = []

    # calculate categories using top-k precision
    likelihoods, posns = tensor(preds).topk(k, dim=-1, sorted=False)
            
    # scale predictions so that top 3 likelihoods sum to 1
    norm_factors = 1. / likelihoods.sum(dim=-1)
    likelihoods = norm_factors * likelihoods.transpose(-1,0)
    likelihoods.transpose_(-1,0)
            
    overalllabelsums = defaultdict(float)
    overall_sum = 0.0
    for station in Stations:
        # allocate their normalized likelihoods to the 3 categories for each snippet
        likelihoods_sums = defaultdict(float)
        station_row_idxs = tensor((station_col == station).to_numpy().nonzero()[0])
        station_likelihood_rows = likelihoods.index_select(0, station_row_idxs)
        station_posns_rows = posns.index_select(0, station_row_idxs)
        for (snippet_lhs, snippet_posns) in zip(station_likelihood_rows, station_posns_rows):  #python 3: zip is an iterator (py2 use itertools.izip)
            for lh, posn in zip(snippet_lhs.tolist(), snippet_posns.tolist()):
                likelihoods_sums[posn] += lh

        # order the likelihoods for reporting, and sum up overall totals
        namedlabelsums = defaultdict(float)
        for k,v in likelihoods_sums.items():
            namedlabelsums[learner_class_names[k]] = v
            overalllabelsums[learner_class_names[k]] += v
        station_sum = sum([namedlabelsums[cn] for cn in output_class_names])
        overall_sum += station_sum
        summed_likelihoods_ordered = [str(namedlabelsums[cn]) for cn in output_class_names]

        all_summed_likelihoods_ordered += [str(station_sum)] + summed_likelihoods_ordered

    # prepend the overall total likelihoods (in order)  before the station totals
    overall_summed_likelihoods_ordered = [str(overall_sum)] + [str(overalllabelsums[cn]) for cn in output_class_names]
    all_summed_likelihoods_ordered = overall_summed_likelihoods_ordered + all_summed_likelihoods_ordered

    return (all_counts_ordered, all_summed_likelihoods_ordered)


quarters = []
for year in range(2010,2017):
    for month in ["01", "04", "07", "10"]:
        quarter = str(year) + '-' + month + "-01"
        quarters += [quarter]

for quarter in quarters:
    print('\n\n', str(datetime.now()), 'Reading in snippets for', quarter)
    # read in full population of (truecased, preprocessed) snippets for this quarter
    df = pd.read_csv('/data/' + quarter + '_snippets.tsv', sep='\t')

    
    ### SAVE Topic categories
    print('\nProcessing topic categories for later use')        
    learn = load_learner(modeldir, fname='label_category_clas_fine_tuned.pkl',
                         test= TextList.from_df(df, cols='snippet'))

    print(' - running classifier')
    preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)
    
    print(' - analyzing results')
    yhat = np.argmax(preds.numpy(), axis=1)
    
    # set labels to index in hard_categories, or -1 if not hard (based upon top-1)
    hard_label_clas_idxs = [hard_categories.index(label) if label in hard_categories else -1 for label in learn.data.train_ds.classes]
    snippet_hard_labels = np.zeros((len(yhat),1), dtype=np.int8)
    for i, label in enumerate(yhat):
        snippet_hard_labels[i] = hard_label_clas_idxs[label]
        
    # only work with hard snippets from now on
    df = df[snippet_hard_labels>=0]
    snippet_hard_labels = snippet_hard_labels[snippet_hard_labels>=0]
    
    
    ### SAVE US-Foreign
    print('Processing us-foreign for later use')
    
    learn = load_learner(modeldir, fname='label_usforeign_clas_fine_tuned.pkl',
                         test= TextList.from_df(df, cols='snippet'))

    # save this preprocessed test data for use by the other learners
    test_dl = learn.data.test_dl
    df = df['station'] # only need this column from here on out
    
    print(' - running classifier')
    us_foreign_preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)
    
    print(' - analyzing results')
    assert(us_foreign_categories == learn.data.train_ds.classes)
    us_foreign_preds = us_foreign_preds.numpy()
    us_foreign_label_idxs = np.argmax(preds.numpy(), axis=1)
    

    ### OK, now do hard x [US, foriegn, other] x [scary]
    print('Processing emotion')
    learn = load_learner(modeldir, fname='label_emotion' + '_clas_fine_tuned.pkl')
    # use saved test data
    learn.data.test_dl = test_dl
    print(' - running classifier')
    preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)
    
    #yhat = np.argmax(preds.numpy(), axis=1)
    preds = preds.numpy()
    is_scary = preds[:,learn.data.train_ds.classes.index('scary')] > 0.35 # note this cutoff is likely more reasonable than argmax for this category
    
    station_scary = df[is_scary]
    #preds_scary = preds[is_scary]
    us_foreign_preds_scary = us_foreign_preds[is_scary]
    snippet_hard_scary_labels = snippet_hard_labels[is_scary]
    all_likelihoods = []
    for topic_idx, topic_class_name in enumerate(hard_categories):
        print ('  scary x', topic_class_name)
        #preds_scary_fortopic = preds_scary[snippet_hard_scary_labels == topic_idx]
        us_foreign_preds_topic_scary = us_foreign_preds_scary[snippet_hard_scary_labels==topic_idx]
        station_topic_scary = station_scary[snippet_hard_scary_labels==topic_idx]
        
        # learner_class_names = learn.data.train_ds.classes
        # output_class_names = label_set[clas_name]
        # k = 3 if clas_name in ['label_category', 'station', 'supergroups'] else 2
        # #Stations defined globally
        # station_col = df['station'] #for labeled snippets

        
        # across US-foreign last
        counts, likelihoods = getPerStationStats(us_foreign_preds_topic_scary, us_foreign_categories, 
                                                 us_foreign_categories, station_topic_scary, 2)
        
        # just keep likelihoods
        all_likelihoods = all_likelihoods + likelihoods
        
    # append one line with counts for this learner in this quarter
    with open('hard_x_usforeign_x_scary' + '_stats-counts.csv', 'a') as stats_f:
        stats_f.write(",".join([quarter] + all_likelihoods) + "\n")

    
    ### do hard x factinvestigative
    print('Processing factinvestigative')
    learn = load_learner(modeldir, fname='factinvestigative' + '_clas_fine_tuned.pkl')
    
    # use saved test data
    learn.data.test_dl = test_dl
    print(' - running classifier')
    preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)
    
    preds = preds.numpy()
    all_counts = []
    for topic_idx, topic_class_name in enumerate(hard_categories):
        print ('  ', topic_class_name)
        preds_topic = preds[snippet_hard_labels==topic_idx]
        station_topic = df[snippet_hard_labels==topic_idx]

        # across fact investigative for topic
        counts, likelihoods = getPerStationStats(preds_topic, learn.data.train_ds.classes, 
                                                 label_set['factinvestigative'], station_topic, 2)
        
        # just keep counts
        all_counts = all_counts + counts
        
    # append one line with counts for this learner in this quarter
    with open('hard_x_factinvestigative' + '_stats-counts.csv', 'a') as stats_f:
        stats_f.write(",".join([quarter] + all_counts) + "\n")

        
    ### do hard x tone (negative)
    print('Processing tone')
    learn = load_learner(modeldir, fname='label_tone' + '_clas_fine_tuned.pkl')
    
    # use saved test data
    learn.data.test_dl = test_dl
    print(' - running classifier')
    preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)
    
    preds = preds.numpy()
    all_likelihoods = []
    for topic_idx, topic_class_name in enumerate(hard_categories):
        print ('  ', topic_class_name)
        #preds_scary_fortopic = preds_scary[snippet_hard_scary_labels == topic_idx]
        preds_topic = preds[snippet_hard_labels==topic_idx]
        station_topic = df[snippet_hard_labels==topic_idx]

        # across fact investigative for topic
        counts, likelihoods = getPerStationStats(preds_topic, learn.data.train_ds.classes, 
                                                 label_set['label_tone'], station_topic, 2)
        
        # just keep likelihoods
        all_likelihoods = all_likelihoods + likelihoods
        
    # append one line with counts for this learner in this quarter
    with open('hard_x_tone' + '_stats-counts.csv', 'a') as stats_f:
        stats_f.write(",".join([quarter] + all_likelihoods) + "\n")


    del learn; del preds; del yhat; del df; del test_dl; del us_foreign_preds; del _
    gc.collect()
    
print('Done!')