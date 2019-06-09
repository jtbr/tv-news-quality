from collections import defaultdict, deque
from datetime import datetime
import pandas as pd
import random
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from common import Label_DbFields, Synthetic_Category_Group_Names, Other_Synthetic_Group_Names, MultiLabel_Group_Name, Labels, Stations

random.seed(42)

from fastai.text import *
bs = 256 #224 # size of minibatch
# written with fastai v1.0.48

classifiers_to_run = ['label_category', 'label_usforeign', 'factinvestigative', 'label_tone', 'label_emotion']

# this contains the labels for each classifier
label_set = Labels.copy()
label_set['factinvestigative'] = ['investigative', 'noninvestigative', 'opinion', 'other']

# contains the labels for which a classifier is relevant
relevant_col_names = {}
relevant_col_names['factinvestigative'] = ['elections_hard', 'elections_soft', 'business_economics', 'government', 'current_events', 'cultural']
relevant_col_names['label_usforeign'] = ['business_economics', 'government', 'current_events', 'sports', 'cultural']
relevant_col_names['label_tone'] = ['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'sports', 'products', 'anecdotes', 'current_events', 'cultural']
relevant_col_names['label_emotion'] = ['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'anecdotes', 'current_events', 'cultural']

def getIndices(names, nameMap):
    return tensor(sorted([nameMap.index(name) for name in names]))


modeldir = '/data/fastai-models/selected' # the second best auto-trained model among folds

# write headers for each output file
# (only needed the first time we run, in case this gets aborted midway)
for clas_name in classifiers_to_run:
    with open(clas_name + '_stats.csv', 'w') as stats_f:
        header = ["quarter_begin_date"]
        for station in ['overall'] + Stations:
            header += [station+'-'+label for label in (['total'] + label_set[clas_name])]
#        if clas_name in ['label_category', 'station']:
        for station in ['overall'] + Stations:
            header += [station+'-'+cn+'-top_k' for cn in (['sum'] + label_set[clas_name])]
        stats_f.write(",".join(header) + "\n")

dataset_rows = {}

quarters = []
for year in range(2010,2017):
    for month in ["01", "04", "07", "10"]:
        quarter = str(year) + '-' + month + "-01"
        quarters += [quarter]

for quarter in quarters:
    print('\n\n', str(datetime.now()), 'Reading in snippets for', quarter)
    # read in full population of (truecased, preprocessed) snippets for this quarter
    df = pd.read_csv('/data/' + quarter + '_snippets.tsv', sep='\t')

    dataset_rows = {} # what rows are relevant for a particular classifier (apart from label_category)
    for clas_name in classifiers_to_run:

        print('Processing', clas_name)
        
        # take subset of rows that are relevant for this classifier; both from the dataset and the original dataframe
        if clas_name == 'label_category':
            train_df = df
        else:
            train_df = df.iloc[dataset_rows[clas_name],:]

        print(' - loading databunch of size', len(train_df))
        
        learn = load_learner(modeldir, fname=clas_name + '_clas_fine_tuned.pkl',
                             test= TextList.from_df(train_df, cols='snippet'))

#        if clas_name == 'label_category':
#            learn = load_learner(modeldir, fname=clas_name + '_clas_fine_tuned.pkl',
#                                 test= TextList.from_df(train_df, cols='snippet'))
#            learn.data.save('quarter_temp_data') # need to save rather than leave in memory
#            df = df.drop('snippet', axis=1) # remove no-longer-needed column to save memory
#            train_df = train_df.drop('snippet', axis=1)
#        else:
#            learn = load_learner(modeldir, fname=clas_name + '_clas_fine_tuned.pkl')
#            loaded_data = load_data(modeldir, 'quarter_temp_data', bs=bs)
#            learn.data.test_dl = loaded_data.test_dl
#            loaded_data.test_dl = None; loaded_data = None
#            learn.data.test_ds.x.filter_subset(dataset_rows[clas_name])
#            learn.data.test_ds.y.filter_subset(dataset_rows[clas_name])
#            del dataset_rows[clas_name]

#        gc.collect()
        
        print(' - running classifier')
        preds, _ = learn.get_preds(ds_type=DatasetType.Test, ordered=True) # second return value would be true label (if this weren't a test set)

        print(' - analyzing and saving results')
        yhat = np.argmax(preds.numpy(), axis=1)
        labelcounts = [defaultdict(int) for s in ['overall'] + Stations]
        for i, station in enumerate(Stations):
            station_yhat = yhat[train_df['station'] == station]
            for label in station_yhat:
                labelcounts[i+1][label] += 1   # leave index 0 for the overall counts to calculate next

        print("    ", preds[:5])
#        print(labelcounts[0])

        # add 'overall' counts:
        for label_idx, _ in enumerate(learn.data.train_ds.classes):
            labelcounts[0][label_idx] = sum([labelcounts[i+1][label_idx] for i, __ in enumerate(Stations)])

        # translate from NN class index into class name, and make a full list of counts
        all_counts_ordered = []
        for i, station in enumerate(['overall'] + Stations):
            namedlabelcounts = defaultdict(int)
            total = 0
            for k,v in labelcounts[i].items():
                namedlabelcounts[learn.data.train_ds.classes[k]] = v
                total += v

            print("    ", station, namedlabelcounts)

            # label counts in order
            counts_ordered = [str(total)] + [("0" if total == 0 else str(float(namedlabelcounts[cn])/total)) for cn in label_set[clas_name]]
                
            all_counts_ordered += counts_ordered
        
        all_summed_likelihoods_ordered = []
#        if clas_name in ['label_category', 'station']:
        # tbd: how to handle binary classifiers: ads, transitions, nonsense, investigative (currently not being run)
        # calculate categories using top-k precision
        k = 3 if clas_name in ['label_category', 'station', 'supergroups'] else 2
        likelihoods, posns = preds.topk(k, dim=-1, sorted=False)
            
        # scale predictions so that top 3 likelihoods sum to 1
        norm_factors = 1. / likelihoods.sum(dim=-1)
        likelihoods = norm_factors * likelihoods.transpose(-1,0)
        likelihoods.transpose_(-1,0)
            
        overalllabelsums = defaultdict(float)
        overall_sum = 0.0
        for station in Stations:
            # allocate their normalized likelihoods to the 3 categories for each snippet
            likelihoods_sums = defaultdict(float)
            station_row_idxs = tensor((train_df['station'] == station).to_numpy().nonzero()[0])
            station_likelihood_rows = likelihoods.index_select(0, station_row_idxs)
            station_posns_rows = posns.index_select(0, station_row_idxs)
            for (snippet_lhs, snippet_posns) in zip(station_likelihood_rows, station_posns_rows):  #python 3: zip is an iterator (py2 use itertools.izip)
                for lh, posn in zip(snippet_lhs.tolist(), snippet_posns.tolist()):
                    likelihoods_sums[posn] += lh

            # order the likelihoods for reporting, and sum up overall totals
            namedlabelsums = defaultdict(float)
            for k,v in likelihoods_sums.items():
                namedlabelsums[learn.data.train_ds.classes[k]] = v
                overalllabelsums[learn.data.train_ds.classes[k]] += v
            station_sum = sum([namedlabelsums[cn] for cn in label_set[clas_name]])
            overall_sum += station_sum
            summed_likelihoods_ordered = [("0.0" if station_sum == 0. else str(namedlabelsums[cn]/station_sum)) for cn in label_set[clas_name]]
                
            all_summed_likelihoods_ordered += [str(station_sum)] + summed_likelihoods_ordered
            
        # prepend the overall total likelihoods (in order)  before the station totals
        overall_summed_likelihoods_ordered = [str(overall_sum)] + [str(overalllabelsums[cn]/overall_sum) for cn in label_set[clas_name]]
        all_summed_likelihoods_ordered = overall_summed_likelihoods_ordered + all_summed_likelihoods_ordered
        
        # append one line with counts for this learner in this quarter
        with open(clas_name + '_stats.csv', 'a') as stats_f:
            stats_f.write(",".join([quarter] + all_counts_ordered + all_summed_likelihoods_ordered) + "\n")

        # if this is the first classifier (label_category), save the subsets of df for other classifiers to run on
        if clas_name == 'label_category':
            # get the column indices in this learner for the classes for which subsequent classifiers are relevant
            relevant_col_idxes = {}
            for clas, col_list in relevant_col_names.items():
                relevant_col_idxes[clas] = getIndices(col_list, learn.data.train_ds.classes)
            
            # save rows to be classified for the remaining classifiers
            for clas, cols in relevant_col_idxes.items():
                relevant_scores = preds.index_select(1, cols)
                # indexes of rows (snippets) positive for relevant cols (having >0.5 probability among relevant scores)
                dataset_rows[clas] = (relevant_scores.sum(dim=-1) > 0.5).nonzero().squeeze(1).numpy()
           
            del relevant_col_idxes; del relevant_scores

        del learn; del yhat; del preds; del likelihoods; del station_likelihood_rows; del posns; del _
        gc.collect()
        #torch.cuda.empty_cache()
