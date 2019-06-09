# -*- coding: utf-8 -*-
"""
Created on Feb 4 2019

Common functions and data for News classification project

@author: Justin
"""

MONGODB_PORT = 25541

Stations = ['CNN', 'CNBC', 'ALJAZAM', 'BLOOMBERG', 'MSNBC', 'FOXNEWS', 'FBC', 'PBS', 'ABC', 'NBC', 'CBS', 'NPR', 'BBC'] #'COM', 'LINKTV',

# needed because the keys of the Labels dict would not be ordered
Label_DbFields = ['label_category', 'label_usforeign', 'label_factopinion', 'label_investigative', 'label_tone', 'label_emotion', 'station']

Labels = {} # keys are database fields; values are lists of Labels for that question
Labels['label_category'] = ['transitions',  'elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'sports', 'weather', 'products', 'anecdotes', 'current_events', 'cultural', 'ads', 'none']
Labels['label_usforeign'] = ['domestic', 'unknown', 'foreign']
Labels['label_factopinion'] = ['fact', 'opinion', 'other']
Labels['label_investigative'] = ['noninvestigative', 'investigative']
Labels['label_tone'] = ['positive', 'neutral', 'negative']
Labels['label_emotion'] = ['scary', 'neither', 'pleasant']
Labels['station'] = Stations

Categories = {} # label categories for which the given label applies
Categories['label_usforeign'] = ['business_economics', 'government', 'current_events', 'sports', 'cultural']
Categories['label_factopinion'] = ['elections_hard', 'elections_soft', 'business_economics', 'government', 'current_events', 'cultural']
Categories['label_investigative'] = ['elections_hard', 'elections_soft', 'business_economics', 'government', 'current_events', 'cultural']
Categories['label_tone'] = ['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'sports', 'products', 'anecdotes', 'current_events', 'cultural']
Categories['label_emotion'] = ['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'anecdotes', 'current_events', 'cultural']

# hard: meaningful/educational for the body politic
Synthetic_Category_Group_Names = ['supergroups', 'hardsoft', 'ads', 'transitions', 'nonsense']
Synthetic_Category_Groups = { # dict of dicts of label lists
    'supergroups': {'business_tech_economy': ['business_economics', 'products', 'science_tech'], 'government_elections': ['elections_hard', 'elections_soft', 'government'], 
                    'current_affairs': ['current_events', 'anecdotes', 'cultural', 'weather'], 'entertainment': ['entertainment', 'sports'],
                    'other': ['ads', 'transitions', 'none']},
    'hardsoft': {'hard': ['elections_hard', 'business_economics', 'science_tech', 'government', 'current_events', 'cultural'], 
                 'soft': ['elections_soft', 'entertainment', 'sports', 'weather', 'products', 'anecdotes'], 
                 'other_unknown': ['ads','transitions','none']},
    'ads' : {'ads': ['ads'], 'other': ['transitions', 'products', 'weather', 'science_tech', 'entertainment', 'sports', 'business_economics', 'government', 'elections_hard', 'elections_soft', 'current_events', 'cultural', 'anecdotes', 'none']},
    'transitions': {'transitions': ['transitions'], 'other': ['ads', 'products', 'weather', 'science_tech', 'entertainment', 'sports', 'business_economics', 'government', 'elections_hard', 'elections_soft', 'current_events', 'cultural', 'anecdotes', 'none']},
    'nonsense': {'uncategorizable': ['none'], 'categorizable': ['transitions', 'ads', 'products', 'weather', 'science_tech', 'entertainment', 'sports', 'business_economics', 'government', 'elections_hard', 'elections_soft', 'current_events', 'cultural', 'anecdotes']}
}

# fact-investigative, fact-noninvestigative, opinion, other group
# and multilabel, with all the original labels
# here apart because it doesn't derive from label_category
Other_Synthetic_Group_Names = ['factinvestigative']
MultiLabel_Group_Name = 'multilabel'

JTB_WorkerID = "A2FBHZUEK35JBP"
TIEBREAK_WorkerID = "TIEBREAKER"