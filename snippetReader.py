#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pymongo
import logging
import re
import random

import nltk.data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # first time need to nltk.download("wordnet")


logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
_logger = logging.getLogger(__name__)

random.seed(a=1) # seed for predictability

class Reader(object):
    ''' Source reader object feeds other objects to iterate through a source. '''
    def __init__(self):
        ''' init '''
        self.stop = set(stopwords.words('english'))
        self.wn_lemmatizer = WordNetLemmatizer()
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def prepare_words(self, text):
        ''' Prepare text
        '''
        words = []

        # split text into sentences and then words
        sentences = self.sent_detector.tokenize(text.lower())
        for sentence in sentences:
            words += ( nltk.tokenize.word_tokenize(sentence) )

        # drop 's, numbers, punctuation, numbers, short/long and stop-words
        words = [ w for w in words if (len(w) > 2 and len(w) < 20 and
                                       w.isalnum() and
                                       not w.isdigit() and
                                       w not in self.stop) ]

        # lemmatize (remove plurals mainly)
        words = [self.wn_lemmatizer.lemmatize(w) for w in words]

        return words

    def __iter__(self):
        ''' virtual method '''
        pass


class MongoCorpusReader(Reader):
    def __init__(self, mongoURI="mongodb://localhost:27017",
                 db_name=None, collection=None,
                 return_holdouts=False, additional_queries={},
                 rand_idx_low=None, rand_idx_high=None,
                 num_rand = None, rand_pct = None,
                 corpus_dict = None, dict_output="bow",
                 tfidf = None, fulldoc = False):
        ''' init
            :param mongoURI: mongoDB URI. default: mongodb://localhost:27017
            :param db_name: MongoDB database name.
            :param collection: MongoDB Collection name.
            :param return_holdouts: if true, return from the holdout set (otherwise exclude it)
            :param additional_queries: dict of additional fields to match when getting cursor
            :param rand_idx_low/high: min or max value of station_rand_idx to return (incompatible with return_holdouts)
            :param num_rand: (not to be used with rand_idx_???) total #/random documents to return (requires Mongodb >=v3.2)
            :param rand_pct: (not to be used with rand_idx_???) approximate sample pct of records to return (used rather than num_rand when Mongdb is <v3.2)
            :param corpus_dict: the dictionary to use to convert to bag of words, if any (if None, returns words as text rather than word indices)
            :param dict_output: if "index": returns the indices from the dictionary, if "bow": converts these to bow (counts per token), if "text": only removes tokens not in the dictionary; applies only if tdidf isn't given but corpus_dict is
            :param tfidf: if provided, returns tfidf weights for words from bag of words
            :param fulldoc: if True, return raw document with all fields, plus 'text', 'bow', and 'tfidf' as applicable (not to be used directly by gensim)
        '''
        Reader.__init__(self)
        self.conn = None
        self.mongoURI = mongoURI
        self.db_name = db_name
        self.collection = collection
        self.return_holdouts = return_holdouts
        self.additional_queries = additional_queries
        self.rand_idx_low = rand_idx_low
        self.rand_idx_high = rand_idx_high
        self.num_rand = num_rand
        self.rand_pct = rand_pct
        self.corpus_dict = corpus_dict
        self.dict_output = dict_output
        assert(dict_output == "text" or dict_output == "index" or dict_output == "bow")
        self.tfidf = tfidf
        self.fulldoc = fulldoc
        self.text_fields = ['snippet_part1', 'snippet_part2', 'snippet_part3']

    def connect(self):
        if not self.conn:
            try:
                self.conn = pymongo.MongoClient(self.mongoURI)[self.db_name][self.collection]
            except Exception, ex:
                raise Exception("ERROR establishing connection: %s" % ex)

    def len(self):
        '''return number of docs corpus will return'''
        #TODO: this doesn't account for holdouts and additional_queries!
        if self.num_rand:
            return self.num_rand
        elif self.rand_idx_high and self.rand_idx_low:
            return self.rand_idx_high - self.rand_idx_low
        elif self.rand_idx_high:
            return self.rand_idx_high
        elif self.rand_pct:
            #unknowable in advance
            return None
        else:
            self.connect()
            no_docs = self.conn.count()
            if self.rand_idx_low:
                return no_docs - self.rand_idx_low
            else:
                return no_docs

    def get_value(self, value):
        ''' convenience method to retrieve value.
        '''
        if not value:
            return value
        if isinstance(value, list):
            return ' '.join([v.encode('utf-8', 'replace').decode('utf-8', 'replace') for v in value])
        else:
            return value.encode('utf-8', 'replace').decode('utf-8', 'replace')

    def strip_speaker(self, strval):
        ''' remove speaker symbol ("> ") from string, if present
        '''
        if strval and len(strval)>2 :
            return strval.replace("> ", "")
        else:
            return strval

    def __iter__(self):
        ''' Iterate through the collection, returning cleaned, tokenized, lemmatized words in docs
        '''
        self.connect()
        if self.num_rand and not self.rand_pct:
            # return a random sample of the entire collection
            # requires mongodb >= 3.2
            query = []
            query.append( {"$match": {"$exists": self.return_holdouts}} )
            query.append( {"$sample": {"size": self.num_rand}} )
            cursor = self.conn.aggregate(query)
        else:
            query = self.additional_queries
            if self.return_holdouts:
                query["holdout"] = {"$exists": True}

            else:
                # return a random sample of size (rand_idx_high - rand_idx_low + 1) from each station, based upon random indices in database (generated by genRandomIndices), or all docs if these aren't set
                if self.rand_idx_low or self.rand_idx_high:
                    query["station_rand_idx"] = {}
                    if self.rand_idx_low:
                        query["station_rand_idx"]["$gte"] = self.rand_idx_low
                    if self.rand_idx_high:
                        query["station_rand_idx"]["$lt"] = self.rand_idx_high

                query["holdout"] = {"$exists": False}

            cursor = self.conn.find(query, None, no_cursor_timeout=True)

        for doc in cursor:
            if self.rand_pct and (random.random() > self.rand_pct):
                # alternative method for uniform random sampling when mongodb < 3.2
                # (only a good idea when rand_pct is high)
                continue

            content = ""
            for f in self.text_fields:
                content +=" %s" % (self.strip_speaker(self.get_value(doc.get(f))))
            text = self.prepare_words(content)

            result = doc.copy() # note, deepcopy() not needed
            result['text'] = text

            if self.corpus_dict:
                if self.tfidf or self.dict_output == "bow":
                    result['dict_out'] = self.corpus_dict.doc2bow(text)
                elif self.dict_output == "index":
                    idxs = self.corpus_dict.doc2idx(text, -1)
                    result['dict_out'] = [idx for idx in idxs if idx != -1]
                elif self.dict_output == "text":
                    the_dict = set(self.corpus_dict.values())
                    result['dict_out'] = [word for word in result['text'] if word in the_dict]

            if self.tfidf and self.corpus_dict:
                result['tfidf'] = self.tfidf[result['dict_out']] # bow in this case
                if self.fulldoc:
                    yield result
                else:
                    # return as TFIDF
                    yield result['tfidf']
            elif self.corpus_dict:
                if self.fulldoc:
                    yield result
                else:
                    # return as dict_out (word indices, bow, or dict-trimmed text)
                    yield result['dict_out']
            else:
                if self.fulldoc:
                    yield result
                else:
                    # return as text words
                    yield text
        cursor.close()

if __name__ == "__main__":
    pass
