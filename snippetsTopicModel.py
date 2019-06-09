#!/use/bin/env python2
#  Generate dictionary and corpus in Bag of Words (BOW) and then TDIDF form,
#  to be then converted into topics.
#
#  Representative samples of topics will be the first to be classified on mechanical turk.
#
#  to run: 
#    nohup stdbuf -oL python snippetsTopicModel.py >> snippetsTopicModel-10k_gsdmm.log 2>&1 &

from __future__ import print_function
import sys, os
import logging
import pickle
import gensim
#from gensim import corpora, models, similarities
from gsdmm import MovieGroupProcess

from snippetReader import *

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level = logging.DEBUG)

REBUILD_DICT = False   # Forces the dictionary be rebuilt if True
REFILTER_DICT = False  # Re-filters the dictionary if True
REBUILD_TFIDF = False  # Rebuilds the BOW and TFIDF corpuses if True
REGEN_LDA = False      # Re-generates LDA models

# Either perform standard LDA for topic modelling or use GSDMM for document clustering
GENSIM_LDA = False   # Generates LDA model from TFIDF
GSDMM      = True    # Generates clusters using Gibbs Sampling for Dirichlet Multinomial Mixture Model


def main(args):
    mongourl = "mongodb://localhost:25541"
    snippet_db = "snippetdb"
    snippet_collection = "snippets"
#    max_dict_size = 50000
    output_stem = "snippet_topicmodel/snippet"

    #### generate dictionary from large random sample of snippets ####
    raw_dict_fn = output_stem + '_wordids_raw.txt.bz2'
    corpus_dict = []
    if (REBUILD_DICT or not os.path.isfile(raw_dict_fn)):
        print("=== Generating Dictionary") 
        # as there are 26 million snippets, this is slow
        corpus_reader_text = MongoCorpusReader(mongourl, snippet_db, snippet_collection, rand_pct=.5)
        corpus_dict = gensim.corpora.Dictionary(corpus_reader_text)

        # save raw dict (raw text compresses better than pickle, though may be slower to load)
        corpus_dict.save_as_text(raw_dict_fn)
        
    final_dict_fn = output_stem + '_wordids_final.txt.bz2'
    if (REFILTER_DICT or REBUILD_DICT or not os.path.isfile(final_dict_fn)):
        print("=== Filtering Dictionary")
        if not corpus_dict:
            corpus_dict = gensim.corpora.Dictionary.load_from_text(raw_dict_fn)
        # trim corpus_dict to contain words appearing in at least 500 documents and not more than 2.5% of docs (up to DEFAULT_DICT_SIZE)
        corpus_dict.filter_extremes(no_below=500, no_above=0.025) # keep_n=max_dict_size)
        # save final dict
        corpus_dict.save_as_text(final_dict_fn)

    # load back the id->word mapping directly from file
    # this seems to save memory, compared to keeping the object as it was
    corpus_dict = gensim.corpora.Dictionary.load_from_text(final_dict_fn)


    if GENSIM_LDA:
        #### transform from BOW to TFIDF ####
        tfidf_model_fn = output_stem + "_tfidf.model"
        num_topics = 125
        if (REBUILD_TFIDF or REBUILD_DICT or REFILTER_DICT or not os.path.isfile(tfidf_model_fn)):
            print("=== Transforming from BOW to TFIDF")

            # this is from a random sample stratified by station (we sample 20k samples per station)
            corpus_reader_bow = MongoCorpusReader(mongourl, snippet_db, snippet_collection,
                                                  rand_idx_high=20000, corpus_dict=corpus_dict)
            tfidf = gensim.models.TfidfModel(corpus_reader_bow)

            # save tfidf model
            tfidf.save(tfidf_model_fn)

            # TBD: do we even need this now?
            # save tfidf vectors for all documents in matrix market format for later use
            # (this will be big and may take a while)
            gensim.corpora.MmCorpus.serialize(output_stem + '_tfidf.mm',
                                              tfidf[corpus_reader_bow], progress_cnt=10000)
            # NOTE: .mm file is subsequently bzipped and needs to be decompressed before reading!

        else:
            print("=== Loading TFIDF")
            tfidf = gensim.models.TfidfModel.load(tfidf_model_fn)

        if REGEN_LDA:
            #### transform from TFIDF to LDA topic model ####
            print("=== Generating LDA from TFIDF")
            # we're again using the same random sample when transforming from tfidf to LDA model
            corpus_reader_tfidf = MongoCorpusReader(mongourl, snippet_db, snippet_collection,
                                                    rand_idx_high=20000, corpus_dict=corpus_dict,
                                                    tfidf=tfidf)
            # single core version:
            # recommend trying this first to get alpha and eta settings and see how many we need to get converging
            #~ lda = gensim.models.ldamodel.LdaModel(corpus=corpus_reader_tfidf, id2word=corpus_dict,
                                                  #~ num_topics=num_topics,
                                                  #~ iterations=100, passes=3,
                                                  #~ #alpha='auto', eta='auto', # these seem to increase perplexity
                                                  #~ chunksize=8000)
            # use multicore version with recommended #/cores - 1 (not including hyperthreads)
            lda = gensim.models.ldamulticore.LdaMulticore(workers=3, corpus=corpus_reader_tfidf,
                                                          id2word=corpus_dict, num_topics=num_topics,
                                                          eval_every=0, iterations=100, passes=5, chunksize=4000)
            #  save lda model
            lda.save(output_stem + '_lda.model-125.bz2')

        print("=== Loading LDA models")
        lda = gensim.models.ldamulticore.LdaMulticore.load(output_stem + '_lda.model-125.bz2')

        print("=== Gathering top topics and topic coherence")
        # show components of top topics
        corpus_reader_tfidf_small = MongoCorpusReader(mongourl, snippet_db, snippet_collection,
                                                      rand_idx_high=500, corpus_dict=corpus_dict,
                                                      tfidf=tfidf)
        top_topics = lda.top_topics(corpus=corpus_reader_tfidf_small, num_words=15)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average topic coherence: %.4f.' % avg_topic_coherence)

        print(top_topics)

    elif GSDMM:

        print("=== Loading documents for GSDMM")
        MAX_CLUSTERS = 500
        # this is from a random sample stratified by station (we sample 20k samples per station)
        corpus_reader_idxs = MongoCorpusReader(mongourl, snippet_db, snippet_collection,
                                               rand_idx_low=200, rand_idx_high=8200, 
                                               corpus_dict=corpus_dict, fulldoc=True,
                                               dict_output="index", additional_queries = {"recluster": True})
        # gsdmm works with either text words or word indices; the latter is slightly faster and uses less memory

        docs = [doc['dict_out'] for doc in corpus_reader_idxs]
        doc_ids = [doc['_id'] for doc in corpus_reader_idxs]
        print("loaded %d docs" % len(docs))

        CALC_VOCAB_SIZE = True
        if CALC_VOCAB_SIZE:
            # len(corpus_dict) is >= vocab_size since our sample < the corpus
            print("=== Calculating size of used vocabulary")
            V = set()
            for text in docs:
                for word in text:
                    V.add(word)
            vocab_size = len(V)
            V = None
            print("size of used vocab: %d words" % vocab_size)
            # len(corpus_dict) is >= vocab_size since our sample < the corpus
        else:
            # but with a large enough sample, they're nearly the same
            vocab_size = len(corpus_dict)

        print("=== Clustering documents using GSDMM")
        mgp = MovieGroupProcess(K=MAX_CLUSTERS, alpha=0.12, beta=0.08, n_iters=20)
        doc_clusters = mgp.fit(docs, vocab_size) 

        print("cluster count %d" % len(set(doc_clusters)))
        cluster_fn = output_stem + '_gsdmm_doc_clusters_2.pickle'

        data = {"doc_ids": doc_ids, "doc_clusters": doc_clusters}
        with open(cluster_fn, 'wb') as f:
            pickle.dump(data, f)
                         
        for c in range(50):
            print("\nCluster %d:\n" % c)
            print(mgp.cluster_word_distribution[c])

    else:
        print(" NO final analysis selected.")

    print("DONE")


if __name__ == "__main__":
    main(sys.argv)
