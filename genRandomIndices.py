#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Oct 23 2017
@author: Justin

Assigns each document in the snippets mongodb collection a randomly-assigned index (to be used for sampling)

Warning:
 - we assume there are no other database users and the contents are unchanging
 - this will overwrite any existing random indices in the snippets collection of the db

"""

from __future__ import print_function
import sys, random
import pymongo
import logging

MONGODB_PORT = 25541

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
_logger = logging.getLogger(__name__)

def main(args):
    if len(args)>1:
        print("USAGE: genRandomIndices")
        return

    random.seed(a=1) #seed randomizer for predictability (remove arg for true random)

    _logger.info("Connecting and finding stations")

    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.snippetdb
        msnippets = mdb.snippets
        msnippets.ensure_index([('station', pymongo.ASCENDING), ('station_rand_idx', pymongo.ASCENDING)])

        for station in msnippets.distinct("station"):
             # Note that there are at most ~6M per station

            _logger.info("Adding random indices for: %s ..." % station)

            snippetCursor = msnippets.find({"station": station})
            randomSequence = range(snippetCursor.count())
            random.shuffle(randomSequence)

            # save all ids to memory
            # this is needed because for some reason iterating the cursor returns in some cases far more than snippetCursor.count() records when updating during the iteration
            ids = [doc["_id"] for doc in snippetCursor]

            # update all records to add station_rand_idx (in bulk for speed)
            operations = []
            for (i, rec_id) in enumerate(ids):
                operations.append(
                    pymongo.UpdateOne({ "_id": rec_id },
                                      { "$set": { "station_rand_idx": randomSequence[i] } })
                )

                # Send once every 1000 in batch
                if ( len(operations) == 1000 ):
                    msnippets.bulk_write(operations, ordered=False)
                    operations = []

            if ( len(operations) > 0 ):
                msnippets.bulk_write(operations, ordered=False)

    print("Done")

if __name__ == "__main__":
    main(sys.argv)