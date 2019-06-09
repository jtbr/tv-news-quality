#!/bin/bash
# First, run mongodb with port 25541

# convert CC docs to snippets (assumes docs db exists and has been populated with CCs)
python docs2snippets.py

# assign random indices to the snippets in the snippetdb
python genRandomIndices.py

### NOTE THE ABOVE NEED ONLY OCCUR ONCE
### GENERATION OF MODELS BEGIN HERE


#NOTE: The following produces LDA models which we've found don't work well for short text. :(

# generate topic model to optimally choose initial snippets for user-categorization
python snippetsTopicModel.py

# output sample
python getTopicExemplar.py