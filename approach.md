## Approach:

- Generate snippets from CC programs, to mongo database of CCs
  - Snippets are 3 phrases, of ~100 characters each, or less if that's all a person said at once
  - We ask the category of the whole snippet, with the center snippet being the classified if the topic changes during the snippet
- Select random sample of snippets, stratified by station; (for each list of ids for samples for station, select random sample of N, where N is limited by smallest station)
- Cluster snippets in one of the following ways (in practice (1) didn't work well at all)
  1 - Use standard LDA topic modelling:
    - First Perform TF-IDF on BOW for (sample of?) corpus; generate L topics
  2 - Use GSDMM which is designed for small text and works better
    - In practice, we did two passes of GSDMM, the second on only those in the largest clusters on the first pass
  3 - Holdout sets are also marked and saved

- Generate random sample sets for mechanical turk to label
  1) Stratified by station
  2) Random representatives of each topic cluster
  3) one holdout set

- Have these labeled at least twice each using the mturk_hit.html site
- Adjudicate the results and generate a dataset of the "best response for
  each dimension we're analyzing (topic, tone, etc)"

- Now we have a dataset of snippet labels for each dimension; Use this to
  learn classifiers to predict each dimension
    - libshorttext (baseline linear)
    - AWD_LSTM using fastai, after first fine-tuning a language encoder with
      snippets (deep learning)
    - Use doc2vec and embedding methods with classic deep learning (untried)

- Use cross-fold cross-validation to evaluate performance and holding back a
set for final test

- Finally to produce data which can be asessed statistically across
  snippets / stations, run the classifiers on the whole population of
  snippets and analyze the results
    - also run classifiers on subsets of classifier labels in certain other
      dimensions (cross-dimension results)

- (UNUSED) Iterative learning:
  - Return to MTurk a sample of the most uncertain predictions
  - Repeat