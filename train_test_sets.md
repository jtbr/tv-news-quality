# Train and holdout sets

Each set is broken into mechanical turk batches

## Holdout set:
1) station-stratified random sample 520(-1 missing): 40 per station; rand_idx<40 ; labeled by J+N 22 Feb 2019, labeled by 1x MTurkers 14 Mar 2019, tiebreaks by JB
2) first gsdmm clustering 465: 3 per initial cluster
3) secondary gsdmm clustering 399: 7 per secondary cluster

## Training set, broken into mechanical turk batches:

### station stratified random samples
0) size 520: 40 per station; 40<=rand_idx<80 ; manually labeled by JB, labeled by 1x Mturkers Sep 19, 2018
1) size 520: 40 per station; 80<=rand_idx<120 ; labeled by 2x MTurkers Oct 8, 2018, tiebreaks by JB
2) size 520: 40 per station; 120<=rand_idx<160 ; labeled by 2x MTurkers Mar 19, 2019, tiebreaks by JB
3) size 520: 40 per station; 160<=rand_idx<200

### equally sampled from each cluster in initial clustering
10) initial clustering; 465 = first 3 per cluster ; manually labeled by JB (actually 463)
11) initial clustering; 465 = next 3 per cluster (actually 463)
12) initial clustering; 465 = next 3 per cluster (actually 462)

### equally sampled from each cluster in second clustering (out of large initial clusters >=100 members)
20) secondary clustering; 456 = first 8 per cluster ; labeled by 1x MTurkers Feb 5, 2019
21) secondary clustering; 456 = next 8 per cluster


## Investigating my answers vs. Mturkers:
 - Nearly all answers are of good quality
 - Discrepancies arise primarily from 
   1) Differences in deciding what was the "predominant" topic in a snippet
   2) Differences in interpretation of partial/ambiguous excerpts
   3) Differences based in background knowledge
   4) Failure to prioritize earlier topic categories
   5) Ambiguity between tone and emotional content differences

   Deciding whether tone, fact/opinion applies to the presentation or voice excerpts
