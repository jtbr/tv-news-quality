#!/usr/bin/env python
'''
Run libshorttext train and validation using our separate labels and across folds
'''
from libshorttext.classifier import *
from collections import defaultdict
import os, sys
import numpy as np

def main(argv):
    if len(argv) < 3:
        print("newsLearnClassify [labeled_data_dir] [models_dir]")
        sys.exit(1)

    labeled_data_dir = argv[1]
    models_dir = argv[2]

    if not (os.path.exists(labeled_data_dir) and os.path.exists(labeled_data_dir + '/fold_0')):
        print("%s doesn't exist or doesn't contain fold dirs" % labeled_data_dir)
        sys.exit(2)

    if os.path.exists(models_dir):
        print("Models dir already exists: %s!" % models_dir)
        sys.exit(3)
    else:
        os.makedirs(models_dir)

    fold_accuracies = defaultdict(list)

    fold_dirs = sorted(os.listdir(labeled_data_dir))
    for fold_dir in fold_dirs:
        if len(fold_dir)>5 and fold_dir[:5] == 'fold_':
            fold_num = fold_dir[5:]
            full_fold_dir = labeled_data_dir + '/' + fold_dir
            labelgroup_dirs = os.listdir(full_fold_dir)
            for labelgroup in labelgroup_dirs:
                models_prefix = models_dir + '/' + labelgroup + '-fold_' + fold_num
                train_path = full_fold_dir + '/' + labelgroup + '/' + labelgroup + "_train.tsv"

                m, svm_file = train_text(train_path, svm_file=models_prefix+'-train.svm')
                m.save(models_prefix + '-model')

                validation_path = full_fold_dir + '/' + labelgroup + '/' + labelgroup + "_valid.tsv"

               	predict_result = predict_text(validation_path, m, svm_file=models_prefix + '-valid.svm')

                print("Fold {3} Labelgroup {4} Accuracy = {0:.4f}% ({1}/{2})".format(
                    predict_result.get_accuracy()*100, 
                    sum(ty == py for ty, py in zip(predict_result.true_y, predict_result.predicted_y)),
                    len(predict_result.true_y),
                    fold_num, labelgroup))

                predict_result.save(models_prefix + '-valid-predicted_result', True) # always save analyzable version of predictions

                fold_accuracies[labelgroup].append(predict_result.get_accuracy())

    print("Cross-fold cross-validation statistics:")
    for labelgroup, accuracies in fold_accuracies.iteritems():
        accuracies = np.array(accuracies)
        print("For %s, mean %02f, SD %02f" % (labelgroup, 100.*accuracies.mean(), 100.*accuracies.std()))

if __name__ == '__main__':
    main(sys.argv)