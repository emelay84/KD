#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:24:40 2021

@author: ay
"""

#from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
#from utils.utils import transform_mts_to_ucr_format
#from utils.utils import visualize_filter
#from utils.utils import viz_for_survey_paper
#from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.constants import ITERATIONS_STUDENT_ALONE
from utils.utils import read_all_datasets
from utils.constants import FILTERS, FILTERS2
#from sklearn.preprocessing import OneHotEncode

def fit_classifier(output_directory, filters, filters2):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    
 #   print (y_train)

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
 #   classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

 #   classifier.fit(x_train, y_train, x_test, y_test, y_true)


    create_classifier(classifier_name, x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2)


def create_classifier(classifier_name, x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2):
    if classifier_name == 'teacher':
        from teacher import create_teacher
        return create_teacher(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, 128, 256)
    if classifier_name == 'StudentAlone':
        from StudentAlone import create_StudentAlone
        return create_StudentAlone(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2)
    if classifier_name == 'Student':
        from Student import create_Student
        return create_Student(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2)
    # if classifier_name == 'mcnn':
    #     from classifiers import mcnn
    #     return mcnn.Classifier_MCNN(output_directory, verbose)
    # if classifier_name == 'tlenet':
    #     from classifiers import tlenet
    #     return tlenet.Classifier_TLENET(output_directory, verbose)
    # if classifier_name == 'twiesn':
    #     from classifiers import twiesn
    #     return twiesn.Classifier_TWIESN(output_directory, verbose)
    # if classifier_name == 'encoder':
    #     from classifiers import encoder
    #     return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'mcdcnn':
    #     from classifiers import mcdcnn
    #     return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'cnn':  # Time-CNN
    #     from classifiers import cnn
    #     return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'inception':
    #     from classifiers import inception
    #     return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = "/home/ay/anaconda3/envs/KD/projetKD/UCRArchive_2018"

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets()
            
            if classifier_name == 'StudentAlone':
                    iterations = ITERATIONS_STUDENT_ALONE
            else:
                    iterations = ITERATIONS
            
            i = 0
            for filters in FILTERS:
                i = i + 1
                filters2 = FILTERS2[i-1]
                
                for iter in range(iterations):
                    print('\t\titer', iter)
                    
                    trr = ''
                    if iter != 0:
                        trr = '_itr_' + str(iter)
                        
                    if classifier_name == 'teacher':
                        tmp_output_directory = root_dir + '/results/'  + classifier_name + '/' + archive_name + trr + '/'    
                    else:
                        tmp_output_directory = root_dir + '/results/'  + '/results_f1_' + str(filters) + '_f2_' + str(filters2) + '/' + classifier_name + '/' + archive_name + trr + '/'
    
                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                        print('\t\t\tdataset_name: ', dataset_name)
    
                        output_directory = tmp_output_directory + dataset_name + '/'
    
                        create_directory(output_directory)
    
                        fit_classifier(output_directory, filters, filters2)
    
                        print('\t\t\t\tDONE')
    
                        # the creation of this directory means
                        create_directory(output_directory + '/DONE')
                    if classifier_name == 'teacher':
                        break
                        

# elif sys.argv[1] == 'transform_mts_to_ucr_format':
#     transform_mts_to_ucr_format()
# elif sys.argv[1] == 'visualize_filter':
#     visualize_filter(root_dir)
# elif sys.argv[1] == 'viz_for_survey_paper':
#     viz_for_survey_paper(root_dir)
# elif sys.argv[1] == 'viz_cam':
#     viz_cam(root_dir)
# elif sys.argv[1] == 'generate_results_csv':
#     res = generate_results_csv('results.csv', root_dir)
#     print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    dataset_name = sys.argv[1]
    classifier_name = sys.argv[2]
    itr = sys.argv[3]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + 'UCRArchive_2018' + itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', 'UCRArchive_2018', dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, dataset_name)

        fit_classifier(output_directory)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')