#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd

import pickle
import inspect

import tensorflow.compat.v1 as tf
import getopt

from data_features_test_pep8 import *
from model_test_pep8 import *
from evaluation_pep8 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()


#Print result
def help_info():
    """Prints usage instructions"""
    helpinfo = """
    Usage: python predict_test.py [Option]

    Option:
    -h, --help \t\t-- Usage instruction
    -t, --test \t\t-- Model performance on test data
    -d, --drug \t\t-- Predicting density for marketed drugs
    -c, --case \t\t-- Predicting density for case study drugs

    Note: If the option is NULL, default value "--test" will be used.
    """
    print(helpinfo)


def Print_result(Result_opt,Test_target,Test_pred):
    """Prints the results based on the option given"""
    if Result_opt == 1:
        print(Test_pred[:N_test].to_string(index=False))
        print('Test MAE:', mean_absolute_error(Test_target[:N_test], Test_pred.iloc[:N_test,1]))
        print('Test MSE:', mean_squared_error(Test_target[:N_test], Test_pred.iloc[:N_test,1]))
        print('Test R2:', r2_score(Test_target[:N_test], Test_pred.iloc[:N_test,1]))
    elif Result_opt == 2:
        print(Test_pred[N_test:N_test+N_drug].to_string(index=False))
    elif Result_opt == 3: 
        print(Test_pred[-N_case:].to_string(index=False))
    print("\nResults provided by DeepCSP")


# start
# get option
try:
    opts,args = getopt.getopt(sys.argv[1:],'tdch',['test','drugs','case','help'])
    if opts == []:
        Result_opt = 1
    else: 
        for opt_name,opt_value in opts:
            if opt_name in ('-h','--help'):
                help_info()
                exit()            
            if opt_name in ('-t','--test'):
                Result_opt = 1
            if opt_name in ('-d','--drugs'):
                Result_opt = 2
            if opt_name in ('-c','--case'):
                Result_opt = 3
except getopt.GetoptError:
    print ('\nPlease input valid option, usage instruction is as below:')
    help_info()
    sys.exit('Sorry and goodbye!')

#constant
N_test = 17775
N_drug = 926
N_case = 34

# Load the data
data_test = pd.read_csv('../../data/testdata.csv')
substance_names = data_test["Refcode"]

# Set the parameters
model_params = {
    "fp_length": 200,
    "fp_depth": 4,
    "fp_width": 30,
    "h1_size": 512,
    "l2_penalty": 0.00001, 
    "pos_weight": 1.5
}

# Append prediction layer sizes
model_params['prediction_layer_sizes'] = [model_params['fp_length'], model_params['h1_size'], 128, 64]

# Read the data based on the result_opt parameter
if Result_opt == 1:
    testdata = data_test.iloc[:N_test]
else:
    testdata = data_test

# Get the test inputs and labels
test_inputs, test_labels = testdata['SMILES'].values, testdata['Calculated density'].values
test_labels = np.reshape(test_labels, [-1, 1]).astype(np.float32)


# Test set
# Convert the SMILES to array representation
test_substances = array_rep_from_smiles(tuple(test_inputs))
test_substances =  trans(test_substances)

# Create placeholder data for the test substances
rnn_raw_input_test = tf.placeholder(test_substances['rnn_raw_input'].dtype, test_substances['rnn_raw_input'].shape)
atom_features_test = tf.placeholder(test_substances['atom_features'].dtype, test_substances['atom_features'].shape)
bond_features_test = tf.placeholder(test_substances['bond_features'].dtype, test_substances['bond_features'].shape)
atom_neighbors_1_test = tf.placeholder(test_substances['atom_neighbors_1'].dtype, test_substances['atom_neighbors_1'].shape)
atom_neighbors_2_test = tf.placeholder(test_substances['atom_neighbors_2'].dtype, test_substances['atom_neighbors_2'].shape)
atom_neighbors_3_test = tf.placeholder(test_substances['atom_neighbors_3'].dtype, test_substances['atom_neighbors_3'].shape)
atom_neighbors_4_test = tf.placeholder(test_substances['atom_neighbors_4'].dtype, test_substances['atom_neighbors_4'].shape)
atom_neighbors_5_test = tf.placeholder(test_substances['atom_neighbors_5'].dtype, test_substances['atom_neighbors_5'].shape)
bond_neighbors_1_test = tf.placeholder(test_substances['bond_neighbors_1'].dtype, test_substances['bond_neighbors_1'].shape)
bond_neighbors_2_test = tf.placeholder(test_substances['bond_neighbors_2'].dtype, test_substances['bond_neighbors_2'].shape)
bond_neighbors_3_test = tf.placeholder(test_substances['bond_neighbors_3'].dtype, test_substances['bond_neighbors_3'].shape)
bond_neighbors_4_test = tf.placeholder(test_substances['bond_neighbors_4'].dtype, test_substances['bond_neighbors_4'].shape)
bond_neighbors_5_test = tf.placeholder(test_substances['bond_neighbors_5'].dtype, test_substances['bond_neighbors_5'].shape)

# Assign the placeholder data to the corresponding test substance
test_placeholder = test_substances.copy()
test_placeholder['rnn_raw_input'] = rnn_raw_input_test
test_placeholder['atom_features'] = atom_features_test
test_placeholder['bond_features'] = bond_features_test
test_placeholder['atom_neighbors_1'] = atom_neighbors_1_test
test_placeholder['atom_neighbors_2'] = atom_neighbors_2_test
test_placeholder['atom_neighbors_3'] = atom_neighbors_3_test
test_placeholder['atom_neighbors_4'] = atom_neighbors_4_test
test_placeholder['atom_neighbors_5'] = atom_neighbors_5_test
test_placeholder['bond_neighbors_1'] = bond_neighbors_1_test
test_placeholder['bond_neighbors_2'] = bond_neighbors_2_test
test_placeholder['bond_neighbors_3'] = bond_neighbors_3_test
test_placeholder['bond_neighbors_4'] = bond_neighbors_4_test
test_placeholder['bond_neighbors_5'] = bond_neighbors_5_test

test_labels_placeholder = tf.placeholder(test_labels.dtype, [None, test_labels.shape[1]])


# Model
# Load trained variables
fingerprint_variables_file=open(r'../models/MolGAT/fingerprint_variables.json','rb')
fingerprint_variables_file.seek(0)
fingerprint_variables_array =pickle.load(fingerprint_variables_file)

prediction_variables_file=open(r'../models/MolGAT/prediction_variables.json','rb')
prediction_variables_file.seek(0)
prediction_variables_array =pickle.load(prediction_variables_file)

# Create dictionaries for fingerprint and prediction variables
fingerprint_variables = {}
prediction_variables = {}

# Loop through the arrays and convert them to tensors
for key in fingerprint_variables_array.keys():
    fingerprint_variables[key] = tf.convert_to_tensor(
        fingerprint_variables_array[key], dtype=tf.float32)

for key in prediction_variables_array.keys():
    prediction_variables[key] = tf.convert_to_tensor(
        prediction_variables_array[key], dtype=tf.float32)

# Build the neural fingerprint network
test_fps, test_rnn_embedding, test_atom_activations = build_neural_fps_network(
    test_placeholder, fingerprint_variables, model_params)
test_fps, test_att_prob = build_attention(
    test_rnn_embedding, test_placeholder, model_params, is_training=False)

# Build the convolution prediction network
test_normed_predictions = build_convolution_prediction_network(
    test_fps, prediction_variables, model_params)

# Build the loss network
test_predictions, test_loss = build_loss_network_modified(
    test_normed_predictions, test_labels_placeholder, fingerprint_variables, prediction_variables, model_params)


# Predict 
# Run the session and get the output
with tf.Session() as sess:
    test_preds, test_targets, test_atom_activations_vis, test_att_prob_vis = sess.run(
        fetches=[test_predictions, test_labels_placeholder, test_atom_activations, test_att_prob],
        feed_dict={
            rnn_raw_input_test: test_substances['rnn_raw_input'],
            atom_features_test: test_substances['atom_features'],
            bond_features_test: test_substances['bond_features'],
            atom_neighbors_1_test: test_substances['atom_neighbors_1'],
            atom_neighbors_2_test: test_substances['atom_neighbors_2'],
            atom_neighbors_3_test: test_substances['atom_neighbors_3'],
            atom_neighbors_4_test: test_substances['atom_neighbors_4'],
            atom_neighbors_5_test: test_substances['atom_neighbors_5'],
            bond_neighbors_1_test: test_substances['bond_neighbors_1'],
            bond_neighbors_2_test: test_substances['bond_neighbors_2'],
            bond_neighbors_3_test: test_substances['bond_neighbors_3'],
            bond_neighbors_4_test: test_substances['bond_neighbors_4'],
            bond_neighbors_5_test: test_substances['bond_neighbors_5'],
            test_labels_placeholder: test_labels,
        })	

    # Loop through results and store in out_names and out_pre
    out_names = []
    out_pre = []
    for i in range(len(test_preds)):
        out_names.append(str(substance_names[i]))        
        out_pre.append(round(float(test_preds[i][0]), 5))

    # Save and print the results      
    results = {"Refcode": out_names, "Predicted density": out_pre}
    results = pd.DataFrame.from_dict(results, orient="index").T
    results.to_csv("Prediction_results_opt.csv", index=None)
    Print_result(Result_opt, test_targets, results)

