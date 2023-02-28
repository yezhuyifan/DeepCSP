#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow.compat.v1 as tf

import inspect
from six.moves import xrange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()
tf.set_random_seed(321)


degrees = [1, 2, 3, 4, 5]


def build_neural_fps_network(substances, variables, model_params):
    def matmult_neighbors(atom_features, layer, substances, variables):
        # Computes the activations of neighbors by degree
        with tf.name_scope("matmul_neighbors/") as matmul_neighbors_scope:

            activations_by_degree = []
            for degree in degrees:
                atom_neighbor_list = substances['atom_neighbors_{}'.format(degree)]
                bond_neighbor_list = substances['bond_neighbors_{}'.format(degree)]
                
                neighbor_features = [
                    tf.gather(params=atom_features, indices=atom_neighbor_list),
                    tf.gather(params=substances['bond_features'], indices=bond_neighbor_list)]
                stacked_neighbors = tf.concat(axis=2, values=neighbor_features)
                summed_neighbors = tf.reduce_sum(stacked_neighbors, reduction_indices=1)

                neighbor_filter = variables['layer_{}_neighbor_{}_filter'.format(layer, degree)]
                activations = tf.matmul(summed_neighbors, neighbor_filter)
                activations_by_degree.append(activations)
                activations = tf.concat(axis=0, values=activations_by_degree, name="activations")
            return activations

    def update_layer(atom_features, layer, substances, variables):
        # Calculate activations for a single layer of the network
        with tf.name_scope("layer_{}/".format(layer)) as update_layer_scope:
            layer_bias = variables["layer_{}_biases".format(layer)]
            layer_self_filter = variables["layer_{}_self_filter".format(layer)]
            self_activations = tf.matmul(atom_features, layer_self_filter)

            neighbor_activations = matmult_neighbors(atom_features, layer, substances, variables)
            activations = tf.nn.bias_add(tf.add(neighbor_activations, self_activations), layer_bias)

            activations_mean, activations_variance = tf.nn.moments(activations, [0], keep_dims=True)
            activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)
            activations = tf.nn.relu(activations, name="activations")
            return activations

    atom_activations = []
    def write_to_fingerprint(atom_features, layer, substances, variables):
        # Write the atom features to the fingerprint
        with tf.name_scope("layer_{}/".format(layer)) as scope:
            out_weights = variables['layer_output_weights_{}'.format(layer)]
            out_bias = variables['layer_output_bias_{}'.format(layer)]
            hidden = tf.nn.bias_add(tf.matmul(atom_features, out_weights), out_bias)
            atom_outputs = tf.nn.softmax(hidden)

            layer_output = tf.sparse_tensor_dense_matmul(substances['substance_atoms'], atom_outputs, name=scope)
            
            atom_activations.append(atom_outputs)
            return layer_output, atom_outputs

    with tf.name_scope("fingerprint/") as fingerprint_scope:
        atom_features = substances['atom_features']
        fps, rnn_embedding = write_to_fingerprint(atom_features, 0, substances, variables)
        
        num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
        for layer in xrange(len(num_hidden_features)):
            atom_features = update_layer(atom_features, layer, substances, variables)
            fps_layer, rnn_embedding = write_to_fingerprint(atom_features, layer+1, substances, variables)

            fps += fps_layer
            rnn_embedding += rnn_embedding
        return fps, rnn_embedding, atom_activations


def build_attention(rnn_embedding, substances, model_params, is_training=False):
        last_add = tf.constant(0, shape=[1, model_params["fp_length"]], dtype=tf.float32)
        rnn_embedding = tf.concat([rnn_embedding, last_add], axis=0)
        embedded_seq = tf.nn.embedding_lookup(rnn_embedding, substances["rnn_raw_input"])

        q = embedded_seq
        kT = tf.transpose(q, perm=[0, 2, 1])

        qkT = tf.matmul(q, kT)
        qkT = tf.divide(qkT, tf.sqrt(tf.cast(kT.shape[2], dtype=tf.float32)))

        att_prob = tf.nn.softmax(qkT)
        att = tf.matmul(att_prob, q)
        att_fps = tf.reduce_sum(att, axis=1)
        return att_fps, att_prob


def build_convolution_prediction_network(fps, variables, model_params):
    # Build the convolutional prediction network
    with tf.name_scope("convolution_prediction") as convolution_prediction_scope:

        activations = fps
        layer_sizes = model_params['prediction_layer_sizes'] + [1]
        for layer in range(len(layer_sizes) - 1):
            weights = variables['prediction_weights_{}'.format(layer)]
            biases = variables['prediction_biases_{}'.format(layer)]
            activations = tf.nn.bias_add(tf.matmul(activations, weights), biases, name="activations")

            if layer < len(layer_sizes) - 2:
                activations_mean, activations_variance = tf.nn.moments(activations, [0], keep_dims=True)
                activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)
                activations = tf.nn.relu(activations)

        return tf.squeeze(activations, name=convolution_prediction_scope)


def build_loss_network_modified(normed_predictions, labels,
                                fingerprint_variables, prediction_variables,
                                model_params):
    # Build the loss network to calculate the loss
    with tf.name_scope("loss") as loss_scope:

        normed_predictions = tf.reshape(normed_predictions, [-1, 1])
        mse_loss = tf.reduce_sum(tf.pow(normed_predictions - labels, 2))

        # l2 regularization
        fingerprint_regularization = fingerprint_variables['l2_loss'] * model_params['l2_penalty'] / fingerprint_variables['n_params']
        prediction_regularization = prediction_variables['l2_loss'] * model_params['l2_penalty'] / prediction_variables['n_params']
        regularization = tf.add(fingerprint_regularization, prediction_regularization, name="regularization")

        loss = tf.add(mse_loss, regularization, name=loss_scope)
        return normed_predictions, loss
