
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The code is taken from
# https://github.com/google-research/google-research/blob/master/state_of_sparsity/layers/l0_regularization/layers.py.

"""tf.layers-like API for l0-regularization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

import common
import nn
from tensorflow.python.layers import base  # pylint: disable=g-direct-tensorflow-import
# from tensorflow.contrib.layers.python.layers import utils as layer_utils
from tensorflow.python.ops import variables as tf_variables  # pylint: disable=g-direct-tensorflow-import


THETA_LOGALPHA_COLLECTION = "theta_logalpha"


class FlopMask(base.Layer):
    """Base implementation of a fully connected layer with FLOP.

    The hard concrete distribution is described in
    https://arxiv.org/abs/1910.04732.

      Args:
        x: Input, float32 tensor.
        is_training: Boolean specifying whether it is training or eval.
        trainable: Boolean defining whether this layer is trainable or not.
        init_mean: Initialization mean value for hard concrete parameter.
        init_std: Initialization std value for hard concrete parameter.
        eps: Small epsilon value to prevent math op saturation.
        beta: The beta parameter, which controls the "temperature" of
          the distribution. Defaults to 1.0 from the above paper.
        limit_l: The limit_l parameter, which controls the lower bound of the
          stretched distribution. Defaults to -0.1 from the above paper.
        limit_r: The limit_r parameters, which controls the upper bound of the
          stretched distribution. Defaults to 1.1 from the above paper.
        name: String speciying name scope of layer in network.
      Returns:
        Output Tensor of the fully connected operation.
    """

    def __init__(self,
                 activity_regularizer=None,
                 is_training=True,
                 trainable=True,
                 init_mean=0.5,
                 init_std=0.01,
                 eps=1e-6,
                 beta=1.0,
                 limit_l=-0.1,
                 limit_r=1.1,
                 name="flop_mask",
                 **kwargs):
        super(FlopMask, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.is_training = is_training
        self.init_mean = init_mean
        self.init_std = init_std
        self.eps = eps
        self.beta = beta
        self.limit_l = limit_l
        self.limit_r = limit_r

    def build(self, input_shape):
        input_shape = input_shape.as_list()

        input_hidden_size = input_shape[1]

        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha_initializer = tf.random_normal_initializer(
            mean=mean, stddev=self.init_std, dtype=self.dtype)

        self.log_alpha = tf.get_variable(
            "log_alpha",
            shape=input_hidden_size,
            initializer=self.log_alpha_initializer,
            dtype=self.dtype,
            trainable=True)

        # layer_utils.add_variable_to_collection(
        #     self.log_alpha,
        #     [THETA_LOGALPHA_COLLECTION], None)

        self.built = True

    def call(self, inputs):
        if self.is_training:
            x = nn.matmul_train(
                inputs,
                self.log_alpha,
                beta=self.beta,
                limit_l=self.limit_l,
                limit_r=self.limit_r,
                eps=self.eps)
        else:
            x = nn.matmul_eval(
                inputs,
                self.log_alpha,
                limit_l=self.limit_l,
                limit_r=self.limit_r)

        return x


def add_variable_to_collection(var, var_set, name):
    """Add provided variable to a given collection, with some checks."""
    collections = layer_utils.get_variable_collections(var_set, name) or []
    var_list = [var]
    if isinstance(var, tf_variables.PartitionedVariable):
        var_list = [v for v in var]
    for collection in collections:
        for var in var_list:
            if var not in tf.get_collection(collection):
                tf.add_to_collection(collection, var)
