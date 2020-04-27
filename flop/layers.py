
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

"""tf.layers-like API for l0-regularization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import common
import nn
from tensorflow.python.layers import base  # pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib.layers.python.layers import utils as layer_utils
from tensorflow.python.ops import variables as tf_variables  # pylint: disable=g-direct-tensorflow-import


THETA_LOGALPHA_COLLECTION = "theta_logalpha"


class FlopFullyConnected(base.Layer):
    """Base implementation of a fully connected layer with FLOP.
      Args:
        x: Input, float32 tensor.
        num_outputs: Int representing size of output tensor.
        activation: If None, a linear activation is used.
        bias_initializer: Initalizer of the bias vector.
        bias_regularizer: Optional regularizer for the bias vector.
        log_alpha_initializer: Specified initializer of the log_alpha term.
        is_training: Boolean specifying whether it is training or eval.
        use_bias: Boolean specifying whether bias vector should be used.
        eps: Small epsilon value to prevent math op saturation.
        beta: The beta parameter, which controls the "temperature" of
          the distribution. Defaults to 2/3 from the above paper.
        limit_l: The limit_l parameter, which controls the lower bound of the
          stretched distribution. Defaults to -0.1 from the above paper.
        limit_r: The limit_r parameters, which controls the upper bound of the
          stretched distribution. Defaults to 1.1 from the above paper.
        name: String speciying name scope of layer in network.
      Returns:
        Output Tensor of the fully connected operation.
    """

    def __init__(self,
                 num_outputs,
                 activation,
                 bias_initializer,
                 bias_regularizer,
                 log_alpha_initializer,
                 activity_regularizer=None,
                 is_training=True,
                 trainable=True,
                 use_bias=True,
                 eps=common.EPSILON,
                 beta=common.BETA,
                 limit_l=common.LIMIT_L,
                 limit_r=common.LIMIT_R,
                 name="flop_mask",
                 **kwargs):
        super(FlopFullyConnected, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.num_outputs = num_outputs
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.log_alpha_initializer = log_alpha_initializer
        self.is_training = is_training
        self.use_bias = use_bias
        self.eps = eps
        self.beta = beta
        self.limit_l = limit_l
        self.limit_r = limit_r

    def build(self, input_shape):
        input_shape = input_shape.as_list()

        assert input_shape[0] == input_shape[1]

        input_hidden_size = input_shape[1]
        diag_size = input_shape[0]

        if not self.log_alpha_initializer:
            # default log alpha set s.t. \alpha / (\alpha + 1) = .1
            self.log_alpha_initializer = tf.random_normal_initializer(
                mean=2.197, stddev=0.01, dtype=self.dtype)

        self.log_alpha = tf.get_variable(
            "log_alpha",
            shape=diag_size,
            initializer=self.log_alpha_initializer,
            dtype=self.dtype,
            trainable=True)

        layer_utils.add_variable_to_collection(
            self.log_alpha,
            [THETA_LOGALPHA_COLLECTION], None)

        if self.use_bias:
            self.bias = self.add_variable(
                name="bias",
                shape=(self.num_outputs,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
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

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        if self.activation is not None:
            return self.activation(x)
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
