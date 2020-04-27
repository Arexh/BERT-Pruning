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

"""Defines standard network layers that train using l0 regularization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from state_of_sparsity.layers.l0_regularization import common
from state_of_sparsity.layers.utils import layer_utils


def matmul_train(
        x,
        log_alpha,
        beta=common.BETA,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R,
        eps=common.EPSILON):
    """Training computation for a l0-regularized matmul.
    Args:
      x: 2D Tensor representing the input batch.
      log_alpha: The log alpha parameters that control the "location" of the
        distribution.
      beta: The beta parameter, which controls the "temperature" of
        the distribution. Defaults to 2/3 from the above paper.
      limit_l: The limit_l parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      limit_r: The limit_r parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
      eps: Small constant value to use in log and sqrt operations to avoid NaNs.
    Returns:
      Output Tensor of the matmul operation.
    Raises:
      RuntimeError: If the weight_parameters argument is not a 2-tuple.
    """
    x.get_shape().assert_has_rank(2)

    # Sample the z values from the hard-concrete distribution
    weight_noise = common.hard_concrete_sample(
        log_alpha,
        beta,
        limit_l,
        limit_r,
        eps)

    mask_mat = tf.linalg.tensor_diag(weight_noise)

    return tf.matmul(x, mask_mat)


def matmul_eval(
        x,
        log_alpha,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R):
    """Evaluation computation for a l0-regularized matmul.
    Args:
      x: 2D Tensor representing the input batch.
      log_alpha: The log alpha parameters that control the "location" of the
        distribution.
      limit_l: The limit_l parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      limit_r: The limit_r parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
    Returns:
      Output Tensor of the matmul operation.
    Raises:
      RuntimeError: If the weight_parameters argument is not a 2-tuple.
    """
    x.get_shape().assert_has_rank(2)

    # Use the mean of the learned hard-concrete distribution as the
    # deterministic weight noise at evaluation time
    weight_noise = common.hard_concrete_mean(
        log_alpha,
        limit_l,
        limit_r)

    mask_mat = tf.linalg.tensor_diag(weight_noise)

    return tf.matmul(x, mask_mat)


def embedding_lookup_train(
        weight_parameters,
        ids,
        name=None,
        beta=common.BETA,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R,
        eps=common.EPSILON):
    """Training computation for a l0-regularized embedding lookup.
    Args:
      weight_parameters: 2-tuple of Tensors, where the first tensor is the
        unscaled weight values and the second is the log of the alpha values
        for the hard concrete distribution.
      ids: A Tensor with type int32 or int64 containing the ids to be looked up
        in params.
      name: String. Name of the operator.
      beta: The beta parameter, which controls the "temperature" of
        the distribution. Defaults to 2/3 from the above paper.
      limit_l: The limit_l parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      limit_r: The limit_r parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
      eps: Small constant value to use in log and sqrt operations to avoid NaNs.
    Returns:
      Output Tensor of the embedding lookup.
    Raises:
      RuntimeError: If the weight_parameters argument is not a 2-tuple.
    """
    theta, log_alpha = _verify_weight_parameters(weight_parameters)

    # Before we do anything, lookup the theta values and log_alpha
    # values so that we can do our sampling and weight scaling in
    # the lower dimensional output batch
    embedding_theta = layer_utils.gather(theta, ids)
    embedding_log_alpha = layer_utils.gather(log_alpha, ids)

    # Sample the z values for the output batch from the hard-concrete
    embedding_noise = common.hard_concrete_sample(
        embedding_log_alpha,
        beta,
        limit_l,
        limit_r,
        eps)
    return tf.identity(embedding_theta * embedding_noise, name=name)


def embedding_lookup_eval(
        weight_parameters,
        ids,
        name=None,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R):
    """Evaluation computation for a l0-regularized embedding lookup.
    Args:
      weight_parameters: 2-tuple of Tensors, where the first tensor is the
        unscaled weight values and the second is the log of the alpha values
        for the hard concrete distribution.
      ids: A Tensor with type int32 or int64 containing the ids to be looked up
        in params.
      name: String. Name of the operator.
      limit_l: The limit_l parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      limit_r: The limit_r parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
    Returns:
      Output Tensor of the embedding lookup.
    Raises:
      RuntimeError: If the weight_parameters argument is not a 2-tuple.
    """
    theta, log_alpha = _verify_weight_parameters(weight_parameters)

    # Before we do anything, lookup the theta values and log_alpha
    # values so that we can do our sampling and weight scaling in
    # the lower dimensional output batch
    embedding_theta = layer_utils.gather(theta, ids)
    embedding_log_alpha = layer_utils.gather(log_alpha, ids)

    # Calculate the mean of the learned hard-concrete distribution
    # and scale the output embedding vectors
    embedding_noise = common.hard_concrete_mean(
        embedding_log_alpha,
        limit_l,
        limit_r)
    return tf.identity(embedding_theta * embedding_noise, name=name)


def l0_norm(
        log_alpha,
        beta=common.BETA,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R):
    """Calculate the l0-regularization contribution to the loss.
    Args:
      log_alpha: Tensor of the log alpha parameters for the hard concrete
        distribution.
      beta: The beta parameter, which controls the "temperature" of
        the distribution. Defaults to 2/3 from the above paper.
      limit_l: The limit_l parameter, which controls the lower bound of the
        stretched distribution. Defaults to -0.1 from the above paper.
      limit_r: The limit_r parameters, which controls the upper bound of the
        stretched distribution. Defaults to 1.1 from the above paper.
    Returns:
      Scalar tensor containing the unweighted l0-regularization term contribution
      to the loss.
    """
    # Value of the CDF of the hard-concrete distribution evaluated at 0
    reg_per_weight = tf.sigmoid(log_alpha - beta * tf.log(-limit_l / limit_r))
    return tf.reduce_sum(reg_per_weight)
