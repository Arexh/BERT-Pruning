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

import common


def matmul_train(
        x,
        log_alpha,
        beta=common.BETA,
        limit_l=common.LIMIT_L,
        limit_r=common.LIMIT_R,
        eps=common.EPSILON):
    """Training computation for a l0-regularized matmul.

    The hard concrete distribution is described in
    https://arxiv.org/abs/1910.04732.

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

    The hard concrete distribution is described in
    https://arxiv.org/abs/1910.04732.

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