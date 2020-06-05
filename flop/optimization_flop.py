# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import common

# Modify based on
# https://github.com/asappresearch/flop/blob/master/flop/train.py.


def create_optimizer(loss,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     lr_warmup=100,
                     model_dim=768,
                     lambda_lr=1.0,
                     alpha_lr=0.001,
                     target_sparsity=0.8,
                     target_sparsity_warmup=80000):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    lambda_1 = tf.get_variable(
        "lambda_1",
        shape=[1],
        dtype=tf.float32,
        trainable=True,
        initializer=tf.zeros_initializer())

    lambda_2 = tf.get_variable(
        "lambda_2",
        shape=[1],
        dtype=tf.float32,
        trainable=True,
        initializer=tf.zeros_initializer())

    lambda_learning_rate = noam_lr_scheduler(
        init_lr=tf.constant(-lambda_lr, shape=[], dtype=tf.float32),
        warmup=lr_warmup,
        d_model=model_dim,
        step=global_step)

    alpha_learning_rate = noam_lr_scheduler(
        init_lr=tf.constant(alpha_lr, shape=[], dtype=tf.float32),
        warmup=lr_warmup,
        d_model=model_dim,
        step=global_step)

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    target_sparsity = tf.constant(
        max(min(target_sparsity, 1.0), 0.0), shape=[], dtype=tf.float32)
    target_sparsity_warmup = tf.cast(tf.constant(
        target_sparsity_warmup, shape=[], dtype=tf.int32), tf.float32)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    optimizer_alpha = AdamWeightDecayOptimizer(
        learning_rate=alpha_learning_rate,
        epsilon=1e-8)

    optimizer_lambda = AdamWeightDecayOptimizer(
        learning_rate=lambda_learning_rate,
        epsilon=1e-8)

    tvars = tf.trainable_variables()

    prunable_parameters = sum(tvar.shape[0] * tvar.shape[1]
                              for tvar in tvars if '_p/kernel' in tvar.name or '_q/kernel' in tvar.name)
    prunable_parameters = tf.cast(tf.constant(
        prunable_parameters, dtype=tf.int32), tf.float32)

    bias = tf.constant(common.BIAS, shape=[], dtype=tf.float32)

    # Calculate expected sparsity
    vars_dict = {}
    expected_params = tf.constant(0, shape=[], dtype=tf.float32)
    for tvar in tvars:
        if '_p/kernel' in tvar.name or '_q/kernel' in tvar.name or '_g/log_alpha' in tvar.name:
            layer_str = re.findall(
                r'layer_\d+/[a-z/]*_[pqg]', tvar.name)[0][:-2]
            matrix_str = re.findall(r'_[pqg]/', tvar.name)[0][1:2]
            if layer_str not in vars_dict:
                vars_dict[layer_str] = {}
            vars_dict[layer_str][matrix_str] = tvar

    for key, value in vars_dict.items():
        input_feature = tf.constant(value['p'].shape[0], dtype=tf.int32)
        output_feature = tf.constant(value['q'].shape[1], dtype=tf.int32)
        input_feature = tf.cast(input_feature, tf.float32)
        output_feature = tf.cast(output_feature, tf.float32)
        alpha_param = value['g']
        l0_norm = tf.reduce_sum(tf.math.sigmoid(tf.add(alpha_param, bias)))
        expected_params = tf.add(expected_params, tf.add(tf.multiply(
            input_feature, l0_norm), tf.multiply(l0_norm, output_feature)))

    expected_sparsity = tf.subtract(tf.constant(
        1., dtype=tf.float32), tf.divide(expected_params, prunable_parameters))
    target_sparsity = tf.cond(tf.math.greater(target_sparsity_warmup, 0),
                              lambda: tf.multiply(target_sparsity, tf.math.minimum(
                                  1.0, tf.divide(tf.cast(global_step, tf.float32), target_sparsity_warmup))),
                              lambda: target_sparsity)
    lagrangian_loss = tf.multiply(
        lambda_1, tf.subtract(target_sparsity, expected_sparsity))
    lagrangian_loss = tf.add(lagrangian_loss, tf.multiply(
        lambda_2, tf.math.square(tf.subtract(target_sparsity, expected_sparsity))))
    l2_regularization_loss = tf.losses.get_regularization_loss()

    tf.summary.scalar("lagrangian_loss", tf.reshape(lagrangian_loss, []))
    tf.summary.scalar("l2_regularization_loss", tf.reshape(l2_regularization_loss, []))
    tf.summary.scalar("model_lr", learning_rate)
    tf.summary.scalar("lambda_lr",
                      tf.reshape(lambda_learning_rate, []))
    tf.summary.scalar("alpha_lr",
                      tf.reshape(alpha_learning_rate, []))
    tf.summary.scalar("expected_sparsity", tf.reshape(expected_sparsity, []))
    tf.summary.scalar("target_sparsity", tf.reshape(target_sparsity, []))
    tf.summary.scalar("lambda_1", tf.reshape(lambda_1, []))
    tf.summary.scalar("lambda_2", tf.reshape(lambda_2, []))

    if init_lr == 0:
        temp_lst = []
        for tvar in tvars:
            if 'log_alpha' not in tvar.name and 'lambda_' not in tvar.name:
                temp_lst.append(tvar)
        for tvar in temp_lst:
            tvars.remove(tvar)

    final_loss = tf.add(loss, lagrangian_loss)
    final_loss = tf.add(final_loss, l2_regularization_loss)
    grads = tf.gradients(final_loss, tvars)
    var_zip = zip(grads, tvars)

    grads_list = []
    tvars_list = []
    grads_list_lambda = []
    tvars_list_lambda = []
    grads_list_alpha = []
    tvars_list_alpha = []
    for grad, tvar in var_zip:
        if 'log_alpha' not in tvar.name and 'lambda_' not in tvar.name:
            grads_list.append(grad)
            tvars_list.append(tvar)
        elif 'lambda_' in tvar.name:
            grads_list_lambda.append(grad)
            tvars_list_lambda.append(tvar)
        else:
            grads_list_alpha.append(grad)
            tvars_list_alpha.append(tvar)
    
    tf.logging.info("Normal: %d" % len(grads_list))
    tf.logging.info("Lambda: %d" % len(grads_list_lambda))
    tf.logging.info("Alpha: %d" % len(grads_list_alpha))

    # This is how the model was pre-trained.
    (grads_list, _) = tf.clip_by_global_norm(grads_list, clip_norm=1.0)

    model_params = zip(grads_list, tvars_list)
    alpha_params = zip(grads_list_alpha, tvars_list_alpha)
    lambda_params = zip(grads_list_lambda, tvars_list_lambda)

    train_op = optimizer.apply_gradients(
        model_params, global_step=global_step)

    train_op_alpha = optimizer_alpha.apply_gradients(
        alpha_params, global_step=global_step)

    train_op_lambda = optimizer_lambda.apply_gradients(
        lambda_params, global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(
        train_op, train_op_alpha, train_op_lambda, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name    

def noam_lr_scheduler(init_lr, warmup, d_model, step):
    """
    Taken from:
    https://github.com/asappresearch/flambe/blob/master/flambe/optim/noam.py.

    Linear warmup and then quadratic decay.

    Linearly increases the learning rate from 0 to 1 over
    `warmup` steps.
    Quadratically decreases the learning rate after.

    This scheduler is generally used after every training batch.

     Args:
      init_lr: The initial learning rate.
      warmup: The number of linear warmup phases.
      d_model: The index of last step. Default: -1
      step: The current step. Could be training over validation steps.
    Returns:
      The output factor.
    """
    step = tf.cast(step, tf.float32)
    warmup = tf.constant(value=warmup, shape=[], dtype=tf.float32)
    d_model = tf.constant(value=d_model, shape=[], dtype=tf.float32)
    cond = tf.logical_and(tf.equal(step, 0), tf.equal(warmup, 0))

    def true_f():
        return tf.divide(tf.constant([1.0]), tf.math.sqrt(d_model))

    def false_f():
        return tf.cond(tf.math.greater(step, warmup),
                       lambda: tf.divide(tf.divide(tf.constant(
                           [1.0]), tf.math.sqrt(d_model)), tf.math.sqrt(step)),
                       lambda: tf.divide(tf.divide(step, tf.math.sqrt(d_model)), tf.math.pow(warmup, tf.constant([1.5]))))
    return tf.multiply(init_lr, tf.cond(cond, true_f, false_f))
    # if step == 0 and warmup == 0:
    #     return 1. / (d_model ** 0.5)
    # else:
    #     if step > warmup:
    #         return 1. / (d_model ** 0.5) / (step ** 0.5)
    #     else:
    #         return step / (d_model ** 0.5) / (warmup ** 1.5)
