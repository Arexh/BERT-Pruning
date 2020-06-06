import sys
import os
sys.path.append(os.path.join(sys.path[0], "../bert"))
import re
import json
import math
import argparse
import numpy as np
import tensorflow as tf
import modeling_flop
from tensorflow.python.framework import ops
from tensorflow.python import pywrap_tensorflow


LIMIT_L = -0.1
LIMIT_R = 1.1


def hard_concrete_sample(tensor):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    sigmoid_v = np.vectorize(sigmoid)
    tensor = sigmoid_v(tensor) * (LIMIT_R - LIMIT_L) + LIMIT_L
    return np.clip(tensor, 0, 1.0)


def get_index(tensor, threshold=0.3):
    tensor = hard_concrete_sample(tensor)
    indexes = np.array([])
    for i in range(len(tensor)):
        if tensor[i] > threshold:
            indexes = np.append(indexes, i)
    indexes = indexes.astype(int)
    return tensor, indexes


def mask_row(tensor, indexes):
    return tensor[indexes]


def mask_col(tensor, indexes):
    return tensor[:, indexes]


def kernel_map(var_name):
    base = "/".join(var_name.split("/")[:-1])[:-1]
    return base + 'p/kernel', base + 'q/kernel'


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    model = modeling_flop.BertModelHardConcrete(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        factorize=True)
    num_labels = max(num_labels, 1)
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())


def remove_mask(bert_config_file, init_checkpoint, output_dir, threshold=0):
    reader = pywrap_tensorflow.NewCheckpointReader(init_checkpoint)
    kernel_pattern = "^bert/encoder/.*((query|key|value)|(dense))/kernel$"
    var_to_shape_map = reader.get_variable_to_shape_map()
    log_alpha_pattern = ".*_g/log_alpha$"
    log_alphas = []
    tensor_names = []
    for key in var_to_shape_map:
        if "layer_" in key:
            layer_num = int(re.findall(
                r'layer_\d+', key)[0].split("_")[1])
        else:
            layer_num = 0
        if "adam" not in key and "lambda" not in key and "global_step" not in key and "log_alpha" not in key:
            tensor_names.append([layer_num, key])
        if re.match(log_alpha_pattern, key):
            log_alphas.append([layer_num, key])

    log_alphas = sorted(log_alphas, key=lambda x: (x[0], x[1]))
    tensor_names = sorted(tensor_names, key=lambda x: (x[0], x[1]))

    tensors = {}
    for tensor_name in tensor_names:
        tensors[tensor_name[1]] = reader.get_tensor(tensor_name[1])
    dense_total_params = 0
    dense_pruned_params = 0
    dense_origin_params = 0
    dim_dict = {}
    count = 0
    for layer, var_name in log_alphas:
        tensor = reader.get_tensor(var_name)
        length = len(tensor)
        tensor, index = get_index(tensor, threshold=threshold)
        pruned_length = len(index)
        layer_sparsity = pruned_length / length
        p, q = kernel_map(var_name)
        tensor_p = tensors[p]
        tensor_q = tensors[q]
        tensor_p = tensor_p.dot(np.diag(tensor))
        dense_total_params += tensor_p.shape[0] * tensor_p.shape[1]
        dense_total_params += tensor_q.shape[0] * tensor_q.shape[1]
        dense_origin_params += tensor_p.shape[0] * tensor_q.shape[1]
        tensor_p = mask_col(tensor_p, index)
        tensor_q = mask_row(tensor_q, index)
        dense_pruned_params += tensor_p.shape[0] * tensor_p.shape[1]
        dense_pruned_params += tensor_q.shape[0] * tensor_q.shape[1]
        tensors[p] = tensor_p
        tensors[q] = tensor_q
        dim_dict[p] = tensor_p.shape[1]

    non_kernel_params = 0
    for key in tensors.keys():
        if "kernel" not in key:
            tensor = tensors[key]
            if len(tensor.shape) == 2:
                non_kernel_params += tensor.shape[0] * tensor.shape[1]
            else:
                non_kernel_params += tensor.shape[0]
    total_params = dense_origin_params + non_kernel_params
    pruned_total_params = dense_pruned_params + non_kernel_params

    ops.reset_default_graph()
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    bert_config = modeling_flop.BertConfig.from_json_file(bert_config_file)
    bert_config.pruned_layers_dim = dim_dict
    create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=None,
        segment_ids=None,
        labels=None,
        num_labels=2)
    tvars = tf.trainable_variables()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for tensor_name in tensors.keys():
        tvar = [v for v in tvars if v.name == tensor_name + ":0"][0]
        tf.logging.info("Tensor: %s %s", tvar.name, "*INIT_FROM_CKPT*")
        sess.run(tf.assign(tvar, tensors[tensor_name]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_dir, "bert_model_f.ckpt"))
    info = ["dense_total_params: %d" % dense_total_params,
            "dense_pruned_params: %d" % dense_pruned_params,
            "dense_origin_params: %d" % dense_origin_params,
            "dense_sparsity: %f" % (
                1 - dense_pruned_params / dense_total_params),
            "non_kernel_params: %d" % non_kernel_params,
            "total_params: %d" % total_params,
            "pruned_total_params: %d" % pruned_total_params,
            "actual_compact_rate: %f" % (pruned_total_params / total_params)]
    with open(os.path.join(output_dir, "info.txt"), "w") as txt_file:
        for line in info:
            txt_file.write(line + "\n")
    with open(os.path.join(output_dir, "bert_config.json"), "w") as json_file:
        json_file.write(bert_config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", help="bert config file")
    parser.add_argument(
        "--checkpoint", help="factorized checkpoint to remove mask")
    parser.add_argument("--output_folder_dir", help="output folder directory")
    parser.add_argument("--threshold", help="mask pruned threshold", type=float)
    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    remove_mask(
        bert_config_file=args.bert_config_file,
        init_checkpoint=args.checkpoint,
        output_dir=args.output_folder_dir,
        threshold=args.threshold)
