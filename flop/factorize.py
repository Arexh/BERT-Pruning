import re
import os
import copy
import argparse
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.join(sys.path[0], "../bert"))
import optimization_flop
import modeling_flop
from tensorflow.python import pywrap_tensorflow
import modeling

def kernel_map(var_name):
    lst_one = var_name.split("/")
    lst_two = copy.deepcopy(lst_one)
    lst_one[-2] += "_p"
    lst_one[-1] += ":0"
    lst_two[-2] += "_q"
    lst_two[-1] += ":0"
    return "/".join(lst_one), "/".join(lst_two)


def bias_map(var_name):
    lst = var_name.split("/")
    lst[-2] += "_q"
    lst[-1] += ":0"
    return "/".join(lst)


def get_variable_name(param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                       labels, num_labels, finetuned):
    model = modeling_flop.BertModelHardConcrete(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)
    if finetuned:
        num_labels = max(num_labels, 1)
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())


def save_factorized_model(bert_config_file, init_checkpoint, output_dir, finetuned):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=None,
        segment_ids=None,
        labels=None,
        num_labels=2,
        finetuned=finetuned)
    tvars = tf.trainable_variables()
    tvars = tf.get_collection(tf.GraphKeys.VARIABLES)
    total_parameters = 0
    reader = pywrap_tensorflow.NewCheckpointReader(init_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    kernel_pattern = "^bert/encoder/.*((query|key|value)|(dense))/kernel$"
    bias_pattern = "^bert/encoder/.*((query|key|value)|(dense))/bias$"
    tvars_names = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for var in tvars:
        tf.logging.info(var.name)
        tvars_names.append(var.name)
    for key in var_to_shape_map:
        if re.match(bias_pattern, key):
            q = bias_map(key)
            q_var = [v for v in tvars if v.name == q][0]
            tf.logging.info("Tensor: %s %s", q, "*INIT_FROM_CKPT*")
            sess.run(tf.assign(q_var, reader.get_tensor(key)))
            tvars_names.remove(q)
        elif re.match(kernel_pattern, key):
            p, q = kernel_map(key)
            p_var = [v for v in tvars if v.name == p][0]
            q_var = [v for v in tvars if v.name == q][0]
            u, s, v = np.linalg.svd(reader.get_tensor(key))
            smat = np.zeros((u.shape[0], v.shape[0]))
            smat[:s.shape[0], :s.shape[0]] = np.diag(s)
            q_mat = np.dot(smat, v)
            p_mat = u
            tf.logging.info("Tensor: %s %s", p, "*INIT_FROM_CKPT*")
            tf.logging.info("Tensor: %s %s", q, "*INIT_FROM_CKPT*")
            sess.run(tf.assign(p_var, p_mat))
            sess.run(tf.assign(q_var, q_mat))
            tvars_names.remove(p)
            tvars_names.remove(q)
            pass
        elif key + ":0" in tvars_names:
            var = [v for v in tvars if v.name == key + ":0"][0]
            tf.logging.info("Tensor: %s %s", key + ":0", "*INIT_FROM_CKPT*")
            sess.run(tf.assign(var, reader.get_tensor(key)))
            tvars_names.remove(key + ":0")
            pass
        else:
            tf.logging.info("PASSED: %s ", key)
            pass
    for var_name in tvars_names:
        tf.logging.info("Tensor: %s %s", var_name, "*NOT_INIT_FROM_CKPT*")
    saver = tf.train.Saver()
    saver.save(sess, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", help="bert config file")
    parser.add_argument("--checkpoint", help="checkpoint to factorize")
    parser.add_argument("--output_dir", help="output checkpoint directory")
    parser.add_argument(
        "--finetuned", help="whether the checkpoint is finetuned, " +
        "if true then output layer will be loaded", action='store_true')
    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    save_factorized_model(
        bert_config_file=args.bert_config_file,
        init_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        finetuned=args.finetuned)
