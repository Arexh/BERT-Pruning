from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import copy
import os
import re


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

def _get_variable_name(param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name

def save_factorized_model(bert_config_file, init_checkpoint, output_dir):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    modeling_flop.create_model_train(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=None,
        segment_ids=None,
        labels=None,
        use_one_hot_embeddings=False,
        num_labels=2)
    total_loss = tf.constant([0], shape=[], dtype=tf.float32)
    tvars = tf.trainable_variables()
    for param in tvars:
        param_name = _get_variable_name(param.name)
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
    tvars = tf.get_collection(tf.GraphKeys.VARIABLES)
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
            tf.logging.info('PASSED: %s ', key)
            pass
    for var_name in tvars_names:
        tf.logging.info("Tensor: %s %s", var_name, "*NOT_INIT_FROM_CKPT*")
    saver = tf.train.Saver()
    saver.save(sess, output_dir)


if __name__ == "__main__":
    import sys
    sys.path.append(sys.path[0] + "/../bert/")
    import modeling
    import modeling_flop
    import optimization_flop
    tf.logging.set_verbosity(tf.logging.DEBUG)
    save_factorized_model(
        bert_config_file="./../../uncased_L-12_H-768_A-12/bert_config.json",
        init_checkpoint="/../../fine_tune_outputs/SST-2/lr_3e-5/model.ckpt-6313",
        output_dir="./../uncased_L-12_H-768_A-12_SST-2_f/bert_model_f.ckpt")