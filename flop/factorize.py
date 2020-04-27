from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import factorize
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


def save_factorized_model(bert_config_file, init_checkpoint, output_dir):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    model = modeling_hardconcrete.BertModelHardConcrete(
        config=bert_config,
        is_training=False,
        input_ids=input_ids)
    reader = pywrap_tensorflow.NewCheckpointReader(init_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    kernel_pattern = "^bert/encoder/.*((query|key|value)|(dense))/kernel$"
    bias_pattern = "^bert/encoder/.*((query|key|value)|(dense))/bias$"
    tvar = tf.trainable_variables()
    tvar_names = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for var in tvar:
        tvar_names.append(var.name)
    for key in var_to_shape_map:
        if re.match(bias_pattern, key):
            q = factorize.bias_map(key)
            q_var = [v for v in tvar if v.name == q][0]
            tf.logging.info("Tensor: %s %s", q, "*INIT_FROM_CKPT*")
            sess.run(tf.assign(q_var, reader.get_tensor(key)))
            tvar_names.remove(q)
        elif re.match(kernel_pattern, key):
            p, q = factorize.kernel_map(key)
            p_var = [v for v in tvar if v.name == p][0]
            q_var = [v for v in tvar if v.name == q][0]
            u, s, v = np.linalg.svd(reader.get_tensor(key))
            smat = np.zeros((u.shape[0], v.shape[0]))
            smat[:s.shape[0], :s.shape[0]] = np.diag(s)
            q_mat = np.dot(smat, v)
            p_mat = u
            tf.logging.info("Tensor: %s %s", p, "*INIT_FROM_CKPT*")
            tf.logging.info("Tensor: %s %s", q, "*INIT_FROM_CKPT*")
            sess.run(tf.assign(p_var, p_mat))
            sess.run(tf.assign(q_var, q_mat))
            tvar_names.remove(p)
            tvar_names.remove(q)
            pass
        elif key + ":0" in tvar_names:
            var = [v for v in tvar if v.name == key + ":0"][0]
            tf.logging.info("Tensor: %s %s", key + ":0", "*INIT_FROM_CKPT*")
            sess.run(tf.assign(var, reader.get_tensor(key)))
            tvar_names.remove(key + ":0")
            pass
        else:
            pass
    for var_name in tvar_names:
        tf.logging.info("Tensor: %s %s", var_name, "*NOT_INIT_FROM_CKPT*")
    saver = tf.train.Saver()
    saver.save(sess, output_dir)


if __name__ == "__main__":
    import sys
    sys.path.append(sys.path[0] + "/../bert/")
    import modeling
    import modeling_hardconcrete
    tf.logging.set_verbosity(tf.logging.DEBUG)
    save_factorized_model(
        bert_config_file="./../../uncased_L-12_H-768_A-12/bert_config.json",
        init_checkpoint="./../../uncased_L-12_H-768_A-12/bert_model.ckpt",
        output_dir="./../uncased_L-12_H-768_A-12_f/bert_model_f.ckpt")
