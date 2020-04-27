from modeling import *


class BertModelHardConcrete(BertModel):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
          scope: (optional) variable scope. Defaults to "bert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model_train(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(
                    self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))


def attention_layer_train(from_tensor,
                          to_tensor,
                          attention_mask=None,
                          num_attention_heads=1,
                          size_per_head=512,
                          query_act=None,
                          key_act=None,
                          value_act=None,
                          attention_probs_dropout_prob=0.0,
                          initializer_range=0.02,
                          do_return_2d_tensor=False,
                          batch_size=None,
                          from_seq_length=None,
                          to_seq_length=None):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # query layer matrix factorized here
    query_layer_p = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=None,
        use_bias=False,
        name="query_p",
        kernel_initializer=create_initializer(initializer_range))

    query_layer = tf.layers.dense(
        query_layer_p,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query_q",
        kernel_initializer=create_initializer(initializer_range))

    # # `query_layer` = [B*F, N*H]
    # query_layer = tf.layers.dense(
    #     from_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=query_act,
    #     name="query",
    #     kernel_initializer=create_initializer(initializer_range))

    # key layer matrix factorized here
    key_layer_p = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=None,
        use_bias=False,
        name="key_p",
        kernel_initializer=create_initializer(initializer_range))

    key_layer = tf.layers.dense(
        key_layer_p,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key_q",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    # key_layer = tf.layers.dense(
    #     to_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=key_act,
    #     name="key",
    #     kernel_initializer=create_initializer(initializer_range))

    # value layer matrix factorized here
    value_layer_p = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=None,
        use_bias=False,
        name="value_p",
        kernel_initializer=create_initializer(initializer_range))

    value_layer = tf.layers.dense(
        value_layer_p,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value_q",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    # value_layer = tf.layers.dense(
    #     to_tensor_2d,
    #     num_attention_heads * size_per_head,
    #     activation=value_act,
    #     name="value",
    #     kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def transformer_model_train(input_tensor,
                            attention_mask=None,
                            hidden_size=768,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            intermediate_act_fn=gelu,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            initializer_range=0.02,
                            do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer_train(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    # attention output fractorized here
                    attention_output_p = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        use_bias=False,
                        name="dense_p",
                        kernel_initializer=create_initializer(initializer_range))

                    attention_output = tf.layers.dense(
                        attention_output_p,
                        hidden_size,
                        name="dense_q",
                        kernel_initializer=create_initializer(initializer_range))

                    # attention_output = tf.layers.dense(
                    #     attention_output,
                    #     hidden_size,
                    #     kernel_initializer=create_initializer(initializer_range))

                    attention_output = dropout(
                        attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                # intermidiate output fractorized here
                intermediate_output_p = tf.layers.dense(
                    attention_output,
                    hidden_size,
                    activation=None,
                    use_bias=False,
                    name='dense_p',
                    kernel_initializer=create_initializer(initializer_range))

                intermediate_output = tf.layers.dense(
                    intermediate_output_p,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    name='dense_q',
                    kernel_initializer=create_initializer(initializer_range))

                # intermediate_output = tf.layers.dense(
                #     attention_output,
                #     intermediate_size,
                #     activation=intermediate_act_fn,
                #     kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                # layer output fractorized here
                layer_output_p = tf.layers.dense(
                    intermediate_output,
                    intermediate_size,
                    use_bias=False,
                    name="dense_p",
                    kernel_initializer=create_initializer(initializer_range))

                layer_output = tf.layers.dense(
                    layer_output_p,
                    hidden_size,
                    name="dense_q",
                    kernel_initializer=create_initializer(initializer_range))

                # layer_output = tf.layers.dense(
                #     intermediate_output,
                #     hidden_size,
                #     kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def create_model_train(bert_config, is_training, input_ids, input_mask, segment_ids,
                       labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = BertModelHardConcrete(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    sts = True if num_labels == 0 else False
    num_labels = max(num_labels, 1)

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        if not sts:
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = - \
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        else:
            probabilities = None
            logits = tf.squeeze(logits, [-1])
            per_example_loss = tf.square(logits - labels)

        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder_train(bert_config, num_labels, init_checkpoint, learning_rate,
                           num_train_steps, num_warmup_steps, use_tpu,
                           use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(
                features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        sts = True if num_labels == 0 else False

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                precision = tf.metrics.precision(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                recall = tf.metrics.recall(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(
                    values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "precision": precision,
                    "recall": recall,
                }

            def metric_fn_sts(per_example_loss, label_ids, logits, is_real_example):
                # Display labels and predictions
                concat1 = tf.contrib.metrics.streaming_concat(logits)
                concat2 = tf.contrib.metrics.streaming_concat(label_ids)

                # Compute Pearson correlation
                pearson = tf.contrib.metrics.streaming_pearson_correlation(
                    logits, label_ids)

                # Compute MSE
                # mse = tf.metrics.mean(per_example_loss)
                mse = tf.metrics.mean_squared_error(label_ids, logits)

                # Compute Spearman correlation
                size = tf.size(logits)
                indice_of_ranks_pred = tf.nn.top_k(logits, k=size)[1]
                indice_of_ranks_label = tf.nn.top_k(label_ids, k=size)[1]
                rank_pred = tf.nn.top_k(-indice_of_ranks_pred, k=size)[1]
                rank_label = tf.nn.top_k(-indice_of_ranks_label, k=size)[1]
                rank_pred = tf.to_float(rank_pred)
                rank_label = tf.to_float(rank_label)
                spearman = tf.contrib.metrics.streaming_pearson_correlation(
                    rank_pred, rank_label)

                return {'pred': concat1, 'label_ids': concat2, 'pearson': pearson, 'spearman': spearman, 'MSE': mse}

            if sts:
                eval_metrics = (metric_fn_sts,
                                [per_example_loss, label_ids, logits, is_real_example])
            else:
                eval_metrics = (metric_fn,
                                [per_example_loss, label_ids, logits, is_real_example])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            if sts:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"logits": logits},
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities},
                    scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
