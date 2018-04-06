import tensorflow as tf

class Network(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, activation_function, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_y_domain = tf.placeholder(tf.float32, [None, num_classes], name="input_y_domain")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l2_loss_domain = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity

                if activation_function=="tanh" :
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="activation_function")
                elif activation_function=="elu" :
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="activation_function")
                elif activation_function=="softplus" :
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="activation_function")
                else :
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="activation_function")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Domain layers (unnormalized) scores and predictions
        with tf.name_scope("domain_output"):
            # W_d = tf.get_variable(
            #     "W",
            #     shape=[num_filters_total, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            b_d = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss_domain += tf.nn.l2_loss(W)
            l2_loss_domain += tf.nn.l2_loss(b_d)
            self.scores_domain = tf.nn.xw_plus_b(self.h_drop, W, b_d, name="domain_scores")
            self.predictions_domain = tf.argmax(self.scores_domain, 1, name="domain_predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("domain_loss"):
            losses_domain = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_domain, labels=self.input_y_domain)
            self.loss_domain = tf.reduce_mean(losses_domain) + l2_reg_lambda * l2_loss_domain

        # Accuracy
        with tf.name_scope("domain_accuracy"):
            correct_predictions_domain = tf.equal(self.predictions_domain, tf.argmax(self.input_y_domain, 1))
            self.accuracy_domain = tf.reduce_mean(tf.cast(correct_predictions_domain, "float"), name="domain_accuracy")
