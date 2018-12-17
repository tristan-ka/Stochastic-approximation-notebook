DTYPE = 'float32'
DTYPE_INT = 'int64'

projection_dim = 16

cnn_options = {"embedding": {"dim": 4}, "filters": [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]], "n_highway": 2, "n_characters": 262, "max_characters_per_token": 50, "activation": "relu"}
filters = cnn_options['filters']
max_word_length = cnn_options['max_characters_per_token']
n_filters = sum(f[1] for f in filters)
max_chars = cnn_options['max_characters_per_token']
char_embed_dim = cnn_options['embedding']['dim']
n_chars = cnn_options['n_characters']

ids_placeholder = tf.placeholder('int32', shape=(None, None, max_word_length))
                                 
if n_chars != 262:
    raise InvalidNumberOfCharacters("Set n_characters=262 after training see the README.md")
                                 
if cnn_options['activation'] == 'tanh':
    activation = tf.nn.tanh
elif cnn_options['activation'] == 'relu':
    activation = tf.nn.relu

# the character embeddings
with tf.device("/cpu:0"):
    embedding_weights = tf.get_variable(
            "char_embed", [n_chars, char_embed_dim],
            dtype=DTYPE,
            initializer=tf.random_uniform_initializer(-1.0, 1.0)
    )
    # shape (batch_size, unroll_steps, max_chars, embed_dim)
    char_embedding = tf.nn.embedding_lookup(embedding_weights,
                                            ids_placeholder)
                                            
                                            # the convolutions
def make_convolutions(inp):
    with tf.variable_scope('CNN') as scope:
        convolutions = []
        for i, (width, num) in enumerate(filters):
            if cnn_options['activation'] == 'relu':
                # He initialization for ReLU activation
                # with char embeddings init between -1 and 1
                #w_init = tf.random_normal_initializer(
                #    mean=0.0,
                #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                #)

                # Kim et al 2015, +/- 0.05
                w_init = tf.random_uniform_initializer(
                    minval=-0.05, maxval=0.05)
            elif cnn_options['activation'] == 'tanh':
                # glorot init
                w_init = tf.random_normal_initializer(
                    mean=0.0,
                    stddev=np.sqrt(1.0 / (width * char_embed_dim))
                )
            w = tf.get_variable(
                "W_cnn_%s" % i,
                [1, width, char_embed_dim, num],
                initializer=w_init,
                dtype=DTYPE)
            b = tf.get_variable(
                "b_cnn_%s" % i, [num], dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(
                    inp, w,
                    strides=[1, 1, 1, 1],
                    padding="VALID") + b
            # now max pool
            
            print(conv)
            conv = tf.nn.max_pool(
                    conv, [1, 1, max_chars-width+1, 1],
                    [1, 1, 1, 1], 'VALID')

            # activation
            conv = activation(conv)
            conv = tf.squeeze(conv, squeeze_dims=[2])

            convolutions.append(conv)
            print(convolutions)
    return tf.concat(convolutions, 2)

embedding = make_convolutions(char_embedding)
