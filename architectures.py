import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import *
from keras import Input
from keras import initializers, regularizers
from keras.models import Model

from config_parser import Parser

CLASS_ACTIVATION = 'sigmoid'


def get_spectrogram_encoder(input_shapes, L):
    initializer = initializers.he_uniform()

    input = Input(input_shapes)
    X = input

    _, _, channels = input_shapes

    norm = BatchNormalization(name='motion_norm_1')
    X = norm(X)

    padding = ZeroPadding2D(padding=(1, 1))  # same padding
    cnn = Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_1'
    )
    norm = BatchNormalization(name='motion_norm_2')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = norm(X)
    X = activation(X)
    X = pooling(X)

    padding = ZeroPadding2D(padding=(1, 1))  # same padding
    cnn = Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_2'
    )
    norm = BatchNormalization(name='motion_norm_3')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = norm(X)
    X = activation(X)
    X = pooling(X)

    cnn = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_3'
    )
    norm = BatchNormalization(name='motion_norm_4')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = cnn(X)
    X = norm(X)
    X = activation(X)
    X = pooling(X)

    flatten = Flatten()

    X = flatten(X)

    dense = Dense(units=128,
                  kernel_initializer=initializer,
                  name='motion_dense_1')
    norm = BatchNormalization(name='motion_norm_5')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(units=L,
                  kernel_initializer=initializer,
                  name='motion_dense_2')
    norm = BatchNormalization(name='motion_norm_6')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    output = X

    return Model(inputs=input,
                 outputs=output,
                 name='spectrogram_encoder')


def get_location_encoder(input_shapes, L):
    initializer = initializers.he_uniform()
    window_shape = input_shapes[0]
    features_shape = input_shapes[1]

    window = Input(shape=window_shape)
    features = Input(shape=features_shape)

    X = window

    norm = BatchNormalization(name='location_norm_1')
    X = norm(X)

    lstm = Bidirectional(LSTM(units=128, name='location_BiLSTM'))
    X = lstm(X)

    X = tf.concat([X, features], axis=1)

    dense = Dense(
        units=128,
        kernel_initializer=initializer,
        name='location_dense_1'
    )
    norm = BatchNormalization(name='location_norm_2')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(
        units=64,
        kernel_initializer=initializer,
        name='location_dense_2'
    )
    norm = BatchNormalization(name='location_norm_3')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(
        units=L,
        kernel_initializer=initializer,
        name='location_dense_3'
    )
    norm = BatchNormalization(name='location_norm_4')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    output = X

    return Model(inputs=[window, features],
                 outputs=output,
                 name='location_encoder')


def get_attention_mechanism(L, D):
    initializer = initializers.glorot_uniform()
    regularizer = regularizers.l2(0.01)

    encodings = Input(shape=L)

    D_layer = Dense(units=D,
                    activation='tanh',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name='D_layer')
    G_layer = Dense(units=D,
                    activation='sigmoid',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name='G_layer')
    K_layer = Dense(units=1,
                    kernel_initializer=initializer,
                    name='K_layer')

    attention_ws = D_layer(encodings)
    attention_ws = attention_ws * G_layer(encodings)
    attention_ws = K_layer(attention_ws)

    return Model(inputs=encodings,
                 outputs=attention_ws,
                 name='MIL_attention')


def get_classifier(L, n_units=8):
    input = keras.Input(shape=L)

    X = input
    dense = Dense(
        units=L // 2,
        kernel_initializer=initializers.he_uniform(),
        name='head_dense'
    )
    norm = BatchNormalization(name='head_norm')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(units=n_units,
                  kernel_initializer=initializers.glorot_uniform(),
                  name='final_dense')
    X = dense(X)

    activation = tf.keras.layers.Activation(activation=CLASS_ACTIVATION, name='class_activation')
    y_pred = activation(X)

    return Model(inputs=input,
                 outputs=y_pred,
                 name='classifier')


def get_fusion_MIL(input_shapes):
    conf = Parser()
    conf.get_args()

    motion_shape = list(input_shapes[0])
    loc_w_shape = list(input_shapes[1])
    loc_fts_shape = list(input_shapes[2])

    mot_bag_size = motion_shape[0]
    loc_bag_size = loc_w_shape[0]
    motion_size = motion_shape[1:]

    motion_encoder = get_spectrogram_encoder(motion_size, conf.L)

    location_encoder = get_location_encoder([loc_w_shape[1:], loc_fts_shape[1:]], conf.L)

    MIL_attention = get_attention_mechanism(conf.L,256)
    att_softmax = Softmax(name='attention_softmax')

    n_classes = 2
    classifier = get_classifier(conf.L, n_classes)

    motion_bags = Input(motion_shape)
    loc_w_bags = Input(loc_w_shape)
    loc_fts_bags = Input(loc_fts_shape)

    batch_size = tf.shape(motion_bags)[0]
    motion_instances = tf.reshape(motion_bags, (batch_size * mot_bag_size, *motion_size))
    loc_w_instances = tf.reshape(loc_w_bags, (batch_size * loc_bag_size, *loc_w_shape[1:]))
    loc_fts_instances = tf.reshape(loc_fts_bags, (batch_size * loc_bag_size, *loc_fts_shape[1:]))

    motion_encodings = motion_encoder(motion_instances)
    location_encodings = location_encoder([loc_w_instances, loc_fts_instances])

    motion_encodings = tf.reshape(motion_encodings, (batch_size, mot_bag_size, conf.L))
    location_encodings = tf.reshape(location_encodings, (batch_size, loc_bag_size, conf.L))

    encodings = concatenate([motion_encodings, location_encodings], axis=-2)

    encodings = tf.reshape(encodings, (batch_size * (mot_bag_size + loc_bag_size), conf.L))
    attention_ws = MIL_attention(encodings)
    attention_ws = tf.reshape(attention_ws, (batch_size, mot_bag_size + loc_bag_size))

    attention_ws = tf.expand_dims(att_softmax(attention_ws), -2)
    encodings = tf.reshape(encodings, (batch_size, mot_bag_size + loc_bag_size, conf.L))
    flatten = Flatten()
    pooling = flatten(tf.matmul(attention_ws, encodings))

    y_pred = classifier(pooling)

    return Model([motion_bags, loc_w_bags, loc_fts_bags], y_pred)
