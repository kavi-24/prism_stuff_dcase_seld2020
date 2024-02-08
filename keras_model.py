#
# The SELDnet architecture
#

import keras

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate
from keras.layers import Dense, Activation, Dropout, Reshape, Permute, GRU, BatchNormalization, TimeDistributed

from keras.models import Model
from keras.models import load_model

from keras.optimizers import Adam
keras.backend.set_image_data_format('channels_first')
# keras.backend.set_image_data_format("channels_last")



def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(
            3, 3), padding='same')(spec_cnn)
        print("Conv2D: {}".format(spec_cnn))
        spec_cnn = BatchNormalization()(spec_cnn)
        print("BatchNormalization: {}".format(spec_cnn))
        spec_cnn = Activation('relu')(spec_cnn)
        print("Activation: {}".format(spec_cnn))
        spec_cnn = MaxPooling2D(pool_size=(
            t_pool_size[i], f_pool_size[i]))(spec_cnn)
        print("MaxPooling2D".format(spec_cnn))
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
        print("DropOut: {}".format(spec_cnn))
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    print("Permute: {}".format(spec_cnn))

    # RNN
    # spec_rnn = Reshape((data_out[0][-2], -1))(spec_cnn) # error: total size of new array must be unchanged, input_shape =  [9, 2, 64], output_shape = [60, -1]
    # spec_rnn = Reshape((9, 2, 64))(spec_cnn) # error: total size of new array must be unchanged, input_shape =  [9, 2, 64], output_shape = [60, -1]
    spec_rnn = Reshape(target_shape=(data_out[0][-2], -1))(spec_cnn)
    print(spec_rnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
        print(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = None
    if doa_objective == 'mse':
        model = Model(inputs=spec_start, outputs=[sed, doa])
        model.compile(optimizer=Adam(), loss=[
                      'binary_crossentropy', 'mse'], loss_weights=weights)
    elif doa_objective == 'masked_mse':
        doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
        model = Model(inputs=spec_start, outputs=[sed, doa_concat])
        model.compile(optimizer=Adam(), loss=[
                      'binary_crossentropy', masked_mse], loss_weights=weights)
    else:
        print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
        exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    # TODO fix this hardcoded value of number of classes
    sed_out = y_gt[:, :, :14] >= 0.5
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective == 'mse':
        return load_model(model_file)
    elif doa_objective == 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()
