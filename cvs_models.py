
import tensorflow as tf
from Config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, BatchNormalization, Activation, SpatialDropout1D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential


# the standard model
def build_eeg_drone_model_1(input_shape=(2, 128, 14), num_classes=6):

    model = Sequential()

    model.add(tf.keras.Input(shape=input_shape, name='eeg_input'))

    model.add(TimeDistributed(Conv1D(filters=512, kernel_size=8, padding='same'), name='td_conv_1'))
    model.add(BatchNormalization(name = "batch_normalization_1"))
    model.add(TimeDistributed(Activation('elu'), name='td_relu_1'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_1'))
    model.add(TimeDistributed(Dropout(0.3), name='td_dropout_1'))

    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=6, padding='same'), name='td_conv_2'))
    model.add(BatchNormalization(name = "batch_normalization_2"))
    model.add(TimeDistributed(Activation('elu'), name='td_relu_2'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_2'))
    model.add(TimeDistributed(Dropout(0.3), name='td_dropout_2'))
    model.add(TimeDistributed(Flatten(), name='td_flatten'))

    model.add(Bidirectional(LSTM(units=32, return_sequences=False, dropout=0.35, recurrent_dropout=0.3),
                            name='bidirectional_lstm'))

    model.add(Dense(48, activation='relu', kernel_regularizer=l2(0.01), name='combination_ann'))
    model.add(Dropout(0.4, name='final_dropout'))

    # Changed to dynamically use num_classes (6 for your CSV dict)
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# cnn specialist
def build_eeg_drone_model_2(input_shape=(4, 64, 14), num_classes=6):

    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape, name='eeg_input'))

    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=16, padding='valid'), name='spatial_filter'))
    model.add(BatchNormalization(name = "batch_normalization_1"))
    model.add(TimeDistributed(Activation('elu'), name='spatial_elu'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_1'))
    model.add(TimeDistributed(Dropout(0.3), name='td_dropout_1'))

    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=8, padding='same'), name='td_conv_1'))
    model.add(BatchNormalization(name = "batch_normalization_2"))
    model.add(TimeDistributed(Activation('elu'), name='td_elu_1'))
    model.add(TimeDistributed(SpatialDropout1D(0.2), name='td_spatial_drop_1'))

    model.add(TimeDistributed(Conv1D(filters=384, kernel_size=4, padding='same'), name='td_conv_2'))
    model.add(BatchNormalization(name = "batch_normalization_3"))
    model.add(TimeDistributed(Activation('elu'), name='td_elu_2'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_2'))
    model.add(TimeDistributed(SpatialDropout1D(0.3), name='td_spatial_drop_2'))

    model.add(TimeDistributed(Flatten(), name='td_flatten'))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False, dropout=0.3, recurrent_dropout=0.4),
                            name='bidirectional_lstm'))

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='combination_ann'))
    model.add(Dropout(0.4, name='final_dropout'))

    # Changed to dynamically use num_classes
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# compact model
def build_eeg_drone_model_3(input_shape=(2, 128, 14), num_classes=6):
    inputs = tf.keras.Input(shape=input_shape)

    # treat each time window independently first
    x = TimeDistributed(Conv1D(filters=64, kernel_size=16, padding='same', use_bias=True))(inputs)
    x = BatchNormalization()(x)
    x = TimeDistributed(Activation('elu'))(x)

    # depthwise conv — learns spatial filter per channel
    x = TimeDistributed(Conv1D(filters=128, kernel_size=8, padding='valid',use_bias=True, kernel_regularizer=l2(0.01)))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Activation('elu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    # separable conv — learns temporal filter
    x = TimeDistributed(Conv1D(filters=192, kernel_size=4, padding='same',
                               use_bias=False))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Activation('elu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Flatten())(x)

    # lightweight temporal integration
    x = LSTM(units=16, dropout=0.3)(x)

    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# no time window model
def build_eeg_drone_model_4(input_shape=(256, 14), num_classes=6):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(Conv1D(filters=128, kernel_size=16, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=256, kernel_size=8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=512, kernel_size=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=32, return_sequences=False, dropout=0.3)))
    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# the switch case for the different model options
def build_basic_model(num_of_model):
    if num_of_model == 1:
        return build_eeg_drone_model_1()
    elif num_of_model == 2:
        return build_eeg_drone_model_2()
    elif num_of_model == 3:
        return build_eeg_drone_model_3()
    elif num_of_model == 4:
        return build_eeg_drone_model_4()
    else:
        print("No valid model num given - none loaded")
        return None

