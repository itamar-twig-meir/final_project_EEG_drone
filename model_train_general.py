from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#region load data and basic shaping

label_dictionary = {
    'Blank': 0,
    'Camera': 1,
    'Up': 2,
    'Down': 3,
    'Left': 4,
    'Right': 5
}


eeg_data = pd.read_csv('../drone_project_data/merged_result_256.csv')

num_of_readings_in_thought = 256
num_of_time_window = 8
eeg_channels = 14
num_of_single_readings = eeg_data.shape[0]
num_of_full_readings = int(num_of_single_readings / num_of_readings_in_thought)



def apply_ica_to_csv(csv_data, sfreq=256):

    eeg_only = csv_data.iloc[:, 1:15].values.T / 1e6


    ch_names = [h.replace('eeg.', '').upper() for h in csv_data.columns[1:15].tolist()] # give the eeg data names for metadata
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')#makes metadata
    raw = mne.io.RawArray(eeg_only, info)#combine the 2
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    raw.filter(l_freq=1.0, h_freq=None, verbose=False)

    ica = mne.preprocessing.ICA(n_components=13, random_state=42, method='fastica')
    ica.fit(raw)

    # 3. Apply exclusions
    ica.exclude = [0, 2, 4, 5, 12]
    raw_clean = ica.apply(raw)

    clean_np = raw_clean.get_data().T * 1e6

    # Create a new DataFrame with the same headers
    clean_df = pd.DataFrame(clean_np, columns=csv_data.columns[1:15])

    # Re-add the labels
    clean_df['label'] = csv_data.iloc[:, 15].values

    return clean_df


def make_total_sample_answer_couples(eeg_data, num_of_readings_in_thought= 256, num_of_time_window = 4):

    num_of_single_readings = eeg_data.shape[0]

    if (num_of_single_readings / num_of_readings_in_thought).is_integer():
        num_of_thoughts = int(num_of_single_readings / num_of_readings_in_thought)
    else:
        print(f"num of single readings not devisable by num of readings in thought ({num_of_readings_in_thought}) \
        currently {num_of_single_readings} readings")
        exit()
    if not (num_of_readings_in_thought / num_of_time_window).is_integer():
        print(f"num of full readings not devisable by num of time windows ({num_of_time_window}) \
              currently {num_of_thoughts} full readings ")

    scaler = StandardScaler()

    X_total = scaler.fit_transform(eeg_data.iloc[:, :14].values.astype(np.float32))
    X_total = X_total.reshape(-1, num_of_time_window, int(num_of_readings_in_thought/num_of_time_window), 14)
    Y_total = eeg_data.iloc[::256,14].values

    return np.array(X_total), np.array(Y_total)

def turn_Y_array_into_one_hot(Y_array):
    Y_array= np.array([label_dictionary[label] for label in Y_array])
    return tf.keras.utils.to_categorical(Y_array, num_classes=len(label_dictionary))

eeg_data = apply_ica_to_csv(eeg_data)
X_total, Y_total = make_total_sample_answer_couples(eeg_data, num_of_time_window = num_of_time_window)

Y_total = turn_Y_array_into_one_hot(Y_total)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_total, Y_total, test_size=0.2, shuffle=True, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, shuffle=True, random_state=42)
# I used a sklearn func to both split and shuffle my data into train test valedate groups



train_ok = X_train.shape[0] == Y_train.shape[0]
test_ok = X_test.shape[0] == Y_test.shape[0]
validation_ok = X_val.shape[0] == Y_val.shape[0]

print(f"num of full readings - {num_of_full_readings} \n "
      +f"num of x train examples - {X_train.shape[0]} train ok - {train_ok}, x train shape - {X_train.shape} \n "
      +f"num of X test examples - {X_test.shape[0]} test ok - {test_ok}, x test shape - {X_test.shape} \n "
      +f"num of X test examples - {X_val.shape[0]} validation ok - {validation_ok}, x validation shape - {X_val.shape} \n "
      )

model_num = input("whats the model num? ")

#endregion


#region building the model
model = Sequential()

model.add(tf.keras.Input(shape=(num_of_time_window,int(num_of_readings_in_thought/num_of_time_window),eeg_channels), name = 'eeg_input'))


model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, padding='same'), name='td_conv_1'))
model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_1'))
model.add(TimeDistributed(Activation('elu'), name='td_relu_1'))

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=8, padding='same'), name='td_conv_2'))
model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_2'))
model.add(TimeDistributed(Activation('elu'), name='td_relu_2'))
model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_2'))
model.add(TimeDistributed(Dropout(0.3), name='td_dropout_2'))
model.add(TimeDistributed(Flatten(), name='td_flatten'))


model.add(Bidirectional(LSTM(units=32, return_sequences=False, dropout=0.4,recurrent_dropout=0.4), name='bidirectional_lstm'))

model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='combination_ann'))
model.add(Dropout(0.5, name='final_dropout'))

model.add(Dense(6, activation='softmax', name='output_layer'))

"""model.summary()"""
print("\n \n")

#endregion


#region compiling, training, regulation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    ),
    ModelCheckpoint(
        filepath=f'drone_eeg_model_{model_num}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    verbose=1
)

# 1. Evaluate on the unseen test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 2. Plot Training History
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('EEG Model Accuracy')
plt.legend()
plt.show()

#endregion

