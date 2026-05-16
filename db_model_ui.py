import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, \
    BatchNormalization, Activation, SpatialDropout1D, Concatenate
from tensorflow.keras.regularizers import l2
import os
import moabb
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

# Suppress excessive MNE/MOABB logs to keep the console clean
moabb.set_log_level("ERROR")

HARDWARE_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

physionet_to_drone_map = {
    'rest': 0,
    'hands': 1,
    'feet': 2,
    'left_hand': 3,
    'right_hand': 4
}

label_dictionary = {
    'Blank': 0, 'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4
}


def fetch_moabb_data(subject_ids):

    print(f"\nFetching data for {len(subject_ids)} subjects")

    # Define the paradigm: 4-second trials, specific channels, standard MI bandpass
    paradigm = MotorImagery(fmin=7.0, fmax=35.0,channels=HARDWARE_CHANNELS,resample=160.0)

    dataset = PhysionetMI()
    # Filter out Subject 88, problem with db
    safe_subjects = [s for s in subject_ids if s != 88]
    if len(safe_subjects) < len(subject_ids):
        print("Warning: Subject 88 removed automatically due to 128Hz sampling rate mismatch.")

    # X shape: (Trials, Channels, Time) -> e.g., (1000, 14, 640)
    # labels: list of strings -> e.g., ['left_hand', 'rest', ...]
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=safe_subjects)

    return X, labels


def format_data_for_model(X_moabb, labels_moabb, num_of_time_window=6):

    X = np.transpose(X_moabb, (0, 2, 1))
    samples, timepoints, channels = X.shape

    scaler = StandardScaler()
    X_flat = X.reshape(-1, channels)
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(samples, timepoints, channels)

    readings_per_window = int(timepoints / num_of_time_window)
    valid_timepoints = readings_per_window * num_of_time_window

    X_scaled = X_scaled[:, :valid_timepoints, :]

    X_final = X_scaled.reshape(samples, num_of_time_window, readings_per_window, channels)
    X_final = X_scaled.reshape(samples, num_of_time_window, readings_per_window, channels)

    valid_indices = [i for i, label in enumerate(labels_moabb) if label in physionet_to_drone_map]

    X_filtered = X_final[valid_indices]
    Y_numeric = [physionet_to_drone_map[labels_moabb[i]] for i in valid_indices]
    Y_one_hot = tf.keras.utils.to_categorical(Y_numeric, num_classes=len(label_dictionary))

    return X_filtered, Y_one_hot


def load_eeg_model(model_name):
    """Loads a pre-trained .keras model."""
    if not model_name.endswith('.keras'):
        model_name += '.keras'
    if os.path.exists(model_name):
        try:
            model = tf.keras.models.load_model(model_name)
            print(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            print(f"Error loading the model file: {e}")
            return None
    else:
        print(f"Error: The file '{model_name}' was not found.")
        return None


def plot_confusion_matrix(y_true, y_pred, title):
    # Convert one-hot to integers
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Get the class names from your existing label_dictionary keys
    class_names = list(label_dictionary.keys())

    # Calculate matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title)
    plt.show()


def get_high_certainty_report(model, X, Y_true, threshold=0.70):

    preds = model.predict(X, verbose=0)

    certainties = np.max(preds, axis=1)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(Y_true, axis=1)

    high_conf_mask = certainties >= threshold
    num_high_conf = np.sum(high_conf_mask)
    if num_high_conf == 0:
        return 0.0, 0

    correct_high_conf = np.sum(predicted_classes[high_conf_mask] == true_classes[high_conf_mask])
    acc = (correct_high_conf / num_high_conf) * 100
    return acc, num_high_conf


def train_on_moabb(model):
    train_subjects = list(range(1, 100))
    X_raw, Y_raw = fetch_moabb_data(train_subjects)

    print("Formatting data for CNN-LSTM...")
    X, Y = format_data_for_model(X_raw, Y_raw)

    # 1. Split the data naturally (stratify ensures the natural imbalance is preserved equally)
    X_temp, X_local_test, Y_temp, Y_local_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.176, random_state=42, stratify=Y_temp
    )

    print(f"\nData Split Complete:")
    print(f" - Train: {len(X_train)} samples")
    print(f" - Validation: {len(X_val)} samples")
    print(f" - Local Test: {len(X_local_test)} samples")
    print("-" * 30)

    # 2. Calculate Class Weights mathematically to handle the imbalance
    # This replaces the need to delete data!
    y_train_integers = np.argmax(Y_train, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_integers),
        y=y_train_integers
    )
    class_weights_dict = dict(enumerate(weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # 3. Train the model using 100% of X_train, passing in the class_weight dictionary
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, Y_val),
        class_weight=class_weights_dict,  # <--- THIS IS THE MAGIC BULLET
        callbacks=callbacks,
        verbose=1
    )

    print("\n--- Final Performance Report ---")

    # 4. Evaluate on the natural distributions
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    t_hc_acc, t_hc_count = get_high_certainty_report(model, X_train, Y_train)
    print(
        f"Train Run - Overall Acc: {train_acc * 100:.2f}% | 90%+ Certainty Acc: {t_hc_acc:.2f}% (Count: {t_hc_count})")

    local_loss, local_acc = model.evaluate(X_local_test, Y_local_test, verbose=0)
    l_hc_acc, l_hc_count = get_high_certainty_report(model, X_local_test, Y_local_test)
    print(
        f"Local Test - Overall Acc: {local_acc * 100:.2f}% | 90%+ Certainty Acc: {l_hc_acc:.2f}% (Count: {l_hc_count})")

    y_pred_local = model.predict(X_local_test, verbose=0)
    plot_confusion_matrix(Y_local_test, y_pred_local, "Confusion Matrix: Local Test (Subjects 1-99)")

    save_choice = input("\nTraining complete. Save model? (yes/no): ").strip().lower()
    if save_choice == "yes":
        model_name = input("Enter model name: ")
        model.save(f"{model_name}.keras")
        print(f"Model saved as {model_name}.keras")


def test_on_unseen_subjects(model):
    test_subjects = list(range(100, 110))
    X_raw, Y_raw = fetch_moabb_data(test_subjects)

    print("Formatting test data...")
    X_test, Y_test = format_data_for_model(X_raw, Y_raw)

    print(f"\n--- Evaluation on Unseen Human Data (Subjects 100-109) ---")
    print(f"Test size: {len(X_test)} samples (Natural Distribution)")

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    u_hc_acc, u_hc_count = get_high_certainty_report(model, X_test, Y_test)

    print(f"Unseen Data - Overall Acc: {accuracy * 100:.2f}%")
    print(f"Unseen Data - 90%+ Certainty Acc: {u_hc_acc:.2f}% (Count: {u_hc_count})")

    y_pred_unseen = model.predict(X_test, verbose=0)
    plot_confusion_matrix(Y_test, y_pred_unseen, "Confusion Matrix: Unseen Humans (Natural Distribution)")


def train_and_test_single_subject(model):
    print("\n--- Single Subject Training ---")
    try:
        subject_id = int(input("Enter the Subject ID you want to train and test on (e.g., 1): ").strip())
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # 1. Fetch and format data for just this ONE subject
    X_raw, Y_raw = fetch_moabb_data([subject_id])
    if len(X_raw) == 0:
        print(f"Error: No data found for Subject {subject_id}.")
        return

    print(f"Formatting data for Subject {subject_id}...")
    X, Y = format_data_for_model(X_raw, Y_raw)

    # 2. Split data: 70% Train, 15% Validation, 15% Test
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.176, random_state=42, stratify=Y_temp
    )

    print(f"\nData Split Complete for Subject {subject_id}:")
    print(f" - Train: {len(X_train)} samples")
    print(f" - Validation: {len(X_val)} samples")
    print(f" - Test: {len(X_test)} samples")
    print("-" * 30)

    # 3. Handle class weights for this specific subject
    y_train_integers = np.argmax(Y_train, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_integers),
        y=y_train_integers
    )
    class_weights_dict = dict(enumerate(weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20, # Increased slightly because single-subject data is smaller/noisier
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # 4. Train the model on this specific subject
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=16, # Smaller batch size is often better for single-subject EEG
        validation_data=(X_val, Y_val),
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate and Report
    print(f"\n--- Final Performance Report: Subject {subject_id} ---")
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    t_hc_acc, t_hc_count = get_high_certainty_report(model, X_train, Y_train)
    print(f"Train Run - Overall Acc: {train_acc * 100:.2f}% | 90%+ Certainty Acc: {t_hc_acc:.2f}% (Count: {t_hc_count})")

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    l_hc_acc, l_hc_count = get_high_certainty_report(model, X_test, Y_test)
    print(f"Test Run  - Overall Acc: {test_acc * 100:.2f}% | 90%+ Certainty Acc: {l_hc_acc:.2f}% (Count: {l_hc_count})")

    y_pred_test = model.predict(X_test, verbose=0)
    plot_confusion_matrix(Y_test, y_pred_test, f"Confusion Matrix: Subject {subject_id} Only")

    # 6. Save option
    save_choice = input("\nTraining complete. Save this subject-specific model? (yes/no): ").strip().lower()
    if save_choice == "yes":
        model_name = input("Enter model name: ")
        model.save(f"{model_name}.keras")
        print(f"Model saved as {model_name}.keras")


def save_model_to_keras(model):
    path = input("Enter file path to save model (e.g., /path/to/model.keras): ").strip()

    if not path.endswith('.keras'):
        path += '.keras'

    try:
        model.save(path)
        print(f"Model successfully saved to: {path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def build_eeg_drone_model_1(input_shape = (6, 80, 14)):

    model = Sequential()

    model.add(tf.keras.Input(shape=input_shape,name='eeg_input'))

    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=8, padding='same'), name='td_conv_1'))
    model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_1'))
    model.add(TimeDistributed(Activation('elu'), name='td_relu_1'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=8, padding='same'), name='td_conv_2'))
    model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_2'))
    model.add(TimeDistributed(Activation('elu'), name='td_relu_2'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_2'))
    model.add(TimeDistributed(Dropout(0.2), name='td_dropout_2'))
    model.add(TimeDistributed(Flatten(), name='td_flatten'))

    model.add(Bidirectional(LSTM(units=32, return_sequences=False, dropout=0.3, recurrent_dropout=0.4),
                            name='bidirectional_lstm'))

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='combination_ann'))
    model.add(Dropout(0.4, name='final_dropout'))

    model.add(Dense(5, activation='softmax', name='output_layer'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_eeg_drone_model_2(input_shape = (6, 80, 14)):

    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape, name='eeg_input'))

    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, padding='valid'), name='spatial_filter'))
    model.add(TimeDistributed(BatchNormalization(), name='spatial_batch_norm'))
    model.add(TimeDistributed(Activation('elu'), name='spatial_elu'))

    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=8, padding='same'), name='td_conv_1'))
    model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_1'))
    model.add(TimeDistributed(Activation('elu'), name='td_elu_1'))
    # SpatialDropout drops entire feature maps, forcing the model not to rely on just one good signal
    model.add(TimeDistributed(SpatialDropout1D(0.2), name='td_spatial_drop_1'))

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=4, padding='same'), name='td_conv_2'))
    model.add(TimeDistributed(BatchNormalization(), name='td_batch_norm_2'))
    model.add(TimeDistributed(Activation('elu'), name='td_elu_2'))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2), name='td_max_pool_2'))
    model.add(TimeDistributed(SpatialDropout1D(0.3), name='td_spatial_drop_2'))

    model.add(TimeDistributed(Flatten(), name='td_flatten'))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False, dropout=0.3, recurrent_dropout=0.4),
                            name='bidirectional_lstm'))

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='combination_ann'))
    model.add(Dropout(0.4, name='final_dropout'))

    model.add(Dense(5, activation='softmax', name='output_layer'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_eeg_drone_model_3(input_shape = (6, 80, 14)):

    inputs = tf.keras.Input(shape=input_shape, name='eeg_input')

    x = TimeDistributed(Conv1D(filters=16, kernel_size=1, padding='valid'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)

    branch_1 = TimeDistributed(Conv1D(filters=16, kernel_size=2, padding='same'))(x)
    branch_1 = TimeDistributed(BatchNormalization())(branch_1)
    branch_1 = TimeDistributed(Activation('elu'))(branch_1)

    branch_2 = TimeDistributed(Conv1D(filters=16, kernel_size=4, padding='same'))(x)
    branch_2 = TimeDistributed(BatchNormalization())(branch_2)
    branch_2 = TimeDistributed(Activation('elu'))(branch_2)

    branch_3 = TimeDistributed(Conv1D(filters=16, kernel_size=8, padding='same'))(x)
    branch_3 = TimeDistributed(BatchNormalization())(branch_3)
    branch_3 = TimeDistributed(Activation('elu'))(branch_3)

    x = Concatenate(axis=-1)([branch_1, branch_2, branch_3])


    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(SpatialDropout1D(0.4))(x)

    x = TimeDistributed(Flatten())(x)


    x = Bidirectional(LSTM(units=64, return_sequences=False, dropout=0.35, recurrent_dropout=0.3))(x)


    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)


    outputs = Dense(5, activation='softmax', name='output_layer')(x)

    # Build the final Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_build_type(num_of_model):

    if num_of_model == 1:
        return build_eeg_drone_model_1()
    elif num_of_model == 2:
        return build_eeg_drone_model_2()
    elif num_of_model == 3 :
        return build_eeg_drone_model_3()
    else:
        print("no valid model num given - none loaded")
        return None






current_model = None

while True:
    print("\n" + "=" * 35)
    print("   EEG model - MOABB Training UI")
    print("=" * 35)
    print(f"Model Loaded: {current_model is not None}")
    print("-" * 35)
    print("1. Load Existing Model (.keras)")
    print("2. build a basic model(.keras)")
    print("3. Train Model on DB (Subjects 1-99)")
    print("4. Test Model on Unseen Humans (Subjects 100-109)")
    print("5. save model")
    print("6. Exit")

    choice = input("\nSelect an option (1-6): ").strip()

    if choice == '1':
        name = input("Enter model file name (without .keras): ")
        if name == "yes": name = "../model_versions_general/first_generall_model_48acc"
        current_model = load_eeg_model(name)

    if choice == '2':
        version_num = int(input("basic model num to load - "))
        current_model = model_build_type(version_num)

    elif choice == '3':
        if current_model is None:
            print("Error: Please load a model first (Option 1). (Note: Build a new compiled model if starting fresh).")
            continue
        train_on_moabb(current_model)

    elif choice == '4':
        if current_model is None:
            print("Error: Please load a model first (Option 1).")
            continue
        test_on_unseen_subjects(current_model)

    elif choice == '5':
        print("saving model.")
        save_model_to_keras(current_model)


    elif choice == '6':
        print("Exiting System. Goodbye!")
        break


    else:
        print("Invalid selection. Please enter a number between 1 and 4.")