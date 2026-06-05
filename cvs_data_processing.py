from sklearn.preprocessing import StandardScaler
import re
import hashlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mne

import Config
from Config import *
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt

label_dictionary = {
    'Blank': 0,
    'Camera': 1,
    'Up': 2,
    'Down': 3,
    'Left': 4,
    'Right': 5
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

global_used_training_data = []
global_seen_hashes = set()
global_scaler = None


def if_exists_return_label_index(data):

    label_keys = set(label_dictionary.keys())
    for i in range(data.shape[1]):
        column_unique_values = set(data.iloc[:, i].unique())
        if label_keys.intersection(column_unique_values):
            return i

    print("Error: No single column contains all dictionary values.")
    return None


def return_eeg_columns(data):
    #ill try to find the eeg columns in 3 ways - by searching for known names, by
    #searching for general names, and by searching for value patterns

    standard_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                      'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    found_cols = []

    for name in standard_names:

        pattern = re.compile(rf'(eeg[._])?\b{name}\b', re.IGNORECASE)
        matches = [col for col in data.columns if pattern.search(str(col))]
        matches.sort(key=lambda x: bool(re.search(r'eeg[._]', str(x), re.I)), reverse=True)

        if matches:
            found_cols.append(matches[0])

    if len(found_cols) == 14:
        print("Success: Identified channels via header names.")
        eeg_df = data[found_cols].copy()
        eeg_df.columns = [f"eeg.{name.lower()}" for name in standard_names]
        return eeg_df

    variances = data.var()
    potential_cols = []
    for col in data.columns:
        if data[col].nunique() < 20:
            continue
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        potential_cols.append(col)

    if len(potential_cols) >= 14:
        means = data[potential_cols].mean()
        median_mean = means.median()
        best_14 = (means - median_mean).abs().sort_values().head(14).index.tolist()

        best_14.sort(key=lambda x: data.columns.get_loc(x))
        print("Success: Identified channels via values")
        eeg_df = data[best_14].copy()
        eeg_df.columns = [f"eeg.{name.lower()}" for name in standard_names]
        return eeg_df


def apply_ica_to_csv(csv_eeg_data, sfreq=128):
    ch_names = [col.replace('eeg.', '').strip().upper() for col in csv_eeg_data.columns]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(csv_eeg_data.values.T *  1e-6, info)

    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)

    ica = mne.preprocessing.ICA(n_components=13, random_state=42, method='fastica')
    ica.fit(raw)
    ica.plot_components(inst= raw, title="ICA Components - Spatial Heatmaps")
    plt.show( block=True)
    raw_clean = ica.apply(raw.copy())
    clean_np = raw_clean.get_data().T * 1e6

    return pd.DataFrame(clean_np, columns=csv_eeg_data.columns)


def make_total_sample_answer_couples(eeg_data, labels, num_of_readings_in_thought=256,
                                      num_of_time_window=4, flat=False, overlap=0.75):

    num_of_single_readings = eeg_data.shape[0]
    step = int(num_of_readings_in_thought * (1 - overlap))
    readings_per_window = int(num_of_readings_in_thought / num_of_time_window)
    X_raw = eeg_data.values.astype(np.float32)

    valid_X = []
    valid_Y = []
    print(f"Starting X shape: {X_raw.shape}")
    skipped = 0

    num_of_samples = (num_of_single_readings - num_of_readings_in_thought) // step + 1

    for i in range(num_of_samples):
        start = i * step
        end = start + num_of_readings_in_thought
        if end > num_of_single_readings:
            break

        window_labels = labels[start:end]
        if window_labels.nunique() > 1:
            skipped += 1
            continue

        window_X = X_raw[start:end]
        if flat:
            valid_X.append(window_X.reshape(num_of_readings_in_thought, 14))
        else:
            valid_X.append(window_X.reshape(num_of_time_window, readings_per_window, 14))
        valid_Y.append(window_labels.iloc[0])

    print(f"Skipped {skipped} transition windows out of {num_of_samples} total.")
    print(f"Total valid windows: {len(valid_X)}")
    print(f"Total shape of X: {np.array(valid_X).shape}")
    print(f"Total shape of Y: {np.array(valid_Y).shape}")
    return np.array(valid_X), np.array(valid_Y)


def split_data_and_class_into_train_test_validation(X_total, Y_total, train_ratio=0.8, test_ratio=0.1,
                                                    validation_ratio=0.1):
    global global_scaler, NORMALIZATION_MODE

    # Convert one-hot to integers for stratified splitting
    y_integers = np.argmax(Y_total, axis=1)
    unique_classes = np.unique(y_integers)

    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    X_test_list, Y_test_list = [], []

    # Stratified split per class
    for cls in unique_classes:
        cls_indices = np.where(y_integers == cls)[0]
        cls_indices.sort()

        total_cls = len(cls_indices)
        test_end = int(total_cls * test_ratio)
        train_end = test_end + int(total_cls * train_ratio)

        test_idx = cls_indices[:test_end]
        train_idx = cls_indices[test_end:train_end]
        val_idx = cls_indices[train_end:]

        X_train_list.append(X_total[train_idx])
        Y_train_list.append(Y_total[train_idx])
        X_val_list.append(X_total[val_idx])
        Y_val_list.append(Y_total[val_idx])
        X_test_list.append(X_total[test_idx])
        Y_test_list.append(Y_total[test_idx])

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    Y_val = np.concatenate(Y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    # Shuffle training data
    X_train, Y_train = shuffle_synchronized(X_train, Y_train)

    is_4d = X_train.ndim == 4

    if Config.normalization_mode == 'general':
        # Apply normalization to the entire dataset as one block
        scaler = StandardScaler()
        # Flatten all dimensions except the last one (if we want to keep features)
        # or flatten everything for absolute general normalization
        scaler.fit(X_train.reshape(-1, 1))

        X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        global_scaler = scaler
        print("Fitted general (global) scaler.")

    else:
        # Per-electrode (channel) scaling
        scalers = []
        for ch in range(X_train.shape[-1]):
            scaler = StandardScaler()
            if is_4d:
                # Shape (samples, time_window, readings, channels)
                scaler.fit(X_train[:, :, :, ch].reshape(-1, 1))
                X_train[:, :, :, ch] = scaler.transform(X_train[:, :, :, ch].reshape(-1, 1)).reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])
                X_val[:, :, :, ch] = scaler.transform(X_val[:, :, :, ch].reshape(-1, 1)).reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2])
                X_test[:, :, :, ch] = scaler.transform(X_test[:, :, :, ch].reshape(-1, 1)).reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2])
            else:
                # Shape (samples, time_window, channels)
                scaler.fit(X_train[:, :, ch].reshape(-1, 1))
                X_train[:, :, ch] = scaler.transform(X_train[:, :, ch].reshape(-1, 1)).reshape(X_train.shape[0], X_train.shape[1])
                X_val[:, :, ch] = scaler.transform(X_val[:, :, ch].reshape(-1, 1)).reshape(X_val.shape[0],X_val.shape[1])
                X_test[:, :, ch] = scaler.transform(X_test[:, :, ch].reshape(-1, 1)).reshape(X_test.shape[0],X_test.shape[1])
            scalers.append(scaler)

        global_scaler = scalers
        print(f"Fitted {len(scalers)} per-channel (electrode) scalers.")

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def turn_Y_array_into_one_hot(Y_array):
    Y_array= np.array([label_dictionary[label] for label in Y_array])
    return tf.keras.utils.to_categorical(Y_array, num_classes=len(label_dictionary))


def data_set_not_used_for_training(X_new, Y_new, file_name):
    global global_used_training_data, global_seen_hashes
    if X_new.ndim == 4:
        fingerprint = X_new[0, 0, :10, :].tobytes()
    else:
        fingerprint = X_new[0, :10, :].tobytes()

    data_hash = hashlib.md5(fingerprint).hexdigest()
    if data_hash in global_seen_hashes:
        print("Duplicate Detected: This data has already been added to the session.")
        return False
    global_used_training_data.append((X_new, Y_new, file_name))
    global_seen_hashes.add(data_hash)
    print(f"Success: Added unique dataset.")
    return True


def print_files_used_for_training():
    print("files used: \n")
    for file in global_used_training_data:
        print(file + "\n")


def shuffle_synchronized(X, Y):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    return X[indices], Y[indices]


def load_eeg_model(model_name):
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


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    class_names = list(label_dictionary.keys())

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)


def apply_filters_to_eeg(eeg_df, sfreq=128):
    data = eeg_df.values.astype(np.float64)

    # 1. Highpass at 0.5 Hz — removes DC drift and slow baseline wander
    nyq = sfreq * 0.5
    b, a = butter(3, 0.5 / nyq, btype='high')
    data = filtfilt(b, a, data, axis=0)

    # 2. Bandpass 1–40 Hz — keeps EEG-relevant frequencies, cuts high-freq noise
    b, a = butter(3, [1.0 / nyq, 40.0 / nyq], btype='band')
    data = filtfilt(b, a, data, axis=0)

    # 3. Notch at 50 Hz — removes power-line interference (60 Hz if you're in the US)
    b, a = butter(3, [(50 - 1) / nyq, (50 + 1) / nyq], btype='bandstop')
    data = filtfilt(b, a, data, axis=0)

    return pd.DataFrame(data, columns=eeg_df.columns)


def process_file_pipeline(file_path, apply_scaling_for_inference=False, flat=False):
    try:
        raw_data = pd.read_csv(file_path)
        eeg_only = return_eeg_columns(raw_data)
        if eeg_only is None: return None, None

        if Config.USE_ICA:
            clean_eeg = apply_ica_to_csv(eeg_only)
        else:
            clean_eeg = apply_filters_to_eeg(eeg_only)

        labels = if_exists_return_label_index(raw_data)
        X, Y = make_total_sample_answer_couples(clean_eeg, raw_data.iloc[:, labels], flat=flat)
        Y_one_hot = turn_Y_array_into_one_hot(Y)

        # only apply scaler when doing inference on new data post-training
        if apply_scaling_for_inference and global_scaler is not None:
            if Config.normalization_mode == 'general':
                X = global_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
            else:
                for ch in range(X.shape[-1]):
                    if X.ndim == 4:
                        X[:, :, :, ch] = global_scaler[ch].transform(
                            X[:, :, :, ch].reshape(-1, 1)
                        ).reshape(X.shape[0], X.shape[1], X.shape[2])
                    else:
                        X[:, :, ch] = global_scaler[ch].transform(
                            X[:, :, ch].reshape(-1, 1)
                        ).reshape(X.shape[0], X.shape[1])

        return X, Y_one_hot
    except Exception as e:
        print(f"Failed: {e}")
        return None, None


def add_file_to_training_pool(file_path, flat=False):
    global global_used_training_data
    print(f"\n--- Processing: {file_path} ---")
    # always store raw unscaled data — scaler is fitted at train time
    X, Y_one_hot = process_file_pipeline(file_path, apply_scaling_for_inference=False, flat=flat)
    if X is not None and Y_one_hot is not None:
        if data_set_not_used_for_training(X, Y_one_hot, file_path):
            print(f"Successfully added {file_path} to pool.")
            print(f"Current pool size: {len(global_used_training_data)} files.")
        else:
            print("Data was already added to the pool in the past.")
    else:
        print(f"Failed to add {file_path} due to processing errors.")


def train_model_on_pool(model):
    if not global_used_training_data:
        print("Error: Training pool is empty. Please add data using Option 5 first.")
        return

    print("\n--- Starting Training Workflow ---")

    # concatenate all raw unscaled data from the pool
    X_full = np.concatenate([pair[0] for pair in global_used_training_data], axis=0)
    Y_full = np.concatenate([pair[1] for pair in global_used_training_data], axis=0)

    # split first, then split_data fits the scaler on X_train only and
    # transforms all three splits consistently — no data leakage
    X_train, Y_train, X_test, Y_test, X_val, Y_val = \
        split_data_and_class_into_train_test_validation(X_full, Y_full)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                             min_lr=0.00001, verbose=1)
    ]

    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=100, batch_size=32, callbacks=callbacks)

    print("\n--- Final Evaluation on Unseen Test Data (From Pool) ---")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(Y_test, axis=1)

    max_probs = np.max(y_pred_probs, axis=1)
    high_conf_mask = max_probs >= 0.90
    y_pred_high_conf = y_pred_classes[high_conf_mask]
    y_true_high_conf = y_true_classes[high_conf_mask]

    if len(y_pred_high_conf) > 0:
        high_conf_acc = np.mean(y_pred_high_conf == y_true_high_conf) * 100
        print(f"Confidence (>90%) Filtered Predictions: {len(y_pred_high_conf)}/{len(X_test)}")
        print(f"Accuracy on High-Confidence subset: {high_conf_acc:.2f}%")
    else:
        print("No predictions reached 90% confidence.")

    plot_confusion_matrix(Y_test, y_pred_probs, title="Confusion Matrix: Pooled Test Data")

    save_choice = input("\nTraining complete. Save model? (yes/no): ").strip().lower()
    if save_choice == "yes":
        model_name = input("Enter model name: ").strip()
        model.save(f"{model_name}.keras")
        print(f"Model saved as {model_name}.keras")



