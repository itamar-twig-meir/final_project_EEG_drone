from enum import Enum
import numpy as np
import pandas
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import re
import hashlib
import os
import moabb

#region load data and basic shaping

label_dictionary = {
    'Blank': 0,
    'Camera': 1,
    'Up': 2,
    'Down': 3,
    'Left': 4,
    'Right': 5
}
global_used_training_data = []
global_seen_hashes = set()

def combine_data_and_label(data,label):

    if isinstance(data, np.ndarray):
        if(data.shape[0] == label.shape[0]):
            return  np.concatenate((data, label), axis=1)
        elif (data.shape[0] == label.shape[0]*256):
            return np.concatenate((data, np.repeat(label, 256)), axis=0)
        else:
            print("data and label are not the same size - numpy")
            return None

    elif isinstance(data, pd.DataFrame):
        if data.shape[0] == label.shape[0]:
            return pd.concat([data.reset_index(drop=True), label.reset_index(drop=True)], axis=1)
        elif data.shape[0] == label.shape[0] * 256:
            repeated_labels = label.loc[label.index.repeat(256)].reset_index(drop=True)
            return pd.concat([data.reset_index(drop=True), repeated_labels], axis=1)
        else:
            print("data and label are not the same size - numpy")
            return None

    else:
        print("data isnt numpy or pandas")
        return None

def if_exists_return_label_index(data):

    label_keys = set(label_dictionary.keys())
    for i in range(data.shape[1]):
        column_unique_values = set(data.iloc[:, i].unique())
        if label_keys.intersection(column_unique_values):
            return i

    print("Error: No single column contains all dictionary values.")
    return None

def return_label_column(data, single_label):

    for i in range(data.shape[1]):
        column = data.iloc[:, i]

        if column.nunique() == 1 and column.iloc[0] == single_label:
            return  data.iloc[:, i]

    print("Error: No single column contains label values.")
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

def apply_ica_to_csv(csv_eeg_data, sfreq=256):

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

def combine_data(file_list):

    all_dfs = []
    if len(file_list) == 0:
        while True:
            path = input("Write file path. Write 'stop' to stop: ")
            if path.lower() == "stop":
                break
            all_dfs.append(pd.read_csv(path))
    else:
        for file in file_list:
            all_dfs.append(pd.read_csv(file))

    return pd.concat(all_dfs, ignore_index=True)

def make_total_sample_answer_couples(eeg_data, labels, num_of_readings_in_thought=256, num_of_time_window=4):

    num_of_single_readings = eeg_data.shape[0]

    if num_of_single_readings % num_of_readings_in_thought != 0:
        print(f"Error: {num_of_single_readings} readings is not divisible by window size {num_of_readings_in_thought}")
        return None, None
    if num_of_readings_in_thought % num_of_time_window != 0:
        print(f"num of full readings not devisable by num of time windows ({num_of_time_window}) \
              currently {num_of_single_readings / num_of_readings_in_thought} full readings ")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(eeg_data.values.astype(np.float32))

    num_of_samples = int(num_of_single_readings / num_of_readings_in_thought)
    readings_per_window = int(num_of_readings_in_thought / num_of_time_window)
    
    if(X_scaled.shape[0] == labels.shape[0]):
        labels = labels[::256]
        
    X_total = X_scaled.reshape(num_of_samples, num_of_time_window, readings_per_window, 14)
    Y_total = np.array(labels)

    if X_total.shape[0] != Y_total.shape[0]:
        print(f"Warning: X has {X_total.shape[0]} samples but Y has {Y_total.shape[0]} samples.")

    return X_total, Y_total

def turn_Y_array_into_one_hot(Y_array):
    Y_array= np.array([label_dictionary[label] for label in Y_array])
    return tf.keras.utils.to_categorical(Y_array, num_classes=len(label_dictionary))

def data_set_not_used_for_training(X_new, Y_new, file_name):

    global global_used_training_data, global_seen_hashes

    fingerprint = X_new[0, 0, :10, :].tobytes()
    data_hash = hashlib.md5(fingerprint).hexdigest()
    if data_hash in  global_seen_hashes:
        print("Duplicate Detected: This data has already been added to the session.")
        return False
    global_used_training_data.append((X_new, Y_new, file_name))
    global_seen_hashes.add(data_hash)
    print(f"Success: Added uniqe dataset.")

    return True

def print_files_used_for_training():
    print("files used: \n")
    for file in global_used_training_data:
        print(file + "\n")

def shuffle_synchronized(X, Y):

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], Y[indices]

def split_data_and_class_into_train_test_validation(X_total, Y_total, train_ratio=0.8, test_ratio=0.1,validation_ratio=0.1):
    if train_ratio + test_ratio + validation_ratio != 1:
        print("Error: Ratios do not add up to 1.")
        return None

    X_total, Y_total = shuffle_synchronized(X_total, Y_total)

    total = X_total.shape[0]

    test_end = int(total * test_ratio)
    train_end = test_end + int(total * train_ratio)

    X_test, Y_test = X_total[:test_end], Y_total[:test_end]
    X_train, Y_train = X_total[test_end:train_end], Y_total[test_end:train_end]
    X_val, Y_val = X_total[train_end:], Y_total[train_end:]

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

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

def manage_model(X_new, Y_new, model, file_path, use_case="train"):

    if use_case.lower() == "train":
        print("\n--- Starting Training Workflow ---")

        if data_set_not_used_for_training(X_new, Y_new, file_path):
            X_full = np.concatenate([pair[0] for pair in global_used_training_data], axis=0)
            Y_full = np.concatenate([pair[1] for pair in global_used_training_data], axis=0)

            X_train, Y_train, X_test, Y_test, X_val, Y_val = split_data_and_class_into_train_test_validation(X_full, Y_full)

            print(f"Training on {len(X_train)} samples...")
            model.fit(X_train, Y_train,
                      validation_data=(X_val, Y_val),
                      epochs=5,
                      batch_size=32)
            print("\n--- Final Evaluation on Unseen Test Data ---")
            loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            save_choice = input("Training complete. Save model? (yes/no): ").lower()
            if save_choice == "yes":
                model_name = input("Enter model name (e.g., 'eeg_model_v1'): ")
                model.save(f"{model_name}.keras")
                print(f"Model saved as {model_name}.keras")
        else:
            print("Action cancelled: Data is a duplicate.")

    elif use_case.lower() == "test":
        print("\n--- Starting Testing Workflow ---")
        loss, accuracy = model.evaluate(X_new, Y_new, verbose=0)
        print(f"Test Results for this file:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Loss: {loss:.4f}")

    else:
        print("Invalid use case. Please choose 'train' or 'test'.")

def process_file_pipeline(file_path):
    try:

        raw_data = pd.read_csv(file_path)
        eeg_only = return_eeg_columns(raw_data)
        if eeg_only is None: return None, None

        labels = if_exists_return_label_index(raw_data)

        if labels is None:
            print(f"Error: Could not identify a valid label column in {file_path}")
            return None, None

        clean_eeg = apply_ica_to_csv(eeg_only)
        X, Y = make_total_sample_answer_couples(clean_eeg, raw_data.iloc[:, labels])
        Y_one_hot = turn_Y_array_into_one_hot(Y)

        return X, Y_one_hot

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None, None

def add_file_to_training_pool(file_path):
    global global_used_training_data

    print(f"\n--- Processing: {file_path} ---")

    X, Y_one_hot = process_file_pipeline(file_path)

    if X is not None and Y_one_hot is not None:
        if data_set_not_used_for_training(X, Y_one_hot, file_path):
            print(f"Successfully added {file_path} to the pool.")
            print(f"Current pool size: {len(global_used_training_data)} files.")
        else:
          print("data was already added to the pool in the past.")
    else:
        print(f"Failed to add {file_path} due to processing errors.")

current_model = None

while True:
    print("\n" + "=" * 15)
    print("    EEG UI")
    print("=" * 15)
    print(f"Model Loaded: {current_model is not None}")
    print(f"Files in Training Pool: {len(global_used_training_data)}")
    print("-" * 30)
    print("1. Load Model (.keras)")
    print("2. Train Model (Load file -> Add to Pool -> Train)")
    print("3. Test Model (Load file -> Evaluate without saving)")
    print("4. View Training Pool Files")
    print("5. add data to training pool")
    print("6. Exit")

    choice = input("\nSelect an option (1-5): ")

    if choice == '1':
        name = input("Enter model file name (without .keras): ")
        if name == "yes": name = "../model_versions_single/drone_eeg_model_9"
        current_model = load_eeg_model(name)

    elif choice == '2':
        if current_model is None:
            print("Error: Please load a model first (Option 1).")
            continue

        path = input("Enter CSV file path for training: ")
        X, Y = process_file_pipeline(path)

        if X is not None:
            # Note: manage_model handles the global storage and retraining logic
            manage_model(X, Y, current_model, path, use_case="train")

    elif choice == '3':
        if current_model is None:
            print("Error: Please load a model first (Option 1).")
            continue

        path = input("Enter CSV file path for testing: ")
        X, Y = process_file_pipeline(path)

        if X is not None:
            manage_model(X, Y, current_model, path, use_case="test")

    elif choice == '4':
        if not global_used_training_data:
            print("Training pool is empty.")
        else:
            print("\nFiles currently in training session:")
            for item in global_used_training_data:
                # We stored (X, Y, file_name) in data_set_not_used_for_training
                print(f" - {item[2]}")

    elif choice == '5':
        path = input("Enter CSV file path for training: ")
        print("attempting to add file to used data pool")
        if path == "yes": path = "../drone_project_data/merged_result_256.csv"
        add_file_to_training_pool(path)

    elif choice == '6':
        print("Exiting AeroSentry System. Goodbye!")
        break

    else:
        print("Invalid selection. Please enter a number between 1 and 5.")
