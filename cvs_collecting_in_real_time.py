import numpy as np
from cvs_data_processing import apply_filters_to_eeg, global_scaler
import pandas as pd
from Config import *

# 1. Maintain a buffer of incoming EEG data
eeg_buffer = []


def process_real_time_stream(new_reading, model, flat=False):
    global eeg_buffer
    eeg_buffer.append(new_reading)

    # Check if we have collected 256 readings to form a complete window.
    if len(eeg_buffer) >= 256:

        # Convert buffer to DataFrame and apply filters to remove noise.
        window = pd.DataFrame(eeg_buffer[-256:], columns=[f"eeg.{i}" for i in range(14)])
        clean_eeg = apply_filters_to_eeg(window)
        if flat:
            X = clean_eeg.values.reshape(1, 256, 14)
        else:
            X = clean_eeg.values.reshape(1, 4, 64, 14)

        # Transform the window using the per-channel scalers fitted during training.
        if global_scaler:
            for ch in range(14):
                if flat:
                    X[:, :, ch] = global_scaler[ch].transform(X[:, :, ch].reshape(-1, 1)).reshape(1, 256)
                else:
                    X[:, :, :, ch] = global_scaler[ch].transform(X[:, :, :, ch].reshape(-1, 1)).reshape(1, 4, 64)
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)

        return predicted_class

    return None