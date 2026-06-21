import time
import numpy as np
import pandas as pd
from collections import deque
from cvs_data_processing import apply_filters_to_eeg, global_scaler
from Config import *


def process_real_time_stream(model, flat=False):
    # Initialize a fixed-size buffer inside the function
    eeg_buffer = deque(maxlen=256)

    print("Starting Real-Time Inference... (Press Ctrl+C to stop)")

    try:
        while True:

            new_reading = np.random.rand(14)

            eeg_buffer.append(new_reading)

            if len(eeg_buffer) == 256:
                window = pd.DataFrame(list(eeg_buffer), columns=[f"eeg.{i}" for i in range(14)])
                clean_eeg = apply_filters_to_eeg(window)
                if flat:
                    X = clean_eeg.values.reshape(1, 256, 14)
                else:
                    X = clean_eeg.values.reshape(1, 4, 64, 14)

                if global_scaler:
                    for ch in range(14):
                        if flat:
                            X[:, :, ch] = global_scaler[ch].transform(X[:, :, ch].reshape(-1, 1)).reshape(1, 256)
                        else:
                            X[:, :, :, ch] = global_scaler[ch].transform(X[:, :, :, ch].reshape(-1, 1)).reshape(1, 4,64)
                prediction = model.predict(X, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)

                print(f"Prediction: {predicted_class}")


    except KeyboardInterrupt:
        print("\nStopped Real-Time Inference.")