import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Import your existing functions from your main file (assumed to be model.py)
from cvs_model_ui import apply_ica_to_csv, make_total_sample_answer_couples, turn_Y_array_into_one_hot


def load_trained_model(weights_path):
    print(f"Loading model from {weights_path}...")
    return tf.keras.models.load_model(weights_path)


def process_extra_data(file_path,start_index, end_index, usage='general', amount=None):

    df = pd.read_csv(file_path)
    df = df.iloc[:,start_index:end_index]

    if amount is not None:
        if isinstance(amount, float) and 0 < amount <= 1.0:
            df = df.sample(frac=amount, random_state=42)
        elif isinstance(amount, int):
            df = df.head(amount)

    clean_df = apply_ica_to_csv(df)
    X, Y = make_total_sample_answer_couples(clean_df, num_of_time_window=8)
    Y = turn_Y_array_into_one_hot(Y)

    if usage == 'train':
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        return {"train": (x_train, y_train), "val": (x_val, y_val)}
    elif usage == 'test':
        return {"test": (X, Y)}
    else:
        x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
        return {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (x_test, y_test)
        }

model_path = input("model path ")
model = load_trained_model(model_path)

while True:

    user_input = input("load data - format(file path, usage(optional), amount(optional) ")
    parts = [p.strip() for p in user_input.split(',')]
    file_path = parts[0]
    usage = parts[1] if len(parts) > 1 else 'general'
    amount_raw = parts[2] if len(parts) > 2 else None
    amount = None
    if amount_raw:
        try:

            amount = float(amount_raw) if '.' in amount_raw else int(amount_raw)
        except ValueError:
            print(f"Warning: Could not parse amount '{amount_raw}'. Using all data.")
    df = pd.read_csv(file_path)
    up = "up" in df
    down = "down" in df
    blank = "blank" in df
    camera = "camera" in df
    right = "right" in df
    left = "left" in df

    print(f"{up}, contains up \n {down}, contains down \n {blank}, contains blank \n {camera}, contains camera \n {right}, contains right \n {left}, contains left \n")
    print(df.iloc[0:3,].to_string(header=False))
    start_index = int(input("start index - "))
    end_index = int(input("end index - "))
    if start_index > 0:
        start_index = start_index - 1
    data_bundles = process_extra_data(file_path,start_index,end_index, usage=usage, amount=amount)

    if "train" in data_bundles:
        X_train, Y_train = data_bundles["train"]
        X_val, Y_val = data_bundles["val"]

        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5)

    if "test" in data_bundles:
        X_test, Y_test = data_bundles["test"]
        model.evaluate(X_test, Y_test)

