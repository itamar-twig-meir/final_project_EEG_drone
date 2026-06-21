import numpy as np
from Config import *
from cvs_models import *
from cvs_data_processing import *
from cvs_collecting_in_real_time import *

current_model = None
# stores the model once one is selected/loaded
while True:
    print("\n" + "=" * 15)
    print("    EEG UI")
    print("=" * 15)
    print(f"Model Loaded:         {current_model is not None}")
    print(f"Files in Pool:        {len(global_used_training_data)}")
    print("-" * 30)
    #presenting the current status
    print("1. Load Model file (.keras)")
    print("2. Load basic model")
    print("3. Train Model (Train on Current Pool)")
    print("4. View Training Pool Files")
    print("5. Add data to training pool")
    print("6. start real time use")
    print("7. Configure Preprocessing")
    print("8. Test single random sample")
    print("9. Exit")


    choice = input("\nSelect an option (1-8): ")

    # load model by file
    if choice == '1':
        name = input("Enter model file name (without .keras): ")
        if name == "yes": name = "../model_versions_single/drone_eeg_model_9"
        current_model = load_eeg_model(name)

    # choose model from set options
    elif choice == '2':
        version_num = int(input("Enter basic model num to build (1, 2, 3, 4): "))
        current_model = build_basic_model(version_num)
        if current_model is not None:
            print(f"Successfully built model version {version_num}")

    # train model on data pool
    elif choice == '3':
        if current_model is None:
            print("Error: Please load a model first (Option 1 or 2).")
            continue
        train_model_on_pool(current_model)

    # see what files were already added to training pool
    elif choice == '4':
        if not global_used_training_data:
            print("Training pool is empty.")
        else:
            print("\nFiles currently in training session:")
            for item in global_used_training_data:
                print(f" - {item[2]}")

    # add data to pool
    elif choice == '5':
        path = input("Enter CSV file path for training: ")
        if path == "yes":
            path = "../final_project_EEG_drone/drone_project_data/merged_result_256.csv"
        flat_choice = input("Use flat input for model 4? (yes/no): ").strip().lower() == "yes"
        add_file_to_training_pool(path, flat=flat_choice)

    # start real time use of the model after its weights are calibrated/set
    elif choice == '6':
        if current_model is None:
            print("Error: No model loaded/trained.")
            continue
        print("Starting Real-Time Inference... (Press Ctrl+C to stop)")
        try:
            while True:
                raw_data = np.random.rand(14)
                prediction = process_real_time_stream(raw_data, current_model, flat=False)
                if prediction is not None:
                    print(f"Prediction: {prediction}")
        except KeyboardInterrupt:
            print("\nStopped Real-Time Inference.")
        break

    # Configure the normalization and preprocessing
    elif choice == '7':
        print("\n--- Configuration ---")
        mode = input("Enter Normalization mode (general/electrode): ").strip().lower()
        if mode in ['general', 'electrode']: NORMALIZATION_MODE = mode
        ica_choice = input("Use ICA? (yes/no): ").strip().lower()
        USE_ICA = (ica_choice == 'yes')
        print(f"Settings updated: Normalization={NORMALIZATION_MODE}, ICA={USE_ICA}")

    # test single sample
    elif choice == '8':
        if current_model is None:
            print("Error: No model loaded/trained.")
            continue
        test_single_sample(current_model)

    #exit loop
    elif choice == '9':
        print("Exiting EEG System. Goodbye!")
        break

    else:
        print("Invalid selection. Please enter a valid menu number.")