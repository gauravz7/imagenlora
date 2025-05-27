# train_model.py
import os
import json

import config
import utils

def get_training_jsonl_uri() -> str:
    """
    Retrieves the GCS URI for the training JSONL file.
    Tries to read from 'last_jsonl_uri.txt' first, then prompts user.
    """
    try:
        with open("last_jsonl_uri.txt", "r") as f:
            uri = f.read().strip()
            if uri.startswith("gs://") and uri.endswith(".jsonl"):
                print(f"Found training JSONL URI from last session: {uri}")
                return uri
            else:
                print("Found invalid URI in last_jsonl_uri.txt.")
    except FileNotFoundError:
        print("last_jsonl_uri.txt not found.")
    
    while True:
        uri = input(f"Enter the GCS URI of your training JSONL file (e.g., {config.get_gcs_jsonl_file_uri()}): ").strip()
        if uri.startswith("gs://") and uri.endswith(".jsonl"):
            return uri
        else:
            print("Invalid GCS URI format. It should start with 'gs://' and end with '.jsonl'.")

def main():
    print("--- Step 4: Configure and Create Imagen Fine-Tuning Job ---")

    # Get the training data URI
    train_jsonl_gcs_uri = get_training_jsonl_uri()
    if not train_jsonl_gcs_uri:
        print("No training JSONL URI provided. Exiting.")
        return

    # --- Get Fine-Tuning Parameters from User ---
    print("\nPlease provide the following details for the fine-tuning job:")
    
    tuned_model_name = input(f"Enter a display name for your tuned model (default: '{config.DEFAULT_TUNED_MODEL_DISPLAY_NAME}'): ").strip()
    if not tuned_model_name:
        tuned_model_name = config.DEFAULT_TUNED_MODEL_DISPLAY_NAME

    adapter_size = input(f"Enter Adapter Size (e.g., ADAPTER_SIZE_SIXTEEN, default: '{config.DEFAULT_ADAPTER_SIZE}'): ").strip()
    if not adapter_size: # Add more validation if needed based on allowed values
        adapter_size = config.DEFAULT_ADAPTER_SIZE
    # TODO: Could add validation for allowed adapter sizes

    epochs_str = input(f"Enter number of training epochs (default: {config.DEFAULT_EPOCHS}): ").strip()
    try:
        epochs = int(epochs_str) if epochs_str else config.DEFAULT_EPOCHS
        if epochs <= 0:
            print("Epochs must be a positive integer. Using default.")
            epochs = config.DEFAULT_EPOCHS
    except ValueError:
        print("Invalid input for epochs. Using default.")
        epochs = config.DEFAULT_EPOCHS

    lr_multiplier_str = input(f"Enter learning rate multiplier (default: {config.DEFAULT_LEARNING_RATE_MULTIPLIER}): ").strip()
    try:
        lr_multiplier = float(lr_multiplier_str) if lr_multiplier_str else config.DEFAULT_LEARNING_RATE_MULTIPLIER
        if lr_multiplier <= 0:
            print("Learning rate multiplier must be positive. Using default.")
            lr_multiplier = config.DEFAULT_LEARNING_RATE_MULTIPLIER
    except ValueError:
        print("Invalid input for learning rate multiplier. Using default.")
        lr_multiplier = config.DEFAULT_LEARNING_RATE_MULTIPLIER

    print("\n--- Summary of Fine-Tuning Configuration ---")
    print(f"Project ID:         {config.PROJECT_ID}")
    print(f"Location:           {config.LOCATION}")
    print(f"Base Model:         {config.IMAGEN_BASE_MODEL}")
    print(f"Training Data JSONL:{train_jsonl_gcs_uri}")
    print(f"Tuned Model Name:   {tuned_model_name}")
    print(f"Adapter Size:       {adapter_size}")
    print(f"Epochs:             {epochs}")
    print(f"LR Multiplier:      {lr_multiplier}")
    print("--------------------------------------------")

    confirm = input("Proceed with submitting the tuning job? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Tuning job submission cancelled by user.")
        return

    # Create the tuning job
    tuning_job_path = utils.create_imagen_tuning_job(
        project_id=config.PROJECT_ID,
        location=config.LOCATION,
        base_model_uri=config.IMAGEN_BASE_MODEL, # Pass the model ID
        tuned_model_display_name=tuned_model_name,
        training_dataset_uri=train_jsonl_gcs_uri,
        epochs=epochs,
        adapter_size=adapter_size,
        learning_rate_multiplier=lr_multiplier
    )

    if tuning_job_path:
        print(f"\nSuccessfully submitted tuning job. Job Path: {tuning_job_path}")
        # Save the job path for the monitoring script
        with open("last_tuning_job_path.txt", "w") as f:
            f.write(tuning_job_path)
        print(f"The tuning job path has been saved to last_tuning_job_path.txt for monitoring.")
    else:
        print("\nFailed to submit tuning job. Check logs for errors.")

if __name__ == "__main__":
    try:
        utils.get_storage_client()
        utils.get_gemini_client()
        print("Google Cloud clients initialized successfully for training script.")
    except Exception as e:
        print(f"Error initializing Google Cloud clients: {e}")
        print("Please ensure you are authenticated with Google Cloud and APIs are enabled.")
        exit(1)
        
    main()
