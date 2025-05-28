# Imagen LoRA: Easy Fine-Tuning and Image Generation on Vertex AI

Welcome! This project helps you fine-tune Google's Imagen models (powerful AI for creating images from text) using a technique called LoRA (Low-Rank Adaptation). Once fine-tuned, you can use your custom model to generate new images. It all happens on Google Cloud's Vertex AI platform, and this toolkit provides a user-friendly Streamlit web app to manage the process.

You can also use this app for general image generation with standard Imagen models.

## What Can You Do With This?

This application offers two main tools:

1.  **Fine-Tuning Your Own Image Model:**
    *   **Prepare Your Images:** Upload your own pictures. The app helps you compress them, choose which ones to use for training, and even uses AI (Gemini) to help write good descriptions for them. These descriptions become "prompts" that teach your model.
    *   **Train Your Model:** Send your prepared images and prompts to Vertex AI to start the LoRA fine-tuning process for an Imagen model.
    *   **Check on Training:** Keep an eye on how your model training is going.

2.  **Create Images (Imagen Inference):**
    *   Use a fine-tuned Imagen model that you've trained.
    *   Give it a starting image (optional) and some text prompts (what you want to see).
    *   Generate new images based on your instructions!
    *   Save your creations and a list of what you generated to Google Cloud Storage.

## Key Features for Beginners

*   **Simple Web App:** Uses Streamlit for an easy-to-navigate interface. No complex commands needed for most tasks.
*   **Train with Your Images:**
    *   Easily upload images from your computer.
    *   Images are automatically made smaller if they're too big for training.
    *   You decide where your training files are stored in Google Cloud Storage.
    *   AI helps you write good text prompts for your images.
    *   You can customize how your prompts are structured (e.g., "A photo of {MyPet} [{ItemID}]. {MyPet} is {AI-generated description}").
*   **Works with Google's Vertex AI:**
    *   Sends your training jobs to Vertex AI.
    *   Lets you see the status of your training jobs.
*   **Generate Multiple Images:**
    *   Create images using your trained model.
    *   You can give it several text prompts at once to make many images.
    *   Your new images are saved neatly in Google Cloud Storage.
*   **Easy Configuration:**
    *   A simple `config.py` file to tell the app about your Google Cloud project.
    *   An example file (`config.py.example`) is provided to get you started.

## Before You Start: Prerequisites

You'll need a few things set up first:

1.  **Google Cloud Platform (GCP) Account:** If you don't have one, you'll need to sign up. Make sure billing is enabled (some services have costs).
2.  **A GCP Project:**
    *   Inside your GCP account, create a project.
    *   Enable the "Vertex AI API" and "Cloud Storage API" for this project. (You can usually find these by searching in the GCP console.)
3.  **`gcloud` Command-Line Tool:** This is Google Cloud's tool for your computer. Install it from the [Google Cloud SDK page](https://cloud.google.com/sdk/docs/install). After installing, log in with it.
4.  **Python:** Make sure you have Python installed (version 3.9 or newer is best). You can get it from [python.org](https://www.python.org/downloads/).
5.  **Virtual Environment (Recommended):** This keeps project dependencies tidy.
    *   Python has `venv` built-in. Or you can use tools like `conda`.
6.  **Permissions in GCP:** The Google account you use needs permission to:
    *   Use Vertex AI (e.g., "Vertex AI User" role).
    *   Read and write to Google Cloud Storage buckets (e.g., "Storage Object Admin" or "Storage Admin" role).

## Step-by-Step Setup

1.  **Get the Code (Clone the Repository):**
    ```bash
    git clone https://github.com/gauravz7/imagenlora.git
    cd imagenlora
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    Open your terminal or command prompt in the `imagenlora` folder.
    ```bash
    # For Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    # python -m venv .venv  (if python3 doesn't work)
    # .venv\Scripts\activate
    ```
    You should see `(.venv)` at the start of your command prompt line.

3.  **Install Required Python Packages:**
    While your virtual environment is active:
    ```bash
    pip install -r requirements.txt
    ```
    This reads the `requirements.txt` file and installs all the necessary Python libraries.

4.  **Set Up Your Configuration (`config.py`):**
    *   Find the file `config.py.example`. Make a copy of it and name the copy `config.py`.
        ```bash
        cp config.py.example config.py
        ```
    *   Open `config.py` in a text editor.
    *   You'll need to fill in:
        *   `PROJECT_ID`: Your Google Cloud Project ID.
        *   `LOCATION`: The GCP region you want to use (e.g., `us-central1`).
        *   `GCS_BUCKET_NAME`: The name of a Google Cloud Storage bucket you want to use. If it doesn't exist, you might need to create it first in the GCP console.
        *   `LOCAL_IMAGE_DIR`: The path on your computer where your training images are stored (e.g., `/Users/yourname/Pictures/MyTrainingSet` or `C:\Users\yourname\Pictures\MyTrainingSet`).
    *   Review other settings and change them if needed.

5.  **Log In to Google Cloud (for the app to use):**
    If you're running this on your own computer, make sure your `gcloud` tool is logged in so the Python code can access Google Cloud:
    ```bash
    gcloud auth application-default login
    ```
    Follow the instructions that appear in your browser.

## How to Run the App

Once everything is set up:
1.  Make sure your virtual environment is active (you see `(.venv)` in your prompt).
2.  In your terminal, from the `imagenlora` project folder, run:
    ```bash
    streamlit run app.py
    ```
3.  This should open the application in your web browser automatically.
4.  Use the sidebar in the app to choose between "Fine-Tuning Pipeline" or "Imagen Inference."

## Understanding the Files

*   `app.py`: The main brain of the web application.
*   `utils.py`: Contains helper functions for common tasks like talking to Google Cloud Storage, processing images, and interacting with Vertex AI.
*   `config.py`: **Your personal configuration file.** This tells the app your specific Google Cloud details. (This file is ignored by Git, so your secrets stay safe).
*   `config.py.example`: A template to help you create your `config.py`.
*   `requirements.txt`: A list of all Python packages the project needs.
*   `.gitignore`: Tells Git (version control) which files to ignore (like your `config.py`).
*   `README.md`: This file!
*   `data_preparation.py`, `train_model.py`, `monitor_job.py`: These might be older script versions or contain some core logic. The main functionality is now mostly in `app.py` and `utils.py`.

## Important Notes

*   **Google Cloud Storage Buckets:**
    *   For Fine-Tuning: You can tell the app which GCS bucket to use in the sidebar.
    *   For Inference: Images you create are saved to a subfolder (like `imagen_inference_outputs/output-YYYYMMDD-HHMMSS/`) in the GCS bucket you've set up.
*   **Image Compression:** If your uploaded images are too big (over 5MB), the app tries to make them smaller. This will overwrite the original files on your computer if successful.
*   **Costs:** Using Google Cloud services (Vertex AI, Storage) can cost money. Keep an eye on your usage and billing in the GCP console.
*   **Errors:** The app tries to show errors in the interface. For more details, look at the console window where you started Streamlit.

---
