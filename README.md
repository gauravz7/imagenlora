# Imagen LoRA Fine-Tuning and Inference Toolkit on Vertex AI

This project provides a Streamlit-based web application to orchestrate the fine-tuning of Google's Imagen models using Low-Rank Adaptation (LoRA) on Vertex AI, and to perform inference using the fine-tuned models. It also includes a separate tab for general image generation using specified Imagen models on Vertex AI.

## Overview

The application is structured into two main tools accessible via a sidebar:

1.  **Fine-Tuning Pipeline:** Guides users through:
    *   **Data Preparation:** Uploading custom images, image compression, selection of context/target images, AI-assisted generation of descriptive suffixes for prompts using the Gemini API, and creation of the JSONL training data file with flexible prompt formatting.
    *   **Model Fine-Tuning:** Submitting an Imagen LoRA fine-tuning job to Vertex AI.
    *   **Job Monitoring:** Tracking the status of submitted fine-tuning jobs.

2.  **Imagen Inference:** Allows users to:
    *   Use a deployed fine-tuned Imagen model endpoint on Vertex AI.
    *   Provide a context image and one or more textual prompts.
    *   Generate new images based on the inputs.
    *   Save generated images and a manifest to a timestamped folder in Google Cloud Storage.

## Key Features

*   **User-Friendly Interface:** Streamlit app for an interactive experience.
*   **Custom Image Fine-Tuning:**
    *   Upload local images for fine-tuning.
    *   Automatic image compression (target <5MB) before GCS upload, overwriting local files if successful.
    *   Configurable GCS bucket for fine-tuning assets.
    *   Flexible data selection for context and target images.
    *   AI-powered description suffix generation using Gemini for rich prompts.
    *   Customizable prompt structure: `"{SubjectName} [{ContextID}]{OptionalBodyType}. {Pronoun} is {DescriptionSuffixFromGemini}"`.
    *   User inputs for Subject Name, Context ID, optional Body Type, and Pronoun.
    *   Automated creation and upload of JSONL training data to GCS.
*   **Vertex AI Integration:**
    *   Submission of Imagen LoRA fine-tuning jobs using the `v1` API.
    *   Monitoring of fine-tuning job status using the `v1` API.
*   **Multi-Prompt Inference:**
    *   Dedicated inference page for generating images with a fine-tuned model.
    *   Support for multiple input prompts for batch generation.
    *   Outputs (generated images and a JSON manifest) automatically saved to a timestamped folder in GCS.
*   **Configuration:**
    *   Centralized `config.py` for GCP project details, default GCS paths, model IDs.
    *   `config.py.example` provided as a template.
    *   `.gitignore` configured to exclude sensitive files and local data.

## Prerequisites

1.  **Google Cloud Platform (GCP) Account:** Billing enabled.
2.  **GCP Project:**
    *   Vertex AI API enabled.
    *   Cloud Storage API enabled.
3.  **`gcloud` CLI:** Google Cloud SDK installed and authenticated.
4.  **Python:** Version 3.9 or higher.
5.  **Virtual Environment:** Strongly recommended (e.g., `venv`, `conda`).
6.  **Permissions:** The GCP user or service account used must have:
    *   `Vertex AI User` role (or more specific roles for training jobs and endpoints).
    *   `Storage Object Admin` (or `Storage Admin`) role for the GCS bucket used.
    *   Permissions to enable necessary APIs if not already enabled.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/gauravz7/imagenlora.git
    cd imagenlora
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `streamlit`, `google-cloud-aiplatform`, `google-generativeai`, `Pillow`, `google-cloud-storage`, `google-auth`, `requests`)*

4.  **Configure `config.py`:**
    *   Copy `config.py.example` to `config.py`:
        ```bash
        cp config.py.example config.py
        ```
    *   Edit `config.py` and replace placeholder values with your specific GCP `PROJECT_ID`, `LOCATION` (region), `GCS_BUCKET_NAME`, and update `LOCAL_IMAGE_DIR` to the path of your local image dataset. Other defaults can be reviewed and adjusted as needed.

5.  **Authenticate with Google Cloud:**
    *   If running the application locally, authenticate the `gcloud` CLI to provide Application Default Credentials (ADC):
        ```bash
        gcloud auth application-default login
        ```
    *   This allows the Python client libraries to authenticate to GCP services.

## Running the Application

Once setup is complete, run the Streamlit application from the project's root directory:
```bash
streamlit run app.py
```
The application will open in your default web browser. Use the sidebar to navigate between the "Fine-Tuning Pipeline" and "Imagen Inference" tools.

## File Structure

*   `app.py`: Main Streamlit application. Contains UI logic for both fine-tuning and inference pages.
*   `utils.py`: Utility functions for:
    *   Google Cloud Storage operations (upload, list, download).
    *   Image processing (compression, resizing for Gemini).
    *   Gemini API calls for generating description suffixes.
    *   Vertex AI fine-tuning job creation and monitoring.
*   `config.py`: **User-configured** file for project-specific settings (PROJECT_ID, LOCATION, GCS_BUCKET_NAME, etc.). Ignored by Git.
*   `config.py.example`: Template for `config.py`.
*   `requirements.txt`: Python dependencies.
*   `.gitignore`: Specifies intentionally untracked files (e.g., `config.py`, virtual environments, OS files).
*   `README.md`: This file.
*   `data_preparation.py`, `train_model.py`, `monitor_job.py`: Python scripts likely containing core logic or original script versions for the fine-tuning pipeline steps, now largely integrated into `app.py` and `utils.py`. (These might be refactored or removed if their functionality is fully covered by the Streamlit app).

## Usage Notes

*   **GCS Bucket for Fine-Tuning:** The "Fine-Tuning Pipeline" page allows you to specify the GCS bucket to use in the sidebar. This overrides the default in `config.py` for that session.
*   **GCS Bucket for Inference Outputs:** The "Imagen Inference" page saves generated images and a manifest to a subfolder within the GCS bucket defined by `st.session_state.get('user_gcs_bucket_name_ft', config.GCS_BUCKET_NAME)` (i.e., it uses the bucket configured for the fine-tuning pipeline, or the default from `config.py` if the fine-tuning page's sidebar input hasn't been used yet). The output path is `gs://<your-bucket>/imagen_inference_outputs/output-YYYYMMDD-HHMMSS/`.
*   **Image Compression:** Local images selected for upload in the fine-tuning pipeline are automatically compressed if they exceed 5MB. The original local files are overwritten with compressed versions if successful.
*   **API Quotas & Costs:** Be mindful of API usage quotas and associated costs on Google Cloud Platform for Vertex AI (training, endpoints, prediction) and Cloud Storage.
*   **Error Handling:** The application includes basic error handling and logging to the Streamlit interface. Check the terminal console where Streamlit is running for more detailed logs from the Python scripts, especially for `gsutil` or `curl` command outputs during inference.
