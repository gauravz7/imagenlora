# Imagen LoRA Fine-Tuning and Inference on Vertex AI

This project provides a Streamlit-based web application to orchestrate the fine-tuning of Google's Imagen models using Low-Rank Adaptation (LoRA) on Vertex AI, and to perform inference using the fine-tuned models.

## Overview

The application guides users through the following key stages:
1.  **Data Preparation:** Uploading custom images, selecting context and target images, generating descriptive prompts (suffixes) for target images using Gemini, and creating the JSONL training data file.
2.  **Model Fine-Tuning:** Submitting a fine-tuning job to Vertex AI using the prepared JSONL data.
3.  **Job Monitoring:** Tracking the status of the submitted fine-tuning job.
4.  **Inference:** Using a deployed fine-tuned Imagen model endpoint on Vertex AI to generate new images based on a context image and a textual prompt.

## Features

*   **User-Friendly Interface:** Streamlit app for easy interaction with the pipeline.
*   **Custom Image Upload:** Upload your own images for fine-tuning.
*   **Automated Data Preparation:**
    *   Image compression before GCS upload.
    *   Selection of context and target images.
    *   AI-assisted generation of descriptive suffixes for prompts using Gemini.
    *   Flexible prompt construction including subject name, context ID, and optional body type.
    *   Automatic creation of JSONL training data.
*   **Vertex AI Integration:**
    *   Submission of Imagen LoRA fine-tuning jobs.
    *   Monitoring of fine-tuning job status.
    *   Inference using deployed fine-tuned models.
*   **Configurable:** Project settings, GCS paths, and model parameters can be configured.

## Prerequisites

Before you begin, ensure you have the following:
1.  **Google Cloud Platform (GCP) Account:** With billing enabled.
2.  **GCP Project:** A project created with the Vertex AI API and Cloud Storage API enabled.
3.  **`gcloud` CLI:** Google Cloud SDK installed and authenticated.
4.  **Python:** Python 3.9 or higher.
5.  **Virtual Environment:** Recommended (e.g., `venv`, `conda`).
6.  **Permissions:** Ensure your GCP user or service account has necessary permissions for Vertex AI (training, endpoints) and Cloud Storage (read/write).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
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
    (Ensure `requirements.txt` includes `streamlit`, `google-cloud-aiplatform`, `google-generativeai`, `Pillow`, `google-cloud-storage`, `google-auth`)

4.  **Configure `config.py`:**
    *   Copy or rename `config.py.example` to `config.py` if an example is provided.
    *   Edit `config.py` and fill in your specific GCP `PROJECT_ID`, `LOCATION` (region), `GCS_BUCKET_NAME`, and other relevant parameters like `GCS_IMAGE_UPLOAD_PREFIX`, `GCS_JSONL_PREFIX`, `IMAGEN_BASE_MODEL`, etc.

5.  **Authenticate with Google Cloud:**
    *   If running locally, authenticate the `gcloud` CLI for Application Default Credentials (ADC):
        ```bash
        gcloud auth application-default login
        ```
    *   Ensure the authenticated user/service account has the necessary IAM roles (e.g., Vertex AI User, Storage Object Admin).

## Running the Application

Once setup is complete, run the Streamlit application:
```bash
streamlit run app.py
```
This will typically open the application in your default web browser.

## File Structure

*   `app.py`: The main Streamlit application file containing the UI and workflow logic for both fine-tuning and inference.
*   `utils.py`: Contains helper functions for GCS operations, image processing (compression, resizing), Gemini API calls for description generation, and Vertex AI job submission/monitoring.
*   `config.py`: Stores configuration variables for the project (GCP project ID, GCS bucket, model IDs, etc.). **This file needs to be configured by the user.**
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `TrainingImages/` (example directory): A suggested local directory to store images before uploading for fine-tuning.
*   `README.md`: This file.

## Usage Notes

*   **GCS Bucket:** The application allows specifying a GCS bucket for the fine-tuning pipeline via the sidebar. Ensure this bucket exists in your GCP project and the authenticated user/service account has read/write access.
*   **Image Compression:** Images uploaded for fine-tuning will be automatically compressed if they exceed 5MB. The original local files will be overwritten with the compressed versions if compression is successful.
*   **Prompt Engineering for Fine-Tuning:** The application constructs prompts for the JSONL training data in the format: `"{SubjectName} [{ContextID}] {OptionalBodyType}. {Pronoun} is {DescriptionSuffixFromGemini}"`.
    *   `SubjectName`: User-provided name for the subject.
    *   `ContextID`: User-provided ID (e.g., "1") for the context image.
    *   `OptionalBodyType`: User-provided phrase like "with curvy body type".
    *   `Pronoun`: User-selected pronoun ("She", "He", "They", "It").
    *   `DescriptionSuffixFromGemini`: AI-generated description of the subject's clothing, action, pose, etc., in the target image.
*   **Vertex AI Endpoints:** For the "Imagen Inference" tab, you will need to provide the numeric ID of a deployed Vertex AI Endpoint that hosts your fine-tuned Imagen model.
*   **API Quotas & Costs:** Be mindful of API usage quotas and associated costs on Google Cloud Platform, especially for Vertex AI model training, hosting, and Gemini API calls.
