# utils.py
import base64
import io
import os
import json
import time
import random
import tempfile
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageOps
from google.cloud import storage
from google.cloud import aiplatform
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.protobuf.json_format import ParseDict

from google import genai
from google.genai import types

import streamlit as st # Added for st_log
import subprocess # Ensure subprocess is imported for run_shell_command_inference

try:
    from google.colab import auth as google_colab_auth
except ImportError:
    google_colab_auth = None

import config

PROJECT_ID_CONFIG = config.PROJECT_ID
LOCATION_CONFIG = config.LOCATION
GCS_BUCKET_NAME_CONFIG = getattr(config, 'GCS_BUCKET_NAME', 'your-gcs-bucket-name')
GEMINI_STREAM_MODEL_CONFIG = getattr(config, 'GEMINI_MODEL_FOR_STREAMING_DESCRIPTIONS', "gemini-2.0-flash-001")
TARGET_IMAGE_SIZE_FOR_GEMINI_CONFIG = getattr(config, 'TARGET_IMAGE_SIZE_FOR_GEMINI', (1024, 1024))
DEFAULT_GEMINI_MODEL_ID = "gemini-2.0-flash"

def initialize_ai_platform():
    print(f"Initializing AI Platform (Project: {config.PROJECT_ID}, Location: {config.LOCATION})...")
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    print("AI Platform initialized.")

storage_client = None
def get_storage_client():
    global storage_client
    if storage_client is None:
        storage_client = storage.Client(project=PROJECT_ID_CONFIG)
    return storage_client

def upload_to_gcs(local_file_path: str, gcs_bucket_name: str, gcs_blob_name: str) -> Optional[str]:
    client = get_storage_client()
    try:
        bucket = client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_file_path)
        gcs_uri = f"gs://{gcs_bucket_name}/{gcs_blob_name}"
        print(f"Successfully uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        print(f"Error uploading {local_file_path} to GCS: {e}")
        return None

def list_gcs_files(gcs_bucket_name: str, gcs_prefix: str, extensions: Tuple[str, ...]) -> List[str]:
    client = get_storage_client()
    blobs = client.list_blobs(gcs_bucket_name, prefix=gcs_prefix)
    uris = []
    for blob in blobs:
        if blob.name.lower().endswith(extensions) and not blob.name.endswith('/'):
            uris.append(f"gs://{gcs_bucket_name}/{blob.name}")
    return uris

def get_gcs_blob_size(gcs_uri: str) -> Optional[int]:
    client = get_storage_client()
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)
        if blob: return blob.size
        else: print(f"Warning: Blob not found at {gcs_uri}"); return None
    except Exception as e: print(f"Error getting size for GCS URI {gcs_uri}: {e}"); return None

def download_from_gcs(gcs_uri: str, local_download_path: str) -> bool:
    client = get_storage_client()
    try:
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_download_path)
        print(f"Successfully downloaded {gcs_uri} to {local_download_path}")
        return True
    except Exception as e: print(f"Error downloading {gcs_uri}: {e}"); return False

def compress_image_if_needed(
    local_image_path: str,
    max_size_bytes: int = 5 * 1024 * 1024, 
    quality: int = 85, 
    optimize: bool = True
) -> Tuple[str, bool]:
    try:
        original_size = os.path.getsize(local_image_path)
        if original_size <= max_size_bytes:
            return local_image_path, False
        img = Image.open(local_image_path)
        file_ext = os.path.splitext(local_image_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg'] and img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'P': 
             img = img.convert('RGBA' if 'A' in img.mode else 'RGB')
        temp_fd, temp_output_path = tempfile.mkstemp(suffix=file_ext)
        os.close(temp_fd) 
        img_modified = False 
        try:
            current_img = img.copy() 
            if file_ext in ['.jpg', '.jpeg']:
                current_quality = quality
                current_img.save(temp_output_path, quality=current_quality, optimize=optimize)
                img_modified = True
                while os.path.getsize(temp_output_path) > max_size_bytes and current_quality > 20:
                    current_quality -= 10 
                    save_img = current_img.convert('RGB') if current_img.mode in ['RGBA', 'P', 'LA'] else current_img
                    save_img.save(temp_output_path, quality=current_quality, optimize=optimize)
            elif file_ext == '.png':
                current_img.save(temp_output_path, optimize=optimize, compress_level=9)
                img_modified = True
            else: 
                try:
                    current_img.save(temp_output_path)
                    img_modified = True
                except Exception:
                    os.remove(temp_output_path) 
                    temp_fd_png, temp_output_path = tempfile.mkstemp(suffix=".png") 
                    os.close(temp_fd_png)
                    current_img.convert('RGBA').save(temp_output_path, optimize=True, compress_level=9)
                    img_modified = True
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > max_size_bytes:
                w, h = current_img.size
                size_ratio = os.path.getsize(temp_output_path) / max_size_bytes
                if size_ratio > 1: 
                    scale_factor = (1.0 / size_ratio)**0.5 
                    new_w = int(w * scale_factor * 0.95) 
                    new_h = int(h * scale_factor * 0.95)
                    if new_w > 0 and new_h > 0:
                        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        img_modified = True
                        if file_ext in ['.jpg', '.jpeg']:
                             save_img = current_img.convert('RGB') if current_img.mode in ['RGBA', 'P', 'LA'] else current_img
                             save_img.save(temp_output_path, quality=75, optimize=True) 
                        elif file_ext == '.png' or temp_output_path.endswith(".png"): 
                             current_img.save(temp_output_path, optimize=True, compress_level=9)
                        else:
                             current_img.save(temp_output_path)
            if os.path.exists(temp_output_path) and img_modified:
                compressed_size = os.path.getsize(temp_output_path)
                if compressed_size < original_size:
                    os.replace(temp_output_path, local_image_path)
                    return local_image_path, True
                else: 
                    os.remove(temp_output_path)
                    return local_image_path, False
            elif not img_modified:
                 if os.path.exists(temp_output_path): os.remove(temp_output_path) 
                 return local_image_path, False
            else: 
                return local_image_path, False
        finally:
            if os.path.exists(temp_output_path) and temp_output_path != local_image_path: 
                try: os.remove(temp_output_path)
                except OSError: pass
    except Exception as e:
        print(f"Error compressing image {local_image_path}: {e}. Using original.")
        return local_image_path, False

def get_image_bytes_for_vertex_imagen_api(
    image_path_or_gcs_uri: str,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE_FOR_GEMINI_CONFIG
) -> Tuple[Optional[bytes], Optional[str]]:
    local_image_path = image_path_or_gcs_uri
    temp_downloaded_file = None
    if image_path_or_gcs_uri.startswith("gs://"):
        temp_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        temp_downloaded_file = os.path.join(temp_dir, f"imagen_api_img_{temp_id}_{os.path.basename(image_path_or_gcs_uri)}")
        if not download_from_gcs(image_path_or_gcs_uri, temp_downloaded_file):
            return None, None
        local_image_path = temp_downloaded_file
    if not os.path.exists(local_image_path):
        if temp_downloaded_file and os.path.exists(temp_downloaded_file): os.remove(temp_downloaded_file)
        return None, None
    try:
        with Image.open(local_image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            output_io = io.BytesIO()
            img.save(output_io, format='PNG')
            return output_io.getvalue(), 'image/png'
    except Exception as e:
        print(f"Error processing image {local_image_path} for Vertex API: {e}")
        return None, None
    finally:
        if temp_downloaded_file and os.path.exists(temp_downloaded_file): os.remove(temp_downloaded_file)

def generate_description_vertex_stream_from_gcs_uri(
    gcs_file_uri: str,
    subject_name: str, # Parameter for the subject's name
    project_id: str = PROJECT_ID_CONFIG,
    location: str = LOCATION_CONFIG,
    model_id: str = GEMINI_STREAM_MODEL_CONFIG
) -> str:
    """
    Generates a descriptive suffix for a subject in an image using Gemini,
    aligning with the user's feedback and reference snippet structure.
    """
    print(f"Initializing Gemini Client (Project: {project_id}, Location: {location}) for suffix generation...")
    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
    except Exception as e:
        print(f"Error initializing genai.Client: {e}")
        return f"[Error initializing Gemini client: {e}]"

    print(f"Processing GCS URI: {gcs_file_uri} with model: {model_id} for subject: {subject_name}")
    
    mime_type = "image/png" 
    if gcs_file_uri.lower().endswith((".jpg", ".jpeg")): mime_type = "image/jpeg"
    elif gcs_file_uri.lower().endswith(".webp"): mime_type = "image/webp"
    
    image_part = types.Part.from_uri(
        file_uri=gcs_file_uri, # Corrected: was 'uri' in some previous versions
        mime_type=mime_type,
    )

    # Specific instruction for Gemini, incorporating the subject_name
    llm_instruction = (
        f"Describe the person '{subject_name}' in the provided image. Focus on their clothing, action, pose, hairstyle, and other notable visual details. \
            Start your description with a verb (e.g., 'is wearing...', 'stands...'), a prepositional phrase (e.g., 'in a red dress...', 'with flowing hair...'), \
            or an adjective phrase that would naturally complete a sentence about them. Do not repeat the subject's name ('{subject_name}') in your response. \
            Your response should be a concise phrase. Example: if '{subject_name}' is in a green shirt, you might respond 'wearing a green shirt and jeans.' or 'with a thoughtful expression, looking to the side.'"
    )
    text_part = types.Part.from_text(text=llm_instruction)
    
    contents = [
        types.Content(
            role="user",
            parts=[image_part, text_part]
        )
    ]
    
    safety_settings_list = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    generation_config_obj = types.GenerateContentConfig( 
        temperature=1.0, 
        top_p=1.0,       
        max_output_tokens=256, # Reduced for a suffix
        safety_settings=safety_settings_list,
    )

    full_description_suffix = ""
    print(f"Generating content stream with model: {model_id}...")
    try:
        response_stream = client.models.generate_content_stream(
            model=model_id, 
            contents=contents,
            config=generation_config_obj, 
        )
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                full_description_suffix += chunk.text
        
        if not full_description_suffix:
            block_reason_msg = ""
            if hasattr(response_stream, 'prompt_feedback') and response_stream.prompt_feedback and response_stream.prompt_feedback.block_reason:
                 block_reason_msg = f"Reason: {response_stream.prompt_feedback.block_reason}"
            if block_reason_msg:
                print(f"Content generation blocked. {block_reason_msg}")
                return f"[Error: Content generation blocked. {block_reason_msg}]"
            else:
                print("Stream finished, but no descriptive text was generated.")
                return "[Warning: No descriptive text generated]"
        print("Stream finished.")
        return full_description_suffix.strip()
    except Exception as e:
        print(f"Error during generate_content_stream: {e}")
        error_message = str(e)
        if hasattr(e, 'message') and isinstance(e.message, str): 
            error_message = e.message
        elif hasattr(e, 'details') and callable(e.details): 
            try: error_message = e.details()
            except: pass
        return f"[Error during content generation stream: {error_message}]"
    
def get_vertex_ai_client(location: str) -> aiplatform.gapic.JobServiceClient:
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    return aiplatform.gapic.JobServiceClient(client_options=client_options)
    
def get_access_token() -> str:
    credentials, _ = google_auth_default()
    credentials.refresh(GoogleAuthRequest())
    return credentials.token

def create_imagen_tuning_job(
    project_id: str, location: str, base_model_uri: str,
    tuned_model_display_name: str, training_dataset_uri: str,
    epochs: int, adapter_size: str, learning_rate_multiplier: float
) -> Optional[str]:
    tuning_job_payload = {
        "baseModel": base_model_uri, 
        "tunedModelDisplayName": tuned_model_display_name, 
        "supervisedTuningSpec": { 
            "trainingDatasetUri": training_dataset_uri, 
            "hyperParameters": {
                "epochCount": epochs, "adapterSize": adapter_size, 
                "learningRateMultiplier": learning_rate_multiplier, 
            },
        },
    }
    parent = f"projects/{project_id}/locations/{location}"
    api_endpoint = f"https://{location}-aiplatform.googleapis.com"
    api_path = f"v1/{parent}/tuningJobs" 
    headers = {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json; charset=utf-8"}
    import requests
    print(f"\nSubmitting Tuning Job to {api_endpoint}/{api_path} with payload: {json.dumps(tuning_job_payload, indent=2)}")
    try:
        response = requests.post(f"{api_endpoint}/{api_path}", headers=headers, data=json.dumps(tuning_job_payload))
        response.raise_for_status()
        response_json = response.json()
        job_name = response_json.get("name")
        if job_name: print(f"Successfully submitted tuning job. Job Name/Path: {job_name}"); return job_name
        else: print(f"Tuning job submitted, but 'name' not found in response: {response_json}"); return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}\nResponse content: {response.text}")
        return None
    except Exception as e: print(f"An error occurred: {e}"); return None

def monitor_tuning_job(job_name_path: str, location: str) -> Optional[Dict[str, Any]]:
    api_endpoint = f"https://{location}-aiplatform.googleapis.com"
    if not job_name_path.startswith("projects/"): 
        print(f"Invalid job_name_path format: {job_name_path}")
        return None
    api_url = f"{api_endpoint}/v1/{job_name_path}" 
    headers = {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json; charset=utf-8"}
    import requests
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        status_data = response.json()
        print(f"Job State: {status_data.get('state', 'UNKNOWN')}")
        if status_data.get('tunedModel'): print(f"Tuned Model Endpoint: {status_data['tunedModel'].get('endpoint')}")
        return status_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}\nResponse content: {response.text}")
        return None
    except Exception as e: print(f"An error occurred: {e}"); return None

if __name__ == '__main__':
    print("Utils.py loaded.")
    pass

# Helper functions moved from app.py
def st_log(message, level="info"): 
    if level == "info": st.info(message)
    elif level == "success": st.success(message)
    elif level == "warning": st.warning(message)
    elif level == "error": st.error(message)
    else: st.write(message)

def run_shell_command_inference(command: str, capture_output=True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True, check=False)
        if result.stderr: print(f"Shell Command STDERR (Inference):\n{result.stderr}")
        if result.returncode != 0: print(f"WARNING: Shell command (Inference) '{command.split()[0]}...' failed with return code {result.returncode}")
    except Exception as e:
        print(f"Exception during shell command execution for (Inference) '{command.split()[0]}...': {e}")
        return subprocess.CompletedProcess(args=command, returncode=-1, stdout="", stderr=str(e))
    return result
