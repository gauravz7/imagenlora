# app.py
import streamlit as st
import os
import time
import json
import random
from itertools import product
import base64 
import io 
import subprocess 
import tempfile 
from PIL import Image 

import config
import utils 

st.set_page_config(
    page_title="Imagen LoRA Tools", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def get_image_bytes_for_vertex_imagen_api_inference(image_path_or_gcs_uri, target_size: tuple = (1024, 1024)):
    local_image_path = image_path_or_gcs_uri
    temp_downloaded_path = None 
    if image_path_or_gcs_uri.startswith("gs://"):
        st.warning("GCS URI detected. Ensure gsutil is configured.")
        base_name = os.path.basename(image_path_or_gcs_uri)
        with tempfile.NamedTemporaryFile(delete=False, prefix="vertex_img_dl_", suffix=os.path.splitext(base_name)[1]) as tmp_dl_file:
            temp_downloaded_path = tmp_dl_file.name
        gsutil_cp_command = f"gsutil cp '{image_path_or_gcs_uri}' '{temp_downloaded_path}'"
        cp_result = run_shell_command_inference(gsutil_cp_command)
        if cp_result.returncode != 0:
            if os.path.exists(temp_downloaded_path): os.remove(temp_downloaded_path)
            raise FileNotFoundError(f"Failed to download GCS image: {image_path_or_gcs_uri}. Gsutil stderr: {cp_result.stderr}")
        local_image_path = temp_downloaded_path
    if not os.path.exists(local_image_path):
        raise FileNotFoundError(f"Image file not found: {local_image_path}")
    try:
        with Image.open(local_image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            if img.size != target_size:
                try: img = img.resize(target_size, Image.Resampling.LANCZOS)
                except AttributeError: img = img.resize(target_size, Image.LANCZOS)
            output_io = io.BytesIO()
            img.save(output_io, format='PNG')
            image_bytes = output_io.getvalue()
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.remove(temp_downloaded_path)
    return image_bytes, 'image/png'

def render_finetuning_pipeline_page():
    st.title("🖼️ Imagen LoRA Fine-Tuning with Custom Images")
    
    default_bucket_ft = config.GCS_BUCKET_NAME
    if 'user_gcs_bucket_name_ft' not in st.session_state:
        st.session_state.user_gcs_bucket_name_ft = default_bucket_ft
    
    user_gcs_bucket_name_ft_input = st.sidebar.text_input(
        "GCS Bucket Name (Fine-Tuning)", 
        value=st.session_state.user_gcs_bucket_name_ft, 
        key="gcs_bucket_name_ft_widget",
        help="GCS bucket for storing fine-tuning assets (images, JSONL)."
    )
    if user_gcs_bucket_name_ft_input: 
        st.session_state.user_gcs_bucket_name_ft = user_gcs_bucket_name_ft_input
    
    effective_gcs_bucket_name = st.session_state.user_gcs_bucket_name_ft or default_bucket_ft
    
    st.caption(f"Project: {config.PROJECT_ID}, Location: {config.LOCATION}, Bucket: {effective_gcs_bucket_name}")

    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'gcs_jsonl_uri' not in st.session_state: st.session_state.gcs_jsonl_uri = None
    if 'tuning_job_path' not in st.session_state: st.session_state.tuning_job_path = None
    if 'all_gcs_uris_from_upload' not in st.session_state: st.session_state.all_gcs_uris_from_upload = []
    if 'context_uris' not in st.session_state: st.session_state.context_uris = []
    if 'target_uris' not in st.session_state: st.session_state.target_uris = []
    if 'target_descriptions' not in st.session_state: st.session_state.target_descriptions = {} 
    if 'subject_name_for_jsonl' not in st.session_state: st.session_state.subject_name_for_jsonl = "SXR person"
    if 'body_type_input_for_jsonl' not in st.session_state: st.session_state.body_type_input_for_jsonl = ""
    if 'subject_pronoun_for_jsonl' not in st.session_state: st.session_state.subject_pronoun_for_jsonl = "She"


    with st.expander("Step 0: Setup & Authentication", expanded=True):
        if st.button("Initialize & Authenticate Clients", key="auth_button_finetune"):
            try:
                with st.spinner("Initializing..."):
                    utils.get_storage_client(); utils.initialize_ai_platform()
                st.session_state.authenticated = True
                st_log("Clients initialized!", "success")
            except Exception as e: st_log(f"Error: {e}", "error")
        if st.session_state.authenticated: st.success("✅ Clients Authenticated.")
        else: st.warning("⚠️ Clients not authenticated.")

    with st.expander("Step 1: Data Preparation", expanded=False):
        if not st.session_state.authenticated: st.warning("Authenticate in Step 0.")
        else:
            st.subheader("1.1 Upload Local Images to GCS")
            local_image_dir = st.text_input("Local Image Folder:", config.LOCAL_IMAGE_DIR, key="local_img_dir_ft")
            gcs_image_upload_path = f"gs://{effective_gcs_bucket_name}/{config.GCS_IMAGE_UPLOAD_PREFIX.strip('/')}"
            st.write(f"Images will be uploaded from: `{local_image_dir}` to GCS path: `{gcs_image_upload_path}/`")

            if st.button("Upload to GCS", key="upload_gcs_ft"):
                with st.spinner(f"Uploading images from {local_image_dir} to GCS..."):
                    uploaded_uris = []
                    MAX_IMAGE_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024 
                    images_in_local_dir_upload = [f for f in os.listdir(local_image_dir) if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)] if os.path.isdir(local_image_dir) else []
                    if not images_in_local_dir_upload: st_log("No images found or directory invalid.", "warning")
                    else:
                        for filename in images_in_local_dir_upload:
                            local_file_path = os.path.join(local_image_dir, filename)
                            _, compressed = utils.compress_image_if_needed(local_file_path, max_size_bytes=MAX_IMAGE_UPLOAD_SIZE_BYTES)
                            gcs_blob_name = f"{config.GCS_IMAGE_UPLOAD_PREFIX.rstrip('/')}/{filename}"
                            gcs_uri = utils.upload_to_gcs(local_file_path, effective_gcs_bucket_name, gcs_blob_name)
                            if gcs_uri:
                                uploaded_uris.append(gcs_uri)
                                st_log(f"Uploaded {'compressed ' if compressed else ''}{filename} to {gcs_uri}", "info")
                            else: st_log(f"Failed to upload {filename}", "error")
                        if uploaded_uris:
                            st.session_state.all_gcs_uris_from_upload = uploaded_uris
                            st_log(f"Successfully uploaded {len(uploaded_uris)} images.", "success")
                        else: st_log("Image upload failed/no images uploaded.", "error")
            
            if st.checkbox("Alternatively, list images directly from GCS", key="list_gcs_direct_ft"):
                if st.button("List Images from GCS", key="list_gcs_button_ft"):
                    with st.spinner("Listing images..."):
                        gcs_uris = utils.list_gcs_files(effective_gcs_bucket_name, config.GCS_IMAGE_UPLOAD_PREFIX, config.VALID_IMAGE_EXTENSIONS)
                        if gcs_uris:
                            st.session_state.all_gcs_uris_from_upload = gcs_uris
                            st_log(f"Found {len(gcs_uris)} images in GCS.", "success")
                        else: st_log("No images found in GCS path.", "warning")
            if st.session_state.all_gcs_uris_from_upload: st.success(f"Working with {len(st.session_state.all_gcs_uris_from_upload)} GCS image URIs.")

            st.subheader("1.2 Select Context & Target Images")
            context_percentage_select = st.slider("Percentage for Context:", 1, 99, 30, 1, key="context_perc_slider_ft_select") / 100.0
            if st.button("Select Context/Target Images", key="select_images_button_ft_select", disabled=not st.session_state.all_gcs_uris_from_upload):
                with st.spinner("Selecting images..."):
                    all_uris_sel = list(st.session_state.all_gcs_uris_from_upload)
                    random.shuffle(all_uris_sel)
                    num_context_sel = int(len(all_uris_sel) * context_percentage_select)
                    if num_context_sel == 0 and len(all_uris_sel) > 0: num_context_sel = 1
                    if len(all_uris_sel) > 1 and num_context_sel >= len(all_uris_sel): num_context_sel = len(all_uris_sel) - 1
                    st.session_state.context_uris = all_uris_sel[:num_context_sel]
                    st.session_state.target_uris = all_uris_sel[num_context_sel:]
                    st_log(f"Selected {len(st.session_state.context_uris)} context and {len(st.session_state.target_uris)} target images.", "success")

            st.subheader("1.3 Generate Description Suffixes for Target Images")
            # Use session state for persistence and to ensure values are available for JSONL creation
            st.session_state.subject_name_for_jsonl = st.text_input(
                "Subject Name (e.g., Sandra):", 
                value=st.session_state.subject_name_for_jsonl, 
                key="subject_name_input_ft_key_ui"
            )
            st.session_state.body_type_input_for_jsonl = st.text_input(
                "Optional: Body Type/Shape (e.g., curvy body type):", 
                value=st.session_state.body_type_input_for_jsonl, 
                key="body_type_input_ft_val_key_ui"
            )
            pronoun_options = ["She", "He", "They", "It"]
            # Ensure default index is valid
            try:
                pronoun_default_index = pronoun_options.index(st.session_state.subject_pronoun_for_jsonl)
            except ValueError:
                pronoun_default_index = 0 # Default to "She" if current value is not in options
                st.session_state.subject_pronoun_for_jsonl = pronoun_options[0]

            st.session_state.subject_pronoun_for_jsonl = st.selectbox(
                "Subject Pronoun:", 
                pronoun_options, 
                index=pronoun_default_index, 
                key="pronoun_input_ft_key_ui"
            )
            
            if st.button("Generate Description Suffixes", key="gen_desc_suffixes_ft", disabled=not st.session_state.target_uris or not st.session_state.subject_name_for_jsonl.strip()):
                with st.spinner("Generating suffixes..."):
                    desc_map_suffixes = {}
                    prog_bar = st.progress(0)
                    stat_text = st.empty()
                    for i, target_gcs_url in enumerate(st.session_state.target_uris):
                        stat_text.text(f"Processing image {i+1}/{len(st.session_state.target_uris)} for suffix...")
                        # The utils function now takes subject_name and constructs the detailed instruction internally
                        description_suffix = utils.generate_description_vertex_stream_from_gcs_uri(
                            gcs_file_uri=target_gcs_url,
                            subject_name=st.session_state.subject_name_for_jsonl 
                        )
                        desc_map_suffixes[target_gcs_url] = description_suffix
                        log_lvl = "warning" if "[Error:" in description_suffix or "[Warning:" in description_suffix else "info"
                        st_log(f"Suffix for {os.path.basename(target_gcs_url)}: {description_suffix[:100]}...", log_lvl)
                        prog_bar.progress((i + 1) / len(st.session_state.target_uris))
                    st.session_state.target_descriptions = desc_map_suffixes
                    st_log(f"Generated suffixes for {len(desc_map_suffixes)} images.", "success")
                    stat_text.text("Suffix generation complete!")
            
            st.subheader("1.4 Create Training JSONL File")
            max_pairs_jsonl = st.text_input("Max Training Pairs (0 for no limit):", "1000", key="max_pairs_jsonl_ft")
            user_context_id = st.text_input("Context ID for prompts (e.g., 1):", "1", key="user_context_id_jsonl_ft").strip() or "1"
            
            # Retrieve from session state for consistency
            current_subject_name_for_jsonl = st.session_state.subject_name_for_jsonl
            current_body_type_for_jsonl = st.session_state.body_type_input_for_jsonl
            current_pronoun_for_jsonl = st.session_state.subject_pronoun_for_jsonl

            if st.button("Create & Upload JSONL", key="create_jsonl_ft_btn", disabled=not st.session_state.context_uris or not st.session_state.target_descriptions or not current_subject_name_for_jsonl.strip()):
                with st.spinner("Creating JSONL..."):
                    valid_suffixes = {uri: sfx for uri, sfx in st.session_state.target_descriptions.items() if not (sfx.startswith("[Error:") or sfx.startswith("[Warning:"))}
                    if not valid_suffixes: st_log("No valid suffixes for JSONL.", "error")
                    else:
                        jsonl_lines_data = []
                        body_type_segment = f" with {current_body_type_for_jsonl.strip()}" if current_body_type_for_jsonl.strip() else ""
                        base_intro = f"{current_subject_name_for_jsonl} [{user_context_id}]"
                        
                        all_pairs_jsonl = list(product(st.session_state.context_uris, valid_suffixes.keys()))
                        limit_pairs = int(max_pairs_jsonl) if max_pairs_jsonl.isdigit() and int(max_pairs_jsonl) > 0 else len(all_pairs_jsonl)
                        selected_pairs_jsonl = random.sample(all_pairs_jsonl, min(limit_pairs, len(all_pairs_jsonl)))
                        
                        for ctx_gcs, tgt_gcs in selected_pairs_jsonl:
                            suffix = valid_suffixes.get(tgt_gcs, "")
                            cleaned_suffix = ' '.join(suffix.strip().strip('"\'').splitlines())
                            
                            prompt_part1 = f"{base_intro}{body_type_segment}."
                            prompt_part2 = f" {current_pronoun_for_jsonl} is {cleaned_suffix}" # Add space before pronoun
                            final_prompt_jsonl = f"{prompt_part1}{prompt_part2}".strip().replace("  ", " ")
                            
                            def get_mime(uri_str): return f"image/{os.path.splitext(uri_str)[1].lower().replace('.', '') or 'png'}"
                            entry = {"prompt": final_prompt_jsonl,
                                     "context_images": [{"context_id": user_context_id, "context_image": {"mime_type": get_mime(ctx_gcs), "file_uri": ctx_gcs}, "context_prompt": current_subject_name_for_jsonl}],
                                     "target_image": {"mime_type": get_mime(tgt_gcs), "file_uri": tgt_gcs}}
                            jsonl_lines_data.append(json.dumps(entry))
                        
                        if jsonl_lines_data:
                            local_jsonl_path_save = os.path.join(os.getcwd(), config.LOCAL_TRAIN_JSONL_FILENAME)
                            with open(local_jsonl_path_save, 'w') as f_jsonl: f_jsonl.write('\n'.join(jsonl_lines_data))
                            gcs_jsonl_blob_name_upload = f"{config.GCS_JSONL_PREFIX.rstrip('/')}/{config.LOCAL_TRAIN_JSONL_FILENAME}"
                            gcs_uri_upload = utils.upload_to_gcs(local_jsonl_path_save, effective_gcs_bucket_name, gcs_jsonl_blob_name_upload)
                            if gcs_uri_upload:
                                st.session_state.gcs_jsonl_uri = gcs_uri_upload
                                st_log(f"Uploaded JSONL to: {gcs_uri_upload}", "success")
                            else: st_log("Failed to upload JSONL.", "error")
                        else: st_log("No JSONL data generated.", "error")
    
    with st.expander("Step 2: Train Model", expanded=False):
        if not st.session_state.gcs_jsonl_uri: st.warning("Complete Data Prep (Step 1) first.")
        else:
            st.success(f"Using training JSONL: {st.session_state.gcs_jsonl_uri}")
            st.subheader("Fine-Tuning Parameters")
            tuned_model_name = st.text_input("Tuned Model Display Name:", config.DEFAULT_TUNED_MODEL_DISPLAY_NAME, key="tuned_name_ft_train")
            adapter_options = ["ADAPTER_SIZE_ONE", "ADAPTER_SIZE_FOUR", "ADAPTER_SIZE_EIGHT", "ADAPTER_SIZE_SIXTEEN", "ADAPTER_SIZE_THIRTY_TWO"]
            adapter_size = st.selectbox("Adapter Size:", adapter_options, index=adapter_options.index(config.DEFAULT_ADAPTER_SIZE), key="adapter_size_ft_train")
            epochs = st.number_input("Epochs:", min_value=1, value=config.DEFAULT_EPOCHS, key="epochs_input_ft_train")
            lr_multiplier = st.number_input("Learning Rate Multiplier:", min_value=0.001, value=config.DEFAULT_LEARNING_RATE_MULTIPLIER, format="%.3f", key="lr_input_ft_train")

            if st.button("Submit Fine-Tuning Job", key="submit_train_button_ft_train"):
                with st.spinner("Submitting fine-tuning job..."):
                    job_path = utils.create_imagen_tuning_job(
                        project_id=config.PROJECT_ID, location=config.LOCATION,
                        base_model_uri=config.IMAGEN_BASE_MODEL, tuned_model_display_name=tuned_model_name,
                        training_dataset_uri=st.session_state.gcs_jsonl_uri, epochs=epochs,
                        adapter_size=adapter_size, learning_rate_multiplier=lr_multiplier
                    )
                    if job_path:
                        st.session_state.tuning_job_path = job_path
                        st_log(f"Tuning Job Submitted: {job_path}", "success")
                    else: st_log("Failed to submit tuning job.", "error")

    with st.expander("Step 3: Monitor Fine-Tuning Job", expanded=False):
        manual_job_path_input_mon = st.text_input("Or, enter Tuning Job Path manually:", key="manual_job_path_monitor_ft_mon")
        current_job_to_monitor_mon = manual_job_path_input_mon if manual_job_path_input_mon else st.session_state.get('tuning_job_path')

        if not current_job_to_monitor_mon: st.info("No tuning job to monitor.")
        else:
            st.success(f"Monitoring job: {current_job_to_monitor_mon}")
            if st.button("Refresh Job Status", key="refresh_monitor_button_ft_mon"):
                with st.spinner("Fetching job status..."):
                    status_data = utils.monitor_tuning_job(job_name_path=current_job_to_monitor_mon, location=config.LOCATION)
                    if status_data:
                        st_log("Job Status Refreshed.", "info")
                        st.json(status_data)
                        job_state = status_data.get("state", "JOB_STATE_UNSPECIFIED")
                        if job_state == "JOB_STATE_SUCCEEDED":
                            st.balloons(); st_log("SUCCESS: Tuning job completed!", "success")
                            endpoint = status_data.get("tunedModel", {}).get("endpoint")
                            if endpoint: st_log(f"Tuned Model Endpoint: {endpoint}", "success")
                        elif job_state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                            st_log(f"ERROR: Job status: {job_state}.", "error")
                        else: st_log(f"INFO: Job status: {job_state}.", "info")
                    else: st_log("Failed to retrieve job status.", "error")


def render_imagen_inference_page():
    st.title("🎨 Vertex AI Imagen Inference (Fine-Tuned Model)")
    DEFAULT_PROJECT_ID_INF = config.PROJECT_ID 
    DEFAULT_LOCATION_INF = config.LOCATION   
    DEFAULT_PREDICT_ENDPOINT_ID_INF = "" 
    DEFAULT_CONTEXT_ID_INF = "1"
    DEFAULT_SUBJECT_DESC_INF = "sandra"
    DEFAULT_PROMPT_SCENE_DESCRIPTION_INF = "A beautiful woman in a bright red oversized hoodie with printed text on the front, paired with a plaid shirt layered underneath. She's wearing brown cargo shorts that reach below the knee, along with red ankle socks and casual white sneakers with brown accents. The background has a street setting with storefronts, giving it an urban vibe. looking straight into camera"
    DEFAULT_NEGATIVE_PROMPT_INF = "blurry, deformed, ugly, low quality, text, watermark, noise, messy"
    DEFAULT_SEED_INF = 42
    DEFAULT_SAMPLE_COUNT_INF = 1
    DEFAULT_ASPECT_RATIO_INF = "9:16"

    st.sidebar.header("⚙️ GCP & Model Configuration")
    PROJECT_ID_INF = st.sidebar.text_input("GCP Project ID", value=DEFAULT_PROJECT_ID_INF, key="inf_project_id")
    LOCATION_INF = st.sidebar.text_input("GCP Location (Region)", value=DEFAULT_LOCATION_INF, key="inf_location")
    PREDICT_ENDPOINT_ID_INF = st.sidebar.text_input("Vertex AI Endpoint ID (Numeric)", value=DEFAULT_PREDICT_ENDPOINT_ID_INF, help="The numeric ID of your deployed fine-tuned Imagen model endpoint.", key="inf_endpoint_id")

    st.sidebar.header("🖼️ Context Image & Subject")
    CONTEXT_ID_FOR_PREDICTION_INF = st.sidebar.text_input("Context ID (from training)", value=DEFAULT_CONTEXT_ID_INF, key="inf_context_id")
    SUBJECT_DESC_FOR_CONTEXT_INF = st.sidebar.text_input("Subject Description (from training)", value=DEFAULT_SUBJECT_DESC_INF, key="inf_subject_desc")

    col1_inf, col2_inf = st.columns(2)
    with col1_inf:
        st.subheader("👤 Input Context Image")
        uploaded_context_image_inf = st.file_uploader("Upload context image", type=["png", "jpg", "jpeg", "webp"], key="inf_uploader")
        if uploaded_context_image_inf: st.image(uploaded_context_image_inf, caption="Uploaded Context Image", use_column_width=True)
        
        subject_reference_placeholder_inf = f"[{CONTEXT_ID_FOR_PREDICTION_INF}]"
        default_full_prompt_inf = f"{SUBJECT_DESC_FOR_CONTEXT_INF} {subject_reference_placeholder_inf}, {DEFAULT_PROMPT_SCENE_DESCRIPTION_INF}"
        st.info(f"Use `{subject_reference_placeholder_inf}` in your prompt to refer to the subject.")
        predict_prompt_inf = st.text_area("📝 Prediction Prompt", value=default_full_prompt_inf, height=150, key="inf_prompt")

    st.sidebar.header("✨ Prediction Parameters")
    negative_prompt_inf = st.sidebar.text_input("Negative Prompt", DEFAULT_NEGATIVE_PROMPT_INF, key="inf_neg_prompt")
    seed_inf = st.sidebar.number_input("Seed (0 for random)", value=DEFAULT_SEED_INF, min_value=0, key="inf_seed")
    sample_count_inf = st.sidebar.number_input("Number of Images", min_value=1, max_value=4, value=DEFAULT_SAMPLE_COUNT_INF, key="inf_sample_count")
    
    aspect_ratio_options_inf = {"Portrait (9:16)": "9:16", "Landscape (16:9)": "16:9", "Square (1:1)": "1:1"}
    default_aspect_key_inf = [k for k,v in aspect_ratio_options_inf.items() if v == DEFAULT_ASPECT_RATIO_INF][0]
    selected_aspect_display_inf = st.sidebar.selectbox("Aspect Ratio", list(aspect_ratio_options_inf.keys()), index=list(aspect_ratio_options_inf.keys()).index(default_aspect_key_inf), key="inf_aspect_ratio")
    aspect_ratio_inf = aspect_ratio_options_inf[selected_aspect_display_inf]

    if st.button("🎨 Generate Image", type="primary", use_container_width=True, key="inf_generate_button"):
        if not all([PROJECT_ID_INF, LOCATION_INF, PREDICT_ENDPOINT_ID_INF, CONTEXT_ID_FOR_PREDICTION_INF, SUBJECT_DESC_FOR_CONTEXT_INF, uploaded_context_image_inf, predict_prompt_inf]):
            st.error("🚫 Please fill in all configuration details and upload a context image.")
        else:
            with col2_inf:
                st.subheader("⏳ Processing...")
                progress_bar_render_inf = st.progress(0, text="Initializing...")
            
            temp_image_file_path_inf = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_context_image_inf.name)[1]) as tmp_file_inf:
                    tmp_file_inf.write(uploaded_context_image_inf.getvalue())
                    temp_image_file_path_inf = tmp_file_inf.name
                progress_bar_render_inf.progress(10, text="Context image saved.")

                context_img_bytes_inf, context_img_mime_type_inf = get_image_bytes_for_vertex_imagen_api_inference(temp_image_file_path_inf)
                context_img_b64_inf = base64.b64encode(context_img_bytes_inf).decode('utf-8')
                progress_bar_render_inf.progress(30, text="Context image processed.")

                VERTEX_API_EP_RENDER_INF = f"https://{LOCATION_INF}-aiplatform.googleapis.com"
                predict_api_path_inf = f"v1/projects/{PROJECT_ID_INF}/locations/{LOCATION_INF}/endpoints/{PREDICT_ENDPOINT_ID_INF}:predict"
                
                try: context_id_int_inf = int(CONTEXT_ID_FOR_PREDICTION_INF)
                except ValueError: 
                    st.error(f"Context ID '{CONTEXT_ID_FOR_PREDICTION_INF}' must be an integer."); progress_bar_render_inf.empty(); st.stop()

                predict_payload_inf = {
                    'instances': [{'prompt': predict_prompt_inf, 'referenceImages': [{'referenceType': 'REFERENCE_TYPE_SUBJECT', 'referenceId': context_id_int_inf, 'referenceImage': {'bytesBase64Encoded': context_img_b64_inf, 'mimeType': context_img_mime_type_inf}, 'subjectImageConfig': {'subjectDescription': SUBJECT_DESC_FOR_CONTEXT_INF, 'subjectType': 'SUBJECT_TYPE_PERSON'},}]}],
                    'parameters': {'negativePrompt': negative_prompt_inf, 'sampleCount': sample_count_inf, 'aspectRatio': aspect_ratio_inf}
                }
                if seed_inf > 0: predict_payload_inf['parameters']['seed'] = seed_inf
                
                with col1_inf:
                    with st.expander("🔍 View API Request Payload (Debug)", expanded=False): st.json(predict_payload_inf)
                progress_bar_render_inf.progress(50, text="API payload ready.")

                temp_payload_filepath_inf = None
                predict_res_process_inf = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_payload_file_inf:
                        json.dump(predict_payload_inf, tmp_payload_file_inf)
                        temp_payload_filepath_inf = tmp_payload_file_inf.name
                    
                    curl_cmd_predict_inf = (f'curl -s -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json; charset=utf-8" "{VERTEX_API_EP_RENDER_INF}/{predict_api_path_inf}" -d @{temp_payload_filepath_inf}')
                    progress_bar_render_inf.progress(60, text="Sending API request...")
                    predict_res_process_inf = run_shell_command_inference(curl_cmd_predict_inf)
                    progress_bar_render_inf.progress(80, text="API response received.")

                    with col2_inf:
                        if predict_res_process_inf and predict_res_process_inf.returncode == 0 and predict_res_process_inf.stdout:
                            try:
                                pred_json_inf = json.loads(predict_res_process_inf.stdout)
                                st.subheader("🖼️ Generated Image(s)")
                                if "predictions" in pred_json_inf and pred_json_inf["predictions"]:
                                    for i_inf, p_item_inf in enumerate(pred_json_inf["predictions"]):
                                        img_b64_data_inf = p_item_inf.get("bytesBase64Encoded") or p_item_inf.get("image", {}).get("bytesBase64Encoded")
                                        if img_b64_data_inf:
                                            st.image(Image.open(io.BytesIO(base64.b64decode(img_b64_data_inf))), caption=f"Generated Image {i_inf+1}", use_column_width=True)
                                        else: st.error(f"No image data in prediction {i_inf+1}.")
                                    st.success("✅ Image generation successful!")
                                elif "error" in pred_json_inf:
                                    st.error(f"API Error: {pred_json_inf['error'].get('message', 'Unknown')}")
                                else: st.error("Unexpected API response structure.")
                                if st.checkbox("Show Full API Response", key="show_resp_inf_cb"): st.json(pred_json_inf) # Changed key
                            except json.JSONDecodeError: st.error("Failed to decode API JSON response.")
                        elif predict_res_process_inf:
                            st.error(f"API request failed (Code: {predict_res_process_inf.returncode}).")
                            if st.checkbox("Show Curl Error Details", key="show_curl_err_inf_cb"):  # Changed key
                                st.text("Stdout:"); st.code(predict_res_process_inf.stdout or "N/A")
                                st.text("Stderr:"); st.code(predict_res_process_inf.stderr or "N/A")
                finally:
                    if temp_payload_filepath_inf and os.path.exists(temp_payload_filepath_inf): os.remove(temp_payload_filepath_inf)
                progress_bar_render_inf.progress(100, text="Done.")
            finally:
                if temp_image_file_path_inf and os.path.exists(temp_image_file_path_inf): os.remove(temp_image_file_path_inf)
    else: 
        with col2_inf: st.info("🖼️ Generated image(s) will appear here.")

PAGES = {
    "Fine-Tuning Pipeline": render_finetuning_pipeline_page,
    "Imagen Inference": render_imagen_inference_page,
}
st.sidebar.title("🛠️ Imagen LoRA Tools")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="page_selector")
page = PAGES[selection]
page() 

st.sidebar.markdown("---")
st.sidebar.info("This app provides tools for Imagen LoRA fine-tuning and inference on Vertex AI.")
st.sidebar.markdown(f"**Active Project:** `{config.PROJECT_ID}`")
st.sidebar.markdown(f"**Active Location:** `{config.LOCATION}`")
