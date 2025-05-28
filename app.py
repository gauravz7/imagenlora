# app.py
import streamlit as st
import os
import json
import random
from itertools import product

import config
import utils 
import imagen4gen # New module for batch generation
import vertexaipredict # New module for Vertex AI prediction endpoint

st.set_page_config(
    page_title="Imagen LoRA Tools", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fine-tuning pipeline page remains in app.py for now, 
# but uses helpers from utils.py
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

    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'gcs_jsonl_uri' not in st.session_state: st.session_state.gcs_jsonl_uri = None
    if 'tuning_job_path' not in st.session_state: st.session_state.tuning_job_path = None
    if 'all_gcs_uris_from_upload' not in st.session_state: st.session_state.all_gcs_uris_from_upload = []
    if 'context_uris' not in st.session_state: st.session_state.context_uris = []
    if 'target_uris' not in st.session_state: st.session_state.target_uris = []
    if 'target_descriptions' not in st.session_state: st.session_state.target_descriptions = {} 
    if 'subject_name_for_jsonl' not in st.session_state: st.session_state.subject_name_for_jsonl = "SXR person" # Example default
    if 'body_type_input_for_jsonl' not in st.session_state: st.session_state.body_type_input_for_jsonl = ""
    if 'subject_pronoun_for_jsonl' not in st.session_state: st.session_state.subject_pronoun_for_jsonl = "She" # Example default


    with st.expander("Step 0: Setup & Authentication", expanded=True):
        if st.button("Initialize & Authenticate Clients", key="auth_button_finetune_main"): # Ensure unique key
            try:
                with st.spinner("Initializing..."):
                    utils.get_storage_client(); utils.initialize_ai_platform()
                st.session_state.authenticated = True
                utils.st_log("Clients initialized!", "success") # Use utils.st_log
            except Exception as e: utils.st_log(f"Error: {e}", "error") # Use utils.st_log
        if st.session_state.get('authenticated', False): st.success("✅ Clients Authenticated.")
        else: st.warning("⚠️ Clients not authenticated.")

    with st.expander("Step 1: Data Preparation", expanded=False):
        if not st.session_state.get('authenticated', False): st.warning("Authenticate in Step 0.")
        else:
            st.subheader("1.1 Upload Local Images to GCS")
            local_image_dir = st.text_input("Local Image Folder:", config.LOCAL_IMAGE_DIR, key="local_img_dir_ft_main")
            gcs_image_upload_path = f"gs://{effective_gcs_bucket_name}/{config.GCS_IMAGE_UPLOAD_PREFIX.strip('/')}"
            st.write(f"Images will be uploaded from: `{local_image_dir}` to GCS path: `{gcs_image_upload_path}/`")

            if st.button("Upload to GCS", key="upload_gcs_ft_main"):
                with st.spinner(f"Uploading images from {local_image_dir} to GCS..."):
                    uploaded_uris = []
                    MAX_IMAGE_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024 
                    images_in_local_dir_upload = [f for f in os.listdir(local_image_dir) if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)] if os.path.isdir(local_image_dir) else []
                    if not images_in_local_dir_upload: utils.st_log("No images found or directory invalid.", "warning")
                    else:
                        for filename in images_in_local_dir_upload:
                            local_file_path = os.path.join(local_image_dir, filename)
                            _, compressed = utils.compress_image_if_needed(local_file_path, max_size_bytes=MAX_IMAGE_UPLOAD_SIZE_BYTES)
                            gcs_blob_name = f"{config.GCS_IMAGE_UPLOAD_PREFIX.rstrip('/')}/{filename}"
                            gcs_uri = utils.upload_to_gcs(local_file_path, effective_gcs_bucket_name, gcs_blob_name)
                            if gcs_uri:
                                uploaded_uris.append(gcs_uri)
                                utils.st_log(f"Uploaded {'compressed ' if compressed else ''}{filename} to {gcs_uri}", "info")
                            else: utils.st_log(f"Failed to upload {filename}", "error")
                        if uploaded_uris:
                            st.session_state.all_gcs_uris_from_upload = uploaded_uris
                            utils.st_log(f"Successfully uploaded {len(uploaded_uris)} images.", "success")
                        else: utils.st_log("Image upload failed/no images uploaded.", "error")
            
            if st.checkbox("Alternatively, list images directly from GCS", key="list_gcs_direct_ft_main"):
                if st.button("List Images from GCS", key="list_gcs_button_ft_main"):
                    with st.spinner("Listing images..."):
                        gcs_uris = utils.list_gcs_files(effective_gcs_bucket_name, config.GCS_IMAGE_UPLOAD_PREFIX, config.VALID_IMAGE_EXTENSIONS)
                        if gcs_uris:
                            st.session_state.all_gcs_uris_from_upload = gcs_uris
                            utils.st_log(f"Found {len(gcs_uris)} images in GCS.", "success")
                        else: utils.st_log("No images found in GCS path.", "warning")
            if st.session_state.all_gcs_uris_from_upload: st.success(f"Working with {len(st.session_state.all_gcs_uris_from_upload)} GCS image URIs.")

            st.subheader("1.2 Select Context & Target Images")
            context_percentage_select = st.slider("Percentage for Context:", 1, 99, 30, 1, key="context_perc_slider_ft_select_main") / 100.0
            if st.button("Select Context/Target Images", key="select_images_button_ft_select_main", disabled=not st.session_state.all_gcs_uris_from_upload):
                with st.spinner("Selecting images..."):
                    all_uris_sel = list(st.session_state.all_gcs_uris_from_upload)
                    random.shuffle(all_uris_sel)
                    num_context_sel = int(len(all_uris_sel) * context_percentage_select)
                    if num_context_sel == 0 and len(all_uris_sel) > 0: num_context_sel = 1
                    if len(all_uris_sel) > 1 and num_context_sel >= len(all_uris_sel): num_context_sel = len(all_uris_sel) - 1
                    st.session_state.context_uris = all_uris_sel[:num_context_sel]
                    st.session_state.target_uris = all_uris_sel[num_context_sel:]
                    utils.st_log(f"Selected {len(st.session_state.context_uris)} context and {len(st.session_state.target_uris)} target images.", "success")

            st.subheader("1.3 Generate Description Suffixes for Target Images")
            st.session_state.subject_name_for_jsonl = st.text_input("Subject Name (e.g., Sandra):", value=st.session_state.subject_name_for_jsonl, key="subject_name_input_ft_key_ui_main")
            st.session_state.body_type_input_for_jsonl = st.text_input("Optional: Body Type/Shape (e.g., curvy body type):", value=st.session_state.body_type_input_for_jsonl, key="body_type_input_ft_val_key_ui_main")
            pronoun_options = ["She", "He", "They", "It"]
            try:
                pronoun_default_index = pronoun_options.index(st.session_state.subject_pronoun_for_jsonl)
            except ValueError:
                pronoun_default_index = 0 
                st.session_state.subject_pronoun_for_jsonl = pronoun_options[0]
            st.session_state.subject_pronoun_for_jsonl = st.selectbox("Subject Pronoun:", pronoun_options, index=pronoun_default_index, key="pronoun_input_ft_key_ui_main")
            
            if st.button("Generate Description Suffixes", key="gen_desc_suffixes_ft_main", disabled=not st.session_state.target_uris or not st.session_state.subject_name_for_jsonl.strip()):
                with st.spinner("Generating suffixes..."):
                    desc_map_suffixes = {}
                    prog_bar = st.progress(0)
                    stat_text = st.empty()
                    for i, target_gcs_url in enumerate(st.session_state.target_uris):
                        stat_text.text(f"Processing image {i+1}/{len(st.session_state.target_uris)} for suffix...")
                        description_suffix = utils.generate_description_vertex_stream_from_gcs_uri(
                            gcs_file_uri=target_gcs_url,
                            subject_name=st.session_state.subject_name_for_jsonl 
                        )
                        desc_map_suffixes[target_gcs_url] = description_suffix
                        log_lvl = "warning" if "[Error:" in description_suffix or "[Warning:" in description_suffix else "info"
                        utils.st_log(f"Suffix for {os.path.basename(target_gcs_url)}: {description_suffix[:100]}...", log_lvl)
                        prog_bar.progress((i + 1) / len(st.session_state.target_uris))
                    st.session_state.target_descriptions = desc_map_suffixes
                    utils.st_log(f"Generated suffixes for {len(desc_map_suffixes)} images.", "success")
                    stat_text.text("Suffix generation complete!")
            
            st.subheader("1.4 Create Training JSONL File")
            max_pairs_jsonl = st.text_input("Max Training Pairs (0 for no limit):", "1000", key="max_pairs_jsonl_ft_main")
            user_context_id = st.text_input("Context ID for prompts (e.g., 1):", "1", key="user_context_id_jsonl_ft_main").strip() or "1"
            
            current_subject_name_for_jsonl = st.session_state.subject_name_for_jsonl
            current_body_type_for_jsonl = st.session_state.body_type_input_for_jsonl
            current_pronoun_for_jsonl = st.session_state.subject_pronoun_for_jsonl

            if st.button("Create & Upload JSONL", key="create_jsonl_ft_btn_main", disabled=not st.session_state.context_uris or not st.session_state.target_descriptions or not current_subject_name_for_jsonl.strip()):
                with st.spinner("Creating JSONL..."):
                    valid_suffixes = {uri: sfx for uri, sfx in st.session_state.target_descriptions.items() if not (sfx.startswith("[Error:") or sfx.startswith("[Warning:"))}
                    if not valid_suffixes: utils.st_log("No valid suffixes for JSONL.", "error")
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
                            prompt_part2 = f" {current_pronoun_for_jsonl} is {cleaned_suffix}" 
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
                                utils.st_log(f"Uploaded JSONL to: {gcs_uri_upload}", "success")
                            else: utils.st_log("Failed to upload JSONL.", "error")
                        else: utils.st_log("No JSONL data generated.", "error")
    
    with st.expander("Step 2: Train Model", expanded=False):
        if not st.session_state.gcs_jsonl_uri: st.warning("Complete Data Prep (Step 1) first.")
        else:
            st.success(f"Using training JSONL: {st.session_state.gcs_jsonl_uri}")
            st.subheader("Fine-Tuning Parameters")
            tuned_model_name = st.text_input("Tuned Model Display Name:", config.DEFAULT_TUNED_MODEL_DISPLAY_NAME, key="tuned_name_ft_train_main")
            adapter_options = ["ADAPTER_SIZE_ONE", "ADAPTER_SIZE_FOUR", "ADAPTER_SIZE_EIGHT", "ADAPTER_SIZE_SIXTEEN", "ADAPTER_SIZE_THIRTY_TWO"]
            adapter_size = st.selectbox("Adapter Size:", adapter_options, index=adapter_options.index(config.DEFAULT_ADAPTER_SIZE), key="adapter_size_ft_train_main")
            epochs = st.number_input("Epochs:", min_value=1, value=config.DEFAULT_EPOCHS, key="epochs_input_ft_train_main")
            lr_multiplier = st.number_input("Learning Rate Multiplier:", min_value=0.001, value=config.DEFAULT_LEARNING_RATE_MULTIPLIER, format="%.3f", key="lr_input_ft_train_main")

            if st.button("Submit Fine-Tuning Job", key="submit_train_button_ft_train_main"):
                with st.spinner("Submitting fine-tuning job..."):
                    job_path = utils.create_imagen_tuning_job(
                        project_id=config.PROJECT_ID, location=config.LOCATION,
                        base_model_uri=config.IMAGEN_BASE_MODEL, tuned_model_display_name=tuned_model_name,
                        training_dataset_uri=st.session_state.gcs_jsonl_uri, epochs=epochs,
                        adapter_size=adapter_size, learning_rate_multiplier=lr_multiplier
                    )
                    if job_path:
                        st.session_state.tuning_job_path = job_path
                        utils.st_log(f"Tuning Job Submitted: {job_path}", "success")
                    else: utils.st_log("Failed to submit tuning job.", "error")

    with st.expander("Step 3: Monitor Fine-Tuning Job", expanded=False):
        manual_job_path_input_mon = st.text_input("Or, enter Tuning Job Path manually:", key="manual_job_path_monitor_ft_mon_main")
        current_job_to_monitor_mon = manual_job_path_input_mon if manual_job_path_input_mon else st.session_state.get('tuning_job_path')

        if not current_job_to_monitor_mon: st.info("No tuning job to monitor.")
        else:
            st.success(f"Monitoring job: {current_job_to_monitor_mon}")
            if st.button("Refresh Job Status", key="refresh_monitor_button_ft_mon_main"):
                with st.spinner("Fetching job status..."):
                    status_data = utils.monitor_tuning_job(job_name_path=current_job_to_monitor_mon, location=config.LOCATION)
                    if status_data:
                        utils.st_log("Job Status Refreshed.", "info")
                        st.json(status_data)
                        job_state = status_data.get("state", "JOB_STATE_UNSPECIFIED")
                        if job_state == "JOB_STATE_SUCCEEDED":
                            st.balloons(); utils.st_log("SUCCESS: Tuning job completed!", "success")
                            endpoint = status_data.get("tunedModel", {}).get("endpoint")
                            if endpoint: utils.st_log(f"Tuned Model Endpoint: {endpoint}", "success")
                        elif job_state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                            utils.st_log(f"ERROR: Job status: {job_state}.", "error")
                        else: utils.st_log(f"INFO: Job status: {job_state}.", "info")
                    else: utils.st_log("Failed to retrieve job status.", "error")

# Main application logic
PAGES = {
    "Fine-Tuning Pipeline": render_finetuning_pipeline_page,
    "Imagen Inference (Endpoint)": vertexaipredict.render_imagen_inference_page,
    "Batch Generation & Collage (SDK)": imagen4gen.render_batch_generation_page,
}

st.sidebar.title("🛠️ Imagen LoRA Tools")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="page_selector_main")
page_function = PAGES[selection]
page_function() 

st.sidebar.markdown("---")
st.sidebar.info("This app provides tools for Imagen LoRA fine-tuning and inference on Vertex AI.")
st.sidebar.markdown(f"**Active Project:** `{config.PROJECT_ID}`")
st.sidebar.markdown(f"**Active Location:** `{config.LOCATION}`")
