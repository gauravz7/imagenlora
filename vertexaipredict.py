import streamlit as st
import os
import time
import json
import base64
import io
import tempfile
from PIL import Image

import config
import utils # For st_log, run_shell_command_inference, upload_to_gcs

def get_image_bytes_for_vertex_imagen_api_inference(image_path_or_gcs_uri, target_size: tuple = (1024, 1024)):
    local_image_path = image_path_or_gcs_uri
    temp_downloaded_path = None
    if image_path_or_gcs_uri.startswith("gs://"):
        utils.st_log("GCS URI detected for context image. Ensure gsutil is configured.", "warning") # Changed to utils.st_log
        base_name = os.path.basename(image_path_or_gcs_uri)
        with tempfile.NamedTemporaryFile(delete=False, prefix="vertex_img_dl_", suffix=os.path.splitext(base_name)[1]) as tmp_dl_file:
            temp_downloaded_path = tmp_dl_file.name
        gsutil_cp_command = f"gsutil cp '{image_path_or_gcs_uri}' '{temp_downloaded_path}'"
        cp_result = utils.run_shell_command_inference(gsutil_cp_command) # Changed to utils.run_shell_command_inference
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
                except AttributeError: img = img.resize(target_size, Image.LANCZOS) # Fallback for older PIL
            output_io = io.BytesIO()
            img.save(output_io, format='PNG')
            image_bytes = output_io.getvalue()
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.remove(temp_downloaded_path)
    return image_bytes, 'image/png'


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
        st.info(f"Use `{subject_reference_placeholder_inf}` in your prompt to refer to the subject. Enter one prompt per line for multiple generations.")
        predict_prompts_input_inf = st.text_area("📝 Prediction Prompts (one per line)", value=default_full_prompt_inf, height=150, key="inf_prompts")

    st.sidebar.header("✨ Prediction Parameters")
    negative_prompt_inf = st.sidebar.text_input("Negative Prompt", DEFAULT_NEGATIVE_PROMPT_INF, key="inf_neg_prompt")
    seed_inf = st.sidebar.number_input("Seed (0 for random)", value=DEFAULT_SEED_INF, min_value=0, key="inf_seed")
    sample_count_inf = st.sidebar.number_input("Number of Images", min_value=1, max_value=4, value=DEFAULT_SAMPLE_COUNT_INF, key="inf_sample_count")
    
    aspect_ratio_options_inf = {"Portrait (9:16)": "9:16", "Landscape (16:9)": "16:9", "Square (1:1)": "1:1"}
    default_aspect_key_inf = [k for k,v in aspect_ratio_options_inf.items() if v == DEFAULT_ASPECT_RATIO_INF][0]
    selected_aspect_display_inf = st.sidebar.selectbox("Aspect Ratio", list(aspect_ratio_options_inf.keys()), index=list(aspect_ratio_options_inf.keys()).index(default_aspect_key_inf), key="inf_aspect_ratio")
    aspect_ratio_inf = aspect_ratio_options_inf[selected_aspect_display_inf] 

    if st.button("🎨 Generate Images", type="primary", use_container_width=True, key="inf_generate_button"): 
        prompts_to_run = [p.strip() for p in predict_prompts_input_inf.split('\n') if p.strip()]
        if not all([PROJECT_ID_INF, LOCATION_INF, PREDICT_ENDPOINT_ID_INF, CONTEXT_ID_FOR_PREDICTION_INF, SUBJECT_DESC_FOR_CONTEXT_INF, uploaded_context_image_inf, prompts_to_run]): 
            st.error("🚫 Please fill in all configuration details, upload a context image, and enter at least one prompt.")
        else:
            with col2_inf:
                st.subheader("⏳ Processing...")
                overall_progress_bar = st.progress(0, text="Initializing...")
            
            temp_context_image_file_path = None
            current_gcs_bucket_for_inference = st.session_state.get('user_gcs_bucket_name_ft', config.GCS_BUCKET_NAME) # Assuming ft bucket is okay for outputs
            gcs_output_base_folder = f"gs://{current_gcs_bucket_for_inference}/imagen_inference_outputs"
            timestamp_folder = f"output-{time.strftime('%Y%m%d-%H%M%S')}"
            gcs_output_path_for_run = f"{gcs_output_base_folder}/{timestamp_folder}"
            all_generated_image_details = []

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_context_image_inf.name)[1]) as tmp_file: 
                    tmp_file.write(uploaded_context_image_inf.getvalue())
                    temp_context_image_file_path = tmp_file.name
                overall_progress_bar.progress(5, text="Context image saved locally.")

                context_img_bytes_inf, context_img_mime_type_inf = get_image_bytes_for_vertex_imagen_api_inference(temp_context_image_file_path)
                context_img_b64_inf = base64.b64encode(context_img_bytes_inf).decode('utf-8')
                overall_progress_bar.progress(10, text="Context image processed for API.")

                VERTEX_API_EP_RENDER_INF = f"https://{LOCATION_INF}-aiplatform.googleapis.com"
                predict_api_path_inf = f"v1/projects/{PROJECT_ID_INF}/locations/{LOCATION_INF}/endpoints/{PREDICT_ENDPOINT_ID_INF}:predict"
                
                try: context_id_int_inf = int(CONTEXT_ID_FOR_PREDICTION_INF)
                except ValueError: 
                    st.error(f"Context ID '{CONTEXT_ID_FOR_PREDICTION_INF}' must be an integer."); overall_progress_bar.empty(); st.stop()

                for idx, current_prompt in enumerate(prompts_to_run):
                    with col2_inf:
                        st.markdown(f"--- \n#### Generating for Prompt {idx+1}/{len(prompts_to_run)}: \n`{current_prompt}`")
                        prompt_progress_bar = st.progress(0, text=f"Prompt {idx+1}: Preparing payload...")

                    predict_payload_inf = {
                        'instances': [{'prompt': current_prompt, 'referenceImages': [{'referenceType': 'REFERENCE_TYPE_SUBJECT', 'referenceId': context_id_int_inf, 'referenceImage': {'bytesBase64Encoded': context_img_b64_inf, 'mimeType': context_img_mime_type_inf}, 'subjectImageConfig': {'subjectDescription': SUBJECT_DESC_FOR_CONTEXT_INF, 'subjectType': 'SUBJECT_TYPE_PERSON'},}]}],
                        'parameters': {'negativePrompt': negative_prompt_inf, 'sampleCount': sample_count_inf, 'aspectRatio': aspect_ratio_inf}
                    }
                    if seed_inf > 0: predict_payload_inf['parameters']['seed'] = seed_inf
                        
                    prompt_progress_bar.progress(25, text=f"Prompt {idx+1}: Payload ready.")

                    temp_payload_filepath_inf = None
                    predict_res_process_inf = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_payload_file_inf:
                            json.dump(predict_payload_inf, tmp_payload_file_inf)
                            temp_payload_filepath_inf = tmp_payload_file_inf.name
                        
                        curl_cmd_predict_inf = (f'curl -s -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json; charset=utf-8" "{VERTEX_API_EP_RENDER_INF}/{predict_api_path_inf}" -d @{temp_payload_filepath_inf}')
                        prompt_progress_bar.progress(50, text=f"Prompt {idx+1}: Sending API request...")
                        predict_res_process_inf = utils.run_shell_command_inference(curl_cmd_predict_inf) # Changed to utils.run_shell_command_inference
                        prompt_progress_bar.progress(75, text=f"Prompt {idx+1}: API response received.")

                        with col2_inf:
                            if predict_res_process_inf and predict_res_process_inf.returncode == 0 and predict_res_process_inf.stdout:
                                try:
                                    pred_json_inf = json.loads(predict_res_process_inf.stdout)
                                    if "predictions" in pred_json_inf and pred_json_inf["predictions"]:
                                        for i_sample, p_item_inf in enumerate(pred_json_inf["predictions"]):
                                            img_b64_data_inf = p_item_inf.get("bytesBase64Encoded") or p_item_inf.get("image", {}).get("bytesBase64Encoded")
                                            if img_b64_data_inf:
                                                img_data_bytes = base64.b64decode(img_b64_data_inf)
                                                pil_img_inf = Image.open(io.BytesIO(img_data_bytes))
                                                st.image(pil_img_inf, caption=f"Prompt {idx+1} - Image {i_sample+1}", use_column_width=True)
                                                
                                                img_filename = f"prompt_{idx+1}_sample_{i_sample+1}.png"
                                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_gen_img_file:
                                                    pil_img_inf.save(tmp_gen_img_file, format="PNG")
                                                    tmp_gen_img_path = tmp_gen_img_file.name
                                                
                                                gcs_img_blob_name = f"{gcs_output_path_for_run.split(f'gs://{current_gcs_bucket_for_inference}/')[-1]}/{img_filename}"
                                                gcs_img_uri = utils.upload_to_gcs(tmp_gen_img_path, current_gcs_bucket_for_inference, gcs_img_blob_name)
                                                os.remove(tmp_gen_img_path) 

                                                if gcs_img_uri:
                                                    utils.st_log(f"Saved generated image to {gcs_img_uri}", "success") # Changed to utils.st_log
                                                    all_generated_image_details.append({"prompt": current_prompt, "gcs_uri": gcs_img_uri, "parameters": predict_payload_inf['parameters']})
                                                else:
                                                    utils.st_log(f"Failed to save generated image {img_filename} to GCS.", "error") # Changed to utils.st_log

                                            else: st.error(f"No image data in prediction {i_sample+1} for prompt {idx+1}.")
                                        st.success(f"✅ Prompt {idx+1} processed.")
                                    elif "error" in pred_json_inf:
                                        st.error(f"API Error for prompt {idx+1}: {pred_json_inf['error'].get('message', 'Unknown')}")
                                    else: st.error(f"Unexpected API response for prompt {idx+1}.")
                                except json.JSONDecodeError: st.error(f"Failed to decode API JSON for prompt {idx+1}.")
                            elif predict_res_process_inf:
                                st.error(f"API request failed for prompt {idx+1} (Code: {predict_res_process_inf.returncode}).")
                    finally:
                        if temp_payload_filepath_inf and os.path.exists(temp_payload_filepath_inf): os.remove(temp_payload_filepath_inf)
                    prompt_progress_bar.progress(100, text=f"Prompt {idx+1}: Done.")
                    overall_progress_bar.progress(10 + int(85 * ((idx + 1) / len(prompts_to_run)))) 
                
                if all_generated_image_details:
                    manifest_content = json.dumps(all_generated_image_details, indent=2)
                    manifest_filename = "generation_manifest.json"
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_manifest_file:
                        tmp_manifest_file.write(manifest_content)
                        tmp_manifest_path = tmp_manifest_file.name
                    
                    gcs_manifest_blob_name = f"{gcs_output_path_for_run.split(f'gs://{current_gcs_bucket_for_inference}/')[-1]}/{manifest_filename}"
                    gcs_manifest_uri = utils.upload_to_gcs(tmp_manifest_path, current_gcs_bucket_for_inference, gcs_manifest_blob_name)
                    os.remove(tmp_manifest_path)
                    if gcs_manifest_uri:
                        utils.st_log(f"Saved generation manifest to {gcs_manifest_uri}", "success") # Changed to utils.st_log
                        st.markdown(f"**All outputs saved to: `{gcs_output_path_for_run}/`**")

            finally:
                if temp_context_image_file_path and os.path.exists(temp_context_image_file_path):
                    os.remove(temp_context_image_file_path)
            overall_progress_bar.progress(100, text="All processing complete.")
    else: 
        with col2_inf: st.info("🖼️ Generated image(s) will appear here.")
