import streamlit as st
import os
import time
import json
import math
import tempfile
from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel
import vertexai

import config
import utils # For st_log, run_shell_command_inference, upload_to_gcs

# Helper function to create a collage from a list of image paths
def create_collage(image_paths: list, thumb_width: int = 256, thumb_height: int = 256, cols: int = 0, spacing: int = 10, bgcolor='white'):
    if not image_paths:
        return None

    images_pil = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images_pil.append(img)
        except Exception as e:
            utils.st_log(f"Error opening image {p} for collage: {e}", "warning")
            continue
    
    if not images_pil:
        utils.st_log("No valid images to create collage.", "warning")
        return None

    # Resize to thumbnails
    thumbnails = []
    for img_pil in images_pil: # Renamed img to img_pil to avoid conflict with outer scope
        img_pil.thumbnail((thumb_width, thumb_height))
        thumbnails.append(img_pil)

    num_images = len(thumbnails)
    if cols <= 0: # Auto-calculate columns
        cols = math.ceil(math.sqrt(num_images))
        if cols == 0: cols = 1 # Ensure at least 1 column
    rows = math.ceil(num_images / cols)
    
    collage_width = cols * thumb_width + (cols + 1) * spacing 
    collage_height = rows * thumb_height + (rows + 1) * spacing
    
    collage_image = Image.new('RGB', (collage_width, collage_height), color=bgcolor)
    
    for i, thumb in enumerate(thumbnails):
        row_idx = i // cols
        col_idx = i % cols
        
        x_offset = spacing + col_idx * (thumb_width + spacing)
        y_offset = spacing + row_idx * (thumb_height + spacing)
        collage_image.paste(thumb, (x_offset, y_offset))
        
    return collage_image

# Helper function to download a GCS file to a temporary local path
def download_gcs_file_to_temp(gcs_uri: str, suffix=".png") -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_dl_file:
        temp_downloaded_path = tmp_dl_file.name
    
    gsutil_cp_command = f"gsutil cp '{gcs_uri}' '{temp_downloaded_path}'"
    cp_result = utils.run_shell_command_inference(gsutil_cp_command)
    
    if cp_result.returncode != 0:
        if os.path.exists(temp_downloaded_path):
            os.remove(temp_downloaded_path) 
        raise RuntimeError(f"Failed to download GCS image: {gcs_uri}. gsutil stderr: {cp_result.stderr}")
    
    return temp_downloaded_path


def render_batch_generation_page():
    st.title("🏭 Batch Image Generation & Collage")

    default_gcs_bucket_batch = config.GCS_BUCKET_NAME
    if 'user_gcs_bucket_name_batch' not in st.session_state:
        st.session_state.user_gcs_bucket_name_batch = default_gcs_bucket_batch
    
    user_gcs_bucket_input_batch = st.sidebar.text_input(
        "GCS Bucket Name (Batch Outputs)", 
        value=st.session_state.user_gcs_bucket_name_batch, 
        key="gcs_bucket_name_batch_widget",
        help="GCS bucket for storing generated images and collage."
    )
    if user_gcs_bucket_input_batch:
        st.session_state.user_gcs_bucket_name_batch = user_gcs_bucket_input_batch
    
    effective_gcs_bucket_batch = st.session_state.user_gcs_bucket_name_batch or default_gcs_bucket_batch

    st.caption(f"Project: {config.PROJECT_ID}, Location: {config.LOCATION}, Output Bucket: {effective_gcs_bucket_batch}")

    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if not st.session_state.get('authenticated', False):
        st.warning("⚠️ Clients not authenticated. Please authenticate on the Fine-Tuning page (Step 0) first or click below.")
        if st.button("Attempt to Initialize & Authenticate Clients", key="auth_button_batch_page"):
            try:
                with st.spinner("Initializing..."):
                    utils.get_storage_client(); utils.initialize_ai_platform() 
                st.session_state.authenticated = True
                utils.st_log("Clients initialized!", "success")
                st.rerun()
            except Exception as e: utils.st_log(f"Error: {e}", "error")
        return

    st.subheader("📝 Input Prompts")
    prompts_input = st.text_area("Enter prompts (one per line):", height=200, 
                                 placeholder="A futuristic cityscape at sunset\nA serene forest with a hidden waterfall\nA cat wearing a tiny wizard hat")

    st.subheader("⚙️ Generation Parameters")
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        aspect_ratio_options_batch = {"Portrait (9:16)": "9:16", "Landscape (16:9)": "16:9", "Square (1:1)": "1:1"}
        selected_aspect_display_batch = st.selectbox("Aspect Ratio", list(aspect_ratio_options_batch.keys()), index=0, key="batch_aspect_ratio")
        aspect_ratio_batch = aspect_ratio_options_batch[selected_aspect_display_batch]
        
        num_images_per_prompt = st.number_input("Images per prompt", min_value=1, max_value=4, value=1, key="batch_num_images", help="Currently, only the first image from each prompt is used in the collage.")

    with col_param2:
        negative_prompt_batch = st.text_input("Negative Prompt", "blurry, low quality, text, watermark", key="batch_neg_prompt")

    if st.button("🚀 Generate Images & Create Collage", type="primary", use_container_width=True, key="batch_generate_button"):
        prompts = [p.strip() for p in prompts_input.split('\n') if p.strip()]
        if not prompts:
            st.error("🚫 Please enter at least one prompt.")
            return
        if not effective_gcs_bucket_batch:
            st.error("🚫 Please configure the GCS Bucket Name in the sidebar.")
            return

        if 'batch_generation_model' not in st.session_state:
            with st.spinner("Initializing Vertex AI Imagen Model..."):
                try:
                    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
                    st.session_state.batch_generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-05-20")
                    utils.st_log("Vertex AI Imagen Model initialized.", "success")
                except Exception as e:
                    utils.st_log(f"Error initializing Vertex AI Model: {e}", "error")
                    return 
        
        generation_model = st.session_state.batch_generation_model
        
        gcs_output_base_folder = "batch_image_generations"
        timestamp_folder = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        gcs_run_output_path_prefix = f"{gcs_output_base_folder}/{timestamp_folder}" 

        generated_image_gcs_uris = []
        
        st.subheader("🖼️ Generated Images")
        generation_progress_bar = st.progress(0)

        for i, prompt_text in enumerate(prompts):
            st.markdown(f"--- \n**Processing Prompt {i+1}/{len(prompts)}:** `{prompt_text}`")
            try:
                with st.spinner(f"Generating image for: \"{prompt_text[:50]}...\""):
                    images_generated = generation_model.generate_images(
                        prompt=prompt_text,
                        number_of_images=num_images_per_prompt,
                        aspect_ratio=aspect_ratio_batch,
                        negative_prompt=negative_prompt_batch,
                        add_watermark=True 
                    )
                
                if images_generated and images_generated.images:
                    pil_image_to_save = images_generated.images[0]._pil_image 
                    
                    st.image(pil_image_to_save, caption=f"Generated for: \"{prompt_text[:50]}...\"", use_column_width=False, width=300)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
                        pil_image_to_save.save(tmp_img_file.name, format="PNG")
                        temp_local_path = tmp_img_file.name
                    
                    img_filename = f"prompt_{i+1}_image_0.png"
                    gcs_blob_name = f"{gcs_run_output_path_prefix}/{img_filename}"
                    
                    gcs_uri = utils.upload_to_gcs(temp_local_path, effective_gcs_bucket_batch, gcs_blob_name)
                    
                    if gcs_uri:
                        utils.st_log(f"Image for prompt {i+1} uploaded to: {gcs_uri}", "success")
                        generated_image_gcs_uris.append(gcs_uri)
                    else:
                        utils.st_log(f"Failed to upload image for prompt {i+1} to GCS.", "error")
                    
                    os.remove(temp_local_path) 

                else:
                    utils.st_log(f"No image returned for prompt: \"{prompt_text}\". It might have been filtered.", "warning")

            except Exception as e:
                utils.st_log(f"Error generating image for prompt \"{prompt_text}\": {e}", "error")
            
            generation_progress_bar.progress((i + 1) / len(prompts))

        if generated_image_gcs_uris:
            st.markdown("--- \n### 🖼️ Image Collage")
            with st.spinner("Downloading images and creating collage..."):
                downloaded_temp_paths_for_collage = []
                try:
                    for gcs_uri_item in generated_image_gcs_uris: # Renamed gcs_uri to avoid conflict
                        try:
                            local_path = download_gcs_file_to_temp(gcs_uri_item)
                            downloaded_temp_paths_for_collage.append(local_path)
                        except Exception as e:
                            utils.st_log(f"Failed to download {gcs_uri_item} for collage: {e}", "error")
                    
                    if downloaded_temp_paths_for_collage:
                        collage_img = create_collage(downloaded_temp_paths_for_collage, thumb_width=256, thumb_height=256, cols=0) 
                        if collage_img:
                            st.image(collage_img, caption="Generated Image Collage", use_column_width=True)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_collage_file:
                                collage_img.save(tmp_collage_file.name, format="PNG")
                                collage_gcs_blob_name = f"{gcs_run_output_path_prefix}/collage.png"
                                collage_gcs_uri_upload = utils.upload_to_gcs(tmp_collage_file.name, effective_gcs_bucket_batch, collage_gcs_blob_name) # Renamed gcs_uri
                                if collage_gcs_uri_upload:
                                    utils.st_log(f"Collage uploaded to: {collage_gcs_uri_upload}", "success")
                                os.remove(tmp_collage_file.name)
                        else:
                            utils.st_log("Failed to create collage.", "error")
                    else:
                        utils.st_log("No images were successfully downloaded for the collage.", "warning")
                finally:
                    for path in downloaded_temp_paths_for_collage:
                        if os.path.exists(path):
                            os.remove(path)
            st.success(f"All outputs for this run (if any) are in GCS path: gs://{effective_gcs_bucket_batch}/{gcs_run_output_path_prefix}/")
        else:
            utils.st_log("No images were generated to create a collage.", "info")
