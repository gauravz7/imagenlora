# data_preparation.py
import os
import random
import json
from typing import List, Tuple, Dict, Optional, Any
from itertools import product

import config
import utils

def upload_local_images_to_gcs() -> List[str]:
    """
    Uploads images from the local directory specified in config.LOCAL_IMAGE_DIR
    to the GCS path specified by config.GCS_BUCKET_NAME and config.GCS_IMAGE_UPLOAD_PREFIX.
    Returns a list of GCS URIs of the uploaded images.
    """
    print(f"--- Step 0: Uploading local images to GCS ---")
    if not os.path.isdir(config.LOCAL_IMAGE_DIR):
        print(f"Error: Local image directory not found: {config.LOCAL_IMAGE_DIR}")
        return []

    uploaded_gcs_uris = []
    print(f"Scanning local directory: {config.LOCAL_IMAGE_DIR} for images...")
    for filename in os.listdir(config.LOCAL_IMAGE_DIR):
        if filename.lower().endswith(config.VALID_IMAGE_EXTENSIONS):
            local_file_path = os.path.join(config.LOCAL_IMAGE_DIR, filename)
            # Sanitize filename for GCS (e.g., replace spaces, though GCS handles many chars)
            gcs_blob_name = f"{config.GCS_IMAGE_UPLOAD_PREFIX.rstrip('/')}/{filename}"
            
            gcs_uri = utils.upload_to_gcs(local_file_path, config.GCS_BUCKET_NAME, gcs_blob_name)
            if gcs_uri:
                uploaded_gcs_uris.append(gcs_uri)
        else:
            print(f"Skipping non-image file: {filename}")
            
    print(f"Uploaded {len(uploaded_gcs_uris)} images to {config.get_gcs_image_upload_path()}/")
    return uploaded_gcs_uris

def select_context_and_target_images(
    all_gcs_image_uris: List[str],
    context_percentage: float = 0.3
) -> Tuple[List[str], List[str]]:
    """
    Randomly selects context and target images from a list of GCS URIs.
    Args:
        all_gcs_image_uris: List of all available image GCS URIs.
        context_percentage: Percentage of images to be used as context images (e.g., 0.3 for 30%).
    Returns:
        A tuple containing (list_of_context_image_uris, list_of_target_image_uris).
    """
    print(f"\n--- Step 1: Selecting Context and Target Images ---")
    if not all_gcs_image_uris:
        print("No GCS image URIs provided for selection.")
        return [], []

    random.shuffle(all_gcs_image_uris) # Shuffle for random selection
    
    num_context_images = int(len(all_gcs_image_uris) * context_percentage)
    if num_context_images == 0 and len(all_gcs_image_uris) > 0: # Ensure at least one context image if possible
        num_context_images = 1
    
    # Ensure there's at least one target image if we have more than one image overall
    if len(all_gcs_image_uris) > 1 and num_context_images >= len(all_gcs_image_uris):
        num_context_images = len(all_gcs_image_uris) - 1

    context_uris = all_gcs_image_uris[:num_context_images]
    target_uris = all_gcs_image_uris[num_context_images:]

    print(f"Total images available: {len(all_gcs_image_uris)}")
    print(f"Selected {len(context_uris)} context images ({context_percentage*100:.1f}%).")
    print(f"Selected {len(target_uris)} target images.")
    
    if not context_uris: print("Warning: No context images were selected. Check total image count and percentage.")
    if not target_uris: print("Warning: No target images were selected. Check total image count and percentage.")
    
    return context_uris, target_uris

def generate_descriptions_for_targets(
    target_image_uris: List[str],
    subject_desc_for_context: str
) -> Dict[str, str]:
    """
    Generates descriptions for each target image using Gemini.
    Args:
        target_image_uris: List of GCS URIs for target images.
        subject_desc_for_context: The user-provided subject description (e.g., "sandra", "my dog").
    Returns:
        A dictionary mapping target GCS URI to its generated description.
    """
    print(f"\n--- Step 2: Generating Descriptions for Target Images ---")
    descriptions_map = {}
    # This prompt template is similar to the one in the notebook.
    # {subject_name} will be replaced by subject_desc_for_context.
    desc_prompt_template = (
        f"Generate a single descriptive sentence or short paragraph about {{subject_name}}, "
        f"a person or subject, detailing their action, clothing (if applicable), and the background "
        f"setting of this image for an image generation task. Avoid section headings or lists."
    )

    for i, target_gcs_uri in enumerate(target_image_uris):
        print(f"\nProcessing target image {i+1}/{len(target_image_uris)}: {target_gcs_uri}")

        # Pre-check image size (optional, but good practice from notebook)
        original_file_size = utils.get_gcs_blob_size(target_gcs_uri)
        if original_file_size is not None and original_file_size > config.MAX_IMAGE_FILE_SIZE_BYTES_FOR_GEMINI:
            print(f"WARNING: Original file size ({original_file_size} bytes) for {target_gcs_uri} "
                  f"exceeds max ({config.MAX_IMAGE_FILE_SIZE_BYTES_FOR_GEMINI} bytes). Skipping description.")
            descriptions_map[target_gcs_uri] = f"A photo of {subject_desc_for_context}. [Error: Image file too large for processing]"
            continue
        elif original_file_size is None:
             print(f"WARNING: Could not determine original file size for {target_gcs_uri}. Proceeding with caution.")


        scene_desc = utils.generate_image_description_with_gemini(
            gcs_image_uri=target_gcs_uri,
            prompt_text_template=desc_prompt_template,
            subject_name=subject_desc_for_context
        )
        descriptions_map[target_gcs_uri] = scene_desc
        if "[Error:" in scene_desc:
            print(f"Skipping {target_gcs_uri} for JSONL due to description error.")
            
    print(f"\nGenerated descriptions for {sum(1 for desc in descriptions_map.values() if '[Error:' not in desc)} target images.")
    return descriptions_map

def create_training_jsonl(
    context_image_uris: List[str],
    target_image_descriptions: Dict[str, str], # Map of target_uri to description
    subject_desc_for_context: str,
    max_pairs: Optional[int] = None
) -> Optional[str]:
    """
    Creates the train.jsonl file by pairing context images with target images and their descriptions.
    Args:
        context_image_uris: List of GCS URIs for context images.
        target_image_descriptions: Dict mapping target GCS URI to its description.
        subject_desc_for_context: The user-provided subject description.
        max_pairs: Optional limit for the number of training pairs in the JSONL.
    Returns:
        The GCS URI of the uploaded JSONL file, or None if an error occurs.
    """
    print(f"\n--- Step 3: Creating Training JSONL File ---")
    if not context_image_uris:
        print("Error: No context images provided. Cannot create training JSONL.")
        return None
    if not target_image_descriptions:
        print("Error: No target image descriptions provided. Cannot create training JSONL.")
        return None

    jsonl_lines = []
    context_id_counter = 1 # To give unique context IDs if needed, though notebook uses one primary.
                           # For "all pairs", each context image is distinct.

    # Create all possible (context_uri, target_uri) pairs
    all_possible_pairs = list(product(context_image_uris, target_image_descriptions.keys()))
    
    print(f"Total possible (context_image, target_image) pairs: {len(all_possible_pairs)}")

    if max_pairs is not None and max_pairs < len(all_possible_pairs):
        print(f"Limiting to {max_pairs} randomly selected pairs.")
        selected_pairs = random.sample(all_possible_pairs, max_pairs)
    else:
        if max_pairs is not None:
             print(f"Requested max_pairs ({max_pairs}) is >= total possible pairs. Using all pairs.")
        selected_pairs = all_possible_pairs
    
    print(f"Generating JSONL entries for {len(selected_pairs)} pairs...")

    for context_gcs_uri, target_gcs_uri in selected_pairs:
        scene_desc = target_image_descriptions.get(target_gcs_uri)
        if not scene_desc or "[Error:" in scene_desc:
            print(f"Skipping pair ({context_gcs_uri}, {target_gcs_uri}) due to missing or error in target description.")
            continue

        # Use a unique ID for each context image in the pair for clarity in the JSONL
        # This context_id is specific to the JSONL entry.
        current_context_id_str = f"ctx_{os.path.basename(context_gcs_uri).split('.')[0]}"


        # Post-process scene_desc (similar to notebook)
        cleaned_scene_desc = scene_desc.strip()
        unwanted_prefixes = ["here's a detailed description of the image:", "action:", "clothing:", "background:"]
        for prefix in unwanted_prefixes:
            if cleaned_scene_desc.lower().startswith(prefix.lower()): # Case-insensitive prefix check
                cleaned_scene_desc = cleaned_scene_desc[len(prefix):].strip()
        cleaned_scene_desc = cleaned_scene_desc.strip('"').strip("'")
        cleaned_scene_desc = ' '.join(cleaned_scene_desc.splitlines())
        if cleaned_scene_desc and not cleaned_scene_desc.endswith(('.', '!', '?')):
            cleaned_scene_desc += '.'

        # Insert context ID into the prompt
        # The notebook inserts it after the subject name. We'll try to replicate that.
        subject_name_lower = subject_desc_for_context.lower()
        cleaned_desc_lower = cleaned_scene_desc.lower()
        insert_pos = cleaned_desc_lower.find(subject_name_lower)

        if insert_pos != -1:
            end_pos_subject_name = insert_pos + len(subject_name_lower)
            # Ensure we use the original casing for the parts of the description
            final_prompt = cleaned_scene_desc[:end_pos_subject_name] + f" [{current_context_id_str}]" + cleaned_scene_desc[end_pos_subject_name:]
        else:
            final_prompt = f"{cleaned_scene_desc} [{current_context_id_str}]"
            print(f"Warning: Subject name '{subject_desc_for_context}' not found in description for target {os.path.basename(target_gcs_uri)}. Appending [{current_context_id_str}] at the end.")

        # Determine mime types (simple approach based on extension)
        def get_mime_type_from_uri(uri: str) -> str:
            ext = os.path.splitext(uri)[1].lower().replace('.', '')
            return f"image/{ext if ext in ['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'] else 'jpeg'}"

        context_mime_type = get_mime_type_from_uri(context_gcs_uri)
        target_mime_type = get_mime_type_from_uri(target_gcs_uri)

        jsonl_entry_dict = {
            "prompt": final_prompt,
            "context_images": [{
                "context_id": current_context_id_str, # Use the generated ID
                "context_image": {"mime_type": context_mime_type, "file_uri": context_gcs_uri},
                "context_prompt": subject_desc_for_context # This is the general subject description
            }],
            "target_image": {"mime_type": target_mime_type, "file_uri": target_gcs_uri}
        }
        jsonl_lines.append(json.dumps(jsonl_entry_dict))
        # print(f"Added to JSONL. Prompt snippet: {final_prompt[:80]}...")

    if not jsonl_lines:
        print("Error: No training data generated for JSONL. Check image availability, descriptions, and pairing logic.")
        return None
    
    # Min 32 lines for Imagen tuning, but this can be a soft warning
    if len(jsonl_lines) < 32:
        print(f"Warning: Only {len(jsonl_lines)} examples generated. Imagen fine-tuning typically requires at least 32 examples.")

    local_jsonl_path = os.path.join(os.getcwd(), config.LOCAL_TRAIN_JSONL_FILENAME) # Save in CWD
    with open(local_jsonl_path, 'w') as f:
        f.write('\n'.join(jsonl_lines))
    print(f"\nSuccessfully wrote {len(jsonl_lines)} lines to local JSONL file: {local_jsonl_path}")

    # Upload to GCS
    gcs_jsonl_blob_name = f"{config.GCS_JSONL_PREFIX.rstrip('/')}/{config.LOCAL_TRAIN_JSONL_FILENAME}"
    gcs_jsonl_uri = utils.upload_to_gcs(local_jsonl_path, config.GCS_BUCKET_NAME, gcs_jsonl_blob_name)

    if gcs_jsonl_uri:
        print(f"Uploaded training JSONL to: {gcs_jsonl_uri}")
        # Optionally, print first few lines from GCS (requires download or cat, skip for now to simplify)
        # utils.run_shell_command(f"gsutil cat '{gcs_jsonl_uri}' | head -n 2") # If using shell
        return gcs_jsonl_uri
    else:
        print(f"Error: Failed to upload JSONL file to GCS.")
        return None


def main():
    print("Starting data preparation pipeline...")

    # --- User Inputs ---
    subject_desc_for_context = input("Enter the subject description for context images (e.g., 'sandra', 'my cat Tom'): ").strip()
    if not subject_desc_for_context:
        print("Subject description cannot be empty. Exiting.")
        return

    max_pairs_str = input("Enter the maximum number of training pairs for the JSONL (e.g., 1000, or leave blank for no limit): ").strip()
    max_pairs_limit = None
    if max_pairs_str.isdigit():
        max_pairs_limit = int(max_pairs_str)
        if max_pairs_limit <= 0:
            print("Maximum pairs must be a positive number. Setting to no limit.")
            max_pairs_limit = None
    elif max_pairs_str:
        print("Invalid input for maximum pairs. Setting to no limit.")
    
    # Step 0: Upload local images
    # Check if local directory has images first
    if not any(fname.lower().endswith(config.VALID_IMAGE_EXTENSIONS) for fname in os.listdir(config.LOCAL_IMAGE_DIR)):
        print(f"No images found in {config.LOCAL_IMAGE_DIR}. Please add images to this directory.")
        # Ask user if they want to proceed by listing images already in GCS
        proceed_gcs = input(f"Do you want to try listing images directly from GCS path {config.get_gcs_image_upload_path()}/ instead? (yes/no): ").lower()
        if proceed_gcs == 'yes':
            all_gcs_uris = utils.list_gcs_files(config.GCS_BUCKET_NAME, config.GCS_IMAGE_UPLOAD_PREFIX, config.VALID_IMAGE_EXTENSIONS)
            if not all_gcs_uris:
                print(f"No images found in GCS path {config.get_gcs_image_upload_path()}/ either. Exiting.")
                return
            print(f"Found {len(all_gcs_uris)} images in GCS. Proceeding with these.")
        else:
            print("Exiting.")
            return
    else:
        all_gcs_uris = upload_local_images_to_gcs()
        if not all_gcs_uris:
            print("Image upload failed or no images were uploaded. Exiting.")
            return

    # Step 1: Select context and target images
    context_uris, target_uris = select_context_and_target_images(all_gcs_uris)
    if not context_uris or not target_uris:
        print("Failed to select context or target images. Exiting.")
        return

    # Step 2: Generate descriptions for target images
    target_descriptions = generate_descriptions_for_targets(target_uris, subject_desc_for_context)
    valid_descriptions_count = sum(1 for desc in target_descriptions.values() if "[Error:" not in desc)
    if valid_descriptions_count == 0:
        print("No valid descriptions were generated for target images. Exiting.")
        return
    
    # Filter out targets that had description errors before creating JSONL
    valid_target_image_descriptions = {
        uri: desc for uri, desc in target_descriptions.items() if "[Error:" not in desc
    }

    # Step 3: Create and upload training JSONL
    final_jsonl_gcs_uri = create_training_jsonl(
        context_uris,
        valid_target_image_descriptions,
        subject_desc_for_context,
        max_pairs=max_pairs_limit
    )

    if final_jsonl_gcs_uri:
        print(f"\nData preparation complete! Training JSONL is at: {final_jsonl_gcs_uri}")
        # Store this URI for the next step (training)
        with open("last_jsonl_uri.txt", "w") as f: # Persist for next script
            f.write(final_jsonl_gcs_uri)
        print(f"The GCS URI of the JSONL file has been saved to last_jsonl_uri.txt")
    else:
        print("\nData preparation failed to produce a JSONL file.")

if __name__ == "__main__":
    # Basic authentication check / setup for Google Cloud SDK
    # This is usually handled by `gcloud auth application-default login` or environment variables
    try:
        utils.get_storage_client() # Initialize storage client (and implicitly project)
        utils.get_gemini_client()  # Initialize gemini client
        print("Google Cloud clients initialized successfully.")
    except Exception as e:
        print(f"Error initializing Google Cloud clients: {e}")
        print("Please ensure you are authenticated with Google Cloud (e.g., run 'gcloud auth application-default login') "
              "and the necessary APIs (Storage, Vertex AI) are enabled for your project.")
        exit(1)
        
    main()
