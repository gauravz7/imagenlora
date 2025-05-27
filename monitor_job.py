# monitor_job.py
import os
import time
import json

import config
import utils

def get_tuning_job_path() -> str:
    """
    Retrieves the Vertex AI tuning job path.
    Tries to read from 'last_tuning_job_path.txt' first, then prompts user.
    Example path: projects/{PROJECT_ID}/locations/{LOCATION}/tuningJobs/{JOB_ID}
    """
    try:
        with open("last_tuning_job_path.txt", "r") as f:
            path = f.read().strip()
            # Basic validation for the path format
            if path.startswith("projects/") and "/locations/" in path and "/tuningJobs/" in path:
                print(f"Found tuning job path from last session: {path}")
                return path
            else:
                print(f"Found invalid path in last_tuning_job_path.txt: {path}")
    except FileNotFoundError:
        print("last_tuning_job_path.txt not found.")
    
    while True:
        path = input(f"Enter the Vertex AI tuning job path (e.g., projects/your-project/locations/us-central1/tuningJobs/123...): ").strip()
        if path.startswith("projects/") and "/locations/" in path and "/tuningJobs/" in path:
            return path
        else:
            print("Invalid job path format. It should look like 'projects/PROJECT_ID/locations/LOCATION/tuningJobs/JOB_ID'.")


def main():
    print("--- Monitoring Imagen Fine-Tuning Job ---")

    job_to_monitor = get_tuning_job_path()
    if not job_to_monitor:
        print("No tuning job path provided. Exiting.")
        return

    print(f"Attempting to get status for job: {job_to_monitor}")
    
    # Extract location from job_to_monitor path for the API call if needed, or use config.LOCATION
    # Assuming the job is in config.LOCATION
    
    status_data = utils.monitor_tuning_job(job_name_path=job_to_monitor, location=config.LOCATION)

    if status_data:
        print("\n--- Job Status ---")
        print(json.dumps(status_data, indent=2)) # Pretty print the status

        job_state = status_data.get("state", "JOB_STATE_UNSPECIFIED")
        
        if job_state == "JOB_STATE_SUCCEEDED":
            print("\nSUCCESS: Tuning job completed successfully!")
            tuned_model_info = status_data.get("tunedModel", {})
            endpoint = tuned_model_info.get("endpoint")
            if endpoint:
                print(f"Tuned Model Endpoint READY: {endpoint}")
                # The numeric ID is the last part of the endpoint path
                endpoint_id = endpoint.split('/')[-1]
                print(f"===> Endpoint ID for Prediction: {endpoint_id} <===")
            else:
                print("Job succeeded, but tuned model endpoint not found in the response yet. It might still be provisioning.")
        elif job_state in ["JOB_STATE_RUNNING", "JOB_STATE_PENDING"]:
            print(f"\nINFO: Tuning job is currently: {job_state}. Please check again later.")
        elif job_state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"\nERROR: Tuning job ended with state: {job_state}.")
            # You might want to print specific error messages if available in status_data.error
            if status_data.get("error"):
                print(f"Error details: {json.dumps(status_data.get('error'), indent=2)}")
        else:
            print(f"\nINFO: Job is in state: {job_state}.")
    else:
        print("Failed to retrieve job status. Check logs for errors.")

if __name__ == "__main__":
    try:
        utils.get_storage_client() # For any GCS interactions if needed by utils indirectly
        utils.get_gemini_client() # For any Gemini interactions if needed by utils indirectly
        print("Google Cloud clients initialized successfully for monitoring script.")
    except Exception as e:
        print(f"Error initializing Google Cloud clients: {e}")
        print("Please ensure you are authenticated with Google Cloud and APIs are enabled.")
        exit(1)

    main()
