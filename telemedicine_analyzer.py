#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import google.generativeai as genai
import glob 

# --- 1. Configuration ---
# IMPORTANT: Replace with your actual Hugging Face access token (https://huggingface.co/settings/tokens     )
# Please accept the terms and conditons from: https://huggingface.co/pyannote/speaker-diarization-3.1     
YOUR_HF_TOKEN = "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE"

# Your Google AI API key for Gemini (https://aistudio.google.com/app/apikey     )
YOUR_GOOGLE_API_KEY = "PASTE_YOUR_GOOGLE_API_KEY_HERE"

# Path to the folder containing your audio files
INPUT_FOLDER_PATH = "Data_Audio_Files/" # Create a folder named "audio_files" and place your audio files there.

# Path for the final summary output CSV file
OUTPUT_CSV_PATH = "call_summary_output.csv"

# Choose a Whisper model size. "tiny" is the fastest, "base" is a good balance.
# Other options: "small", "medium", "large-v3" (slower, more accurate)
WHISPER_MODEL_SIZE = "base"


# --- 2. NEW: Intent Analysis Function ---

def get_call_intent(model, full_transcript):
    """
    Analyzes the intent of a telemedicine call transcript using the Gemini API.
    """
    if not full_transcript or not full_transcript.strip():
        return "No Transcript"

    # A more detailed prompt for intent classification in a specific domain
    prompt = f"""
    Based on the following transcript from a telemedicine call, identify:
    
    1. The **primary intent** of the call. Choose from one of the following categories:
       - "Appointment Scheduling"
       - "Prescription Refill"
       - "Symptom Discussion"
       - "Medical Advice/Question"
       - "Follow-up Consultation"
       - "Billing Inquiry"
       - "Other"

    2. The **subject matter** of the call. 
       - If the intent is "Symptom Discussion", "Medical Advice/Question", or "Follow-up Consultation", 
         specify the main medical area involved (e.g., "Gastrointestinal", "Cardiovascular", "Respiratory", 
         "Neurological", "Dermatological", "Musculoskeletal", "Psychiatric", "Endocrine", "General Health", etc.).
       - If the call does not involve a medical subject (e.g., Appointment Scheduling, Billing), respond with "Not Applicable".

    Respond in the following format:
    Intent: Subject Matter
    
    Eg: Symptom Discussion: Gastrointestinal

    Transcript:
    "{full_transcript}"
    """


    try:
        # Call the Gemini API
        response = model.generate_content(prompt)
        # Clean up the response to get a single phrase
        intent = response.text.strip()
        return intent
    except Exception as e:
        print(f"      - Could not get intent: {e}")
        return "Error"


# --- 3. MODIFIED: Main Processing Function (for a single file) ---

def process_audio_file(audio_path, diarization_pipeline, transcription_model, intent_model):
    """
    Processes a single audio file for diarization, transcription, and intent analysis.
    Returns a dictionary with the summary.
    """
    print(f"\n--- Processing file: {os.path.basename(audio_path)} ---")
    sr = 16000 # Target sample rate

    # --- Step 3.1: Load Audio ---
    print("  - Loading audio...")
    try:
        # Load and resample audio to 16kHz mono
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        # Write to a temporary file for the diarization pipeline
        temp_wav_path = "temp_audio_16khz_mono.wav"
        sf.write(temp_wav_path, y, sr)
    except Exception as e:
        print(f"  - Error loading audio file {audio_path}: {e}")
        return None

    # --- Step 3.2: Speaker Diarization ---
    print("  - Running speaker diarization...")
    try:
        diarization_result = diarization_pipeline(temp_wav_path)
        num_speakers = len(diarization_result.labels())
        print(f"  - Found {num_speakers} speakers.")
    except Exception as e:
        print(f"  - Error during diarization: {e}")
        # Ensure temp file is removed even on error
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return None

    # --- Step 3.3: Transcription ---
    print("  - Transcribing segments...")
    full_transcript_parts = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_np = y[start_sample:end_sample]

        # Transcribe the segment
        segments, _ = transcription_model.transcribe(segment_np, word_timestamps=False)
        segment_text = "".join(segment.text for segment in segments).strip()

        if segment_text:
            full_transcript_parts.append(f"{speaker}: {segment_text}\n")
    
    full_transcript = "".join(full_transcript_parts)
    print("  - Transcription complete.")

    # --- Step 3.4: Intent Analysis ---
    print("  - Analyzing call intent...")
    call_intent = get_call_intent(intent_model, full_transcript)
    print(f"  - Determined Intent: {call_intent}")

    # --- Step 3.5: Clean up temp file ---
    os.remove(temp_wav_path)
    
    # --- Step 3.6: Return Summary ---
    summary = {
        "filename": os.path.basename(audio_path),
        "number_of_speakers": num_speakers,
        "intent": call_intent
    }
    return summary


# --- 4. MODIFIED: Main Batch Processing Script ---
if __name__ == "__main__":
    # --- Health Checks ---
    if YOUR_HF_TOKEN == "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE" or YOUR_GOOGLE_API_KEY == "PASTE_YOUR_GOOGLE_API_KEY_HERE":
        print("ERROR: Please provide both your Hugging Face and Google AI API keys in the script.")
    elif not os.path.isdir(INPUT_FOLDER_PATH):
        print(f"ERROR: Input folder not found at '{INPUT_FOLDER_PATH}'")
        print("Please create the folder and add your audio files.")
    else:
        print("--- Starting Batch Processing Pipeline ---")
        
        # --- Determine files to skip ---
        # Create a set to store filenames that have already been processed
        processed_files = set()
        # Check if the output CSV already exists
        if os.path.exists(OUTPUT_CSV_PATH):
            try:
                # Read the existing CSV
                existing_df = pd.read_csv(OUTPUT_CSV_PATH)
                # Check if the 'filename' column exists
                if 'filename' in existing_df.columns:
                    # Add all existing filenames to the set for fast lookup
                    processed_files = set(existing_df['filename'].tolist())
                    print(f"Found existing output file '{OUTPUT_CSV_PATH}'. Will skip {len(processed_files)} already processed file(s).")
                else:
                    # If 'filename' column is missing, log a warning but don't skip any files
                    print(f"Warning: Existing CSV '{OUTPUT_CSV_PATH}' does not contain a 'filename' column. Processing all files.")
            except Exception as e:
                # If there's an error reading the CSV, log it but continue, processing all files
                print(f"Warning: Could not read existing CSV '{OUTPUT_CSV_PATH}' ({e}). Processing all files.")
        else:
            # If the CSV doesn't exist, no files have been processed yet
            print(f"No existing output file '{OUTPUT_CSV_PATH}' found. Processing all files.")

        # --- Initialize Models ONCE for efficiency ---
        print("Initializing models (this may take a moment)...")
        try:
            # Diarization Model
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=YOUR_HF_TOKEN
            )
            diarization_pipeline.to(torch.device("cpu")) # Use "cuda" if you have a compatible GPU

            # Transcription Model
            transcription_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

            # Intent Analysis Model
            genai.configure(api_key=YOUR_GOOGLE_API_KEY)
            intent_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("All models initialized successfully.")

        except Exception as e:
            print(f"FATAL: Failed to initialize models: {e}")
            exit() # Exit the script if models can't be loaded

        # --- Find Audio Files ---
        # Look for common audio file types
        supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
        audio_files = []
        for ext in supported_extensions:
            # Use recursive glob if you want to search subdirectories too: glob.glob(os.path.join(INPUT_FOLDER_PATH, "**", ext), recursive=True)
            audio_files.extend(glob.glob(os.path.join(INPUT_FOLDER_PATH, ext)))

        if not audio_files:
            print(f"No audio files found in '{INPUT_FOLDER_PATH}'.")
            exit()

        # --- Process Each File (skipping already processed ones) ---
        new_summaries = [] # Store summaries of newly processed files
        for audio_file_path in audio_files:
            filename = os.path.basename(audio_file_path)
            # Check if the file has already been processed
            if filename in processed_files:
                print(f"\nSkipping already processed file: {filename}")
                continue # Skip to the next file

            # If not processed, process it
            summary = process_audio_file(
                audio_file_path,
                diarization_pipeline,
                transcription_model,
                intent_model
            )
            if summary:
                new_summaries.append(summary) # Add successful summary to the list
        
        # --- Save Final Results (Append new results) ---
        if not new_summaries:
             # Check if any files were skipped due to being processed before
            if processed_files:
                print("\nNo new files were processed. Output CSV is up-to-date.")
            else:
                 print("\nNo files were successfully processed.")
        else:
            # Create a DataFrame from the new summaries
            new_summary_df = pd.DataFrame(new_summaries)

            # Check if the output CSV already exists
            if os.path.exists(OUTPUT_CSV_PATH):
                # If it exists, append the new data without writing the header again
                new_summary_df.to_csv(OUTPUT_CSV_PATH, mode='a', index=False, header=False)
                print(f"\n--- Batch Processing Complete! ---")
                print(f"Summary of {len(new_summaries)} new file(s) appended to '{OUTPUT_CSV_PATH}'.")
            else:
                # If it doesn't exist, create it with the header
                new_summary_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"\n--- Batch Processing Complete! ---")
                print(f"Summary of {len(new_summaries)} new file(s) saved to new file '{OUTPUT_CSV_PATH}'.")
