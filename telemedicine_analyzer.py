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
# IMPORTANT: Replace with your actual Hugging Face access token (https://huggingface.co/settings/tokens)
# Please accept the terms and conditons from: https://huggingface.co/pyannote/speaker-diarization-3.1
YOUR_HF_TOKEN = "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" #"PASTE_YOUR_HUGGING_FACE_TOKEN_HERE"

# Your Google AI API key for Gemini (https://aistudio.google.com/app/apikey)
YOUR_GOOGLE_API_KEY = "AIzaSyASYSqbrn3hHPz7U3ZxEmDOXer3v9lHCzk" #"PASTE_YOUR_GOOGLE_API_KEY_HERE"

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
    Based on the following transcript from a telemedicine call, identify the primary intent of the call.
    Choose from one of the following categories:
    - "Appointment Scheduling": For booking, rescheduling, or canceling appointments.
    - "Prescription Refill": For requesting a refill of an existing medication.
    - "Symptom Discussion": When the patient is describing symptoms for a new or ongoing issue.
    - "Medical Advice/Question": For general medical questions not tied to a new diagnosis.
    - "Follow-up Consultation": Discussing progress or results after a previous appointment.
    - "Billing Inquiry": Questions related to billing, insurance, or payments.
    - "Other": If the intent does not clearly fit into any of the above categories.

    Respond with only the category name.

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
            intent_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("All models initialized successfully.")

        except Exception as e:
            print(f"FATAL: Failed to initialize models: {e}")
            exit() # Exit the script if models can't be loaded

        # --- Find Audio Files ---
        # Look for common audio file types
        supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
        audio_files = []
        for ext in supported_extensions:
            audio_files.extend(glob.glob(os.path.join(INPUT_FOLDER_PATH, ext)))

        if not audio_files:
            print(f"No audio files found in '{INPUT_FOLDER_PATH}'.")
            exit()

        # --- Process Each File ---
        all_summaries = []
        for audio_file_path in audio_files:
            summary = process_audio_file(
                audio_file_path,
                diarization_pipeline,
                transcription_model,
                intent_model
            )
            if summary:
                all_summaries.append(summary)
        
        # --- Save Final Results ---
        if not all_summaries:
            print("\nNo files were successfully processed.")
        else:
            summary_df = pd.DataFrame(all_summaries)
            summary_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\n--- Batch Processing Complete! ---")
            print(f"Summary of {len(all_summaries)} file(s) saved to '{OUTPUT_CSV_PATH}'.")

