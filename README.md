# Telemedicine Call Analysis Pipeline

This project contains a Python script that automates the analysis of telemedicine audio calls. It processes a folder of audio files to:
1.  Detect the number of speakers in each call.
2.  Transcribe the entire conversation.
3.  Use an AI model to determine the primary intent of the call (e.g., "Symptom Discussion," "Appointment Scheduling").
4.  Generate a single CSV summary of all processed files.

## Prerequisites

*   **Python 3.13.3** or higher.
*   **Visual Studio Code** or any other code editor.
*   **Git** installed for version control.
*   **ffmpeg**: A command-line tool required by Whisper for audio processing. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

#### 1. Clone the Repository
First, get the project files from the Git repository.
```
git clone <your-repository-url-here>
cd <repository-name>
```

#### 2. Create and Activate a Virtual Environment
This creates an isolated environment for the project's dependencies.

**On Windows:**
```
# Create the virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate
```

**On macOS / Linux:**
```
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```
After activation, you will see `(venv)` at the beginning of your terminal prompt.

#### 3. Install Required Packages
Install all the necessary Python libraries from the `requirements.txt` file.
```
pip install -r requirements.txt
```

## Configuration

Before running the script, you need to configure a few things.

#### 1. Add Your Audio Files
Create a folder named `audio_files` in the root of the project directory. Place all your audio files (`.mp3`, `.wav`, etc.) inside this folder.

#### 2. Add Your API Keys
Open the main Python script (e.g., `main.py`) and replace the placeholder text with your actual API keys:
*   `YOUR_HF_TOKEN`: Your access token from [Hugging Face](https://huggingface.co/settings/tokens).
*   `YOUR_GOOGLE_API_KEY`: Your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## How to Run the Program

With your virtual environment active and the configuration complete, run the script from the terminal:
```
python main.py
```
The script will print its progress as it processes each file.

## Output

Once the script finishes, a summary file named `call_summary_output.csv` will be created in your project folder. This file contains the filename, number of speakers, and intent for each audio file.

## Git Workflow: Pushing and Pulling Changes

#### Pulling Updates
To get the latest changes from the central repository:
```
git pull origin main
```

#### Pushing Your Changes
To save your local changes to the central repository:
```
# Stage all your changes
git add .

# Commit the changes with a descriptive message
git commit -m "Your descriptive message about the changes"

# Push the changes
git push origin main
```
