# Appointment Booking System with Local Whisper STT and gTTS

This project is a Python-based appointment booking system that uses a voice-enabled agent for interaction. Speech-to-Text (STT) is performed locally using OpenAI's Whisper model, Text-to-Speech (TTS) uses the `gTTS` library (which leverages the Google Translate API), and the agent's core logic utilizes Google's Generative AI models via LangChain.

## Prerequisites

1.  **Python**: Python 3.7+ installed.
2.  **pip**: Python package installer.
3.  **`ffmpeg`**:
    *   `ffmpeg` is required by the `openai-whisper` library for audio processing.
    *   Installation (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install ffmpeg`
    *   For other systems (Windows, macOS), download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's in your system's PATH.
4.  **Whisper STT (Local Model)**:
    *   No API key is required for Whisper.
    *   The `openai-whisper` library will automatically download the specified model (default is "small.en" in this project) on its first use if not already cached. Models are typically stored in `~/.cache/whisper`.
    *   An internet connection is required for this initial model download.
5.  **gTTS (Text-to-Speech)**:
    *   `gTTS` does not require an API key.
    *   It requires an active internet connection at runtime to synthesize speech by interfacing with the Google Translate API.
6.  **Audio Hardware**:
    *   A working microphone connected to your system for voice input.
    *   Speakers or headphones for audio output.
7.  **MySQL Database**:
    *   A running MySQL server instance.
    *   You need to have a database created and user credentials with permissions to create tables and read/write data. The application will attempt to create the `appointments` table if it doesn't exist.
8.  **Environment Variables** (to be set, e.g., in a `.env` file):
    *   `GOOGLE_API_KEY`: Your API key for Google Generative AI (e.g., for the Gemini model used by the agent).
    *   `MYSQL_HOST`: Hostname of your MySQL server (e.g., `localhost`).
    *   `MYSQL_USER`: MySQL username.
    *   `MYSQL_PASSWORD`: MySQL password.
    *   `MYSQL_DATABASE_NAME`: Name of the database to use.
    *   `MYSQL_PORT`: Port number for MySQL (default is `3306`).
    *   *(Optional)* `WHISPER_MODEL_NAME`: You can set this to use a different Whisper model (e.g., "base.en", "medium.en"). If not set, "small.en" is used by default.

## Setup Instructions

1.  **Clone the Repository** (if applicable, otherwise ensure you have all project files).
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install System Dependencies**:
    *   Ensure `ffmpeg` is installed on your system (see Prerequisites).

4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirement.txt
    ```
    This will install all necessary Python packages, including `openai-whisper` for STT, `gTTS` for TTS, `playsound` for audio playback (used by gTTS handler), `sounddevice` for microphone input, and libraries for the agent.

5.  **Whisper Model Download Note**:
    *   The first time you run the application, the `openai-whisper` library will download the specified speech recognition model (e.g., "small.en"). This may take some time and requires an internet connection. Subsequent runs will use the cached model.

6.  **Set Up Environment Variables**:
    Create a `.env` file in the root directory of the project and add your specific configuration:
    ```env
    GOOGLE_API_KEY="your_google_generative_ai_api_key"

    MYSQL_HOST="localhost"
    MYSQL_USER="your_mysql_user"
    MYSQL_PASSWORD="your_mysql_password"
    MYSQL_DATABASE_NAME="your_database_name"
    MYSQL_PORT="3306"

    # Optional: Specify a different Whisper model
    # WHISPER_MODEL_NAME="base.en"
    ```
    Replace keys and MySQL details with your actual values.

## Running the Application

1.  Ensure your MySQL server is running and accessible.
2.  Ensure your microphone is connected and configured.
3.  Verify all required environment variables in `.env` are correctly set.
4.  Run the main script:
    ```bash
    python main.py
    ```

## How to Use

*   When you run `main.py`, the system will greet you using `gTTS`.
*   The console will display "Whisper STT: Press Enter to start recording..." when it's ready for your voice input. Press Enter and speak.
*   The Whisper STT currently records for a fixed duration (e.g., 7 seconds) after you press Enter, then transcribes the audio. It's not a continuous real-time stream of words as they appear.
*   The system will transcribe your speech using the local Whisper model.
*   The agent will process your request and respond. The response will be spoken aloud using `gTTS`. `gTTS` generates the full audio for a sentence before playing it.
*   To quit the application, say "exit" (or type it if STT fails).

## Troubleshooting

*   **Audio Input/Output (Linux - `sounddevice`)**: If `sounddevice` has issues (used for microphone input), you might need to install system dependencies for PortAudio:
    ```bash
    sudo apt-get update
    sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
    ```
    Ensure your microphone is correctly configured in your Linux sound settings.
*   **Whisper STT (`ffmpeg`)**: If you see errors related to `ffmpeg` when running `openai-whisper` for the first time or during transcription, ensure `ffmpeg` is correctly installed and accessible in your system's PATH.
*   **gTTS Playback (`playsound` on Linux)**: `playsound` (especially version 1.2.2) often has issues on Linux due to backend dependencies. If audio playback for gTTS fails or causes errors:
    *   You might need to install GStreamer development libraries: `sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
    *   Alternatively, for more robust playback on Linux, consider replacing `playsound` with a library like `pygame` or `simpleaudio` for playing the temporary MP3 files, though this would require code changes in `TextToSpeechHandler`.

```
