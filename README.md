# Appointment Booking System with Voice Agent

This project is a Python-based appointment booking system that uses a voice-enabled agent for interaction. Speech-to-Text (STT) is handled by Deepgram's real-time streaming API, Text-to-Speech (TTS) is performed locally using Piper TTS, and the agent's core logic uses Google's Generative AI models via LangChain.

## Prerequisites

1.  **Python**: Python 3.7+ installed.
2.  **pip**: Python package installer.
3.  **Deepgram Account**:
    *   A Deepgram account is required for Speech-to-Text.
    *   You'll need a **Deepgram API Key**.
4.  **Piper TTS Setup**:
    *   **Piper Executable**: Download the Piper executable suitable for your system from the [Piper GitHub releases page](https://github.com/rhasspy/piper/releases).
    *   **Piper Voice Model**: Download a voice model for Piper. Each voice consists of an `.onnx` file and a corresponding `.onnx.json` configuration file. You can find voices on [Hugging Face (e.g., rhasspy/piper-voices)](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0).
    *   You will need to set environment variables pointing to the paths of the executable and these two model files.
5.  **Audio Hardware**:
    *   A working microphone connected to your system for voice input.
    *   Speakers or headphones for audio output.
6.  **MySQL Database**:
    *   A running MySQL server instance.
    *   You need to have a database created and user credentials with permissions to create tables and read/write data. The application will attempt to create the `appointments` table if it doesn't exist.
7.  **Environment Variables** (to be set, e.g., in a `.env` file):
    *   `GOOGLE_API_KEY`: Your API key for Google Generative AI (e.g., for the Gemini model used by the agent).
    *   `DEEPGRAM_API_KEY`: Your API key for the Deepgram STT service.
    *   `PIPER_EXE_PATH`: Full path to the downloaded `piper` executable file.
    *   `PIPER_MODEL_ONNX_PATH`: Full path to the chosen Piper `.onnx` voice model file.
    *   `PIPER_MODEL_JSON_PATH`: Full path to the corresponding `.onnx.json` voice configuration file for the chosen model.
    *   `MYSQL_HOST`: Hostname of your MySQL server (e.g., `localhost`).
    *   `MYSQL_USER`: MySQL username.
    *   `MYSQL_PASSWORD`: MySQL password.
    *   `MYSQL_DATABASE_NAME`: Name of the database to use.
    *   `MYSQL_PORT`: Port number for MySQL (default is `3306`).

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

3.  **Install Dependencies**:
    ```bash
    pip install -r requirement.txt
    ```
    This will install all necessary Python packages, including `deepgram-sdk` for speech-to-text, `piper-tts` (which provides tools related to Piper, though you download the executable separately as per above), `sounddevice` for audio I/O, and libraries for the agent.

4.  **Download Piper Executable and Voice Model**:
    *   Download the `piper` executable from [Piper GitHub releases](https://github.com/rhasspy/piper/releases) and place it in a known location.
    *   Download your chosen `.onnx` voice file and its `.onnx.json` config file from a source like [Hugging Face rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0) and place them in a known location.

5.  **Set Up Environment Variables**:
    Create a `.env` file in the root directory of the project and add your specific configuration:
    ```env
    GOOGLE_API_KEY="your_google_generative_ai_api_key"
    DEEPGRAM_API_KEY="your_deepgram_api_key"

    PIPER_EXE_PATH="/path/to/your/piper_executable/piper"
    PIPER_MODEL_ONNX_PATH="/path/to/your/voice_model.onnx"
    PIPER_MODEL_JSON_PATH="/path/to/your/voice_model.onnx.json"

    MYSQL_HOST="localhost"
    MYSQL_USER="your_mysql_user"
    MYSQL_PASSWORD="your_mysql_password"
    MYSQL_DATABASE_NAME="your_database_name"
    MYSQL_PORT="3306"
    ```
    Replace paths and keys with your actual values.

## Running the Application

1.  Ensure your MySQL server is running and accessible.
2.  Ensure your microphone is connected and configured.
3.  Verify all environment variables in `.env` are correctly set.
4.  Run the main script:
    ```bash
    python main.py
    ```

## How to Use

*   When you run `main.py`, the system will greet you using Piper TTS.
*   The console will display "Listening (Deepgram)..." when it's ready for your voice input.
*   Speak your command clearly.
*   The system will transcribe your speech using Deepgram.
*   The agent will process your request and respond. The response will be spoken aloud using Piper TTS.
*   To quit the application, say "exit".

## Troubleshooting Audio (Linux)
If `sounddevice` has issues on Linux (used for both microphone input and Piper audio output), you might need to install system dependencies for PortAudio:
```bash
sudo apt-get update
sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
```
Ensure your microphone is correctly configured in your Linux sound settings.
```
