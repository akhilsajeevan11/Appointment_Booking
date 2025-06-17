# Appointment Booking System with Voice Agent

This project is a Python-based appointment booking system that uses a voice-enabled agent for interaction. Speech-to-Text (STT) and Text-to-Speech (TTS) are both handled by Deepgram's real-time streaming APIs. The agent's core logic uses Google's Generative AI models via LangChain.

## Prerequisites

1.  **Python**: Python 3.7+ installed.
2.  **pip**: Python package installer.
3.  **Deepgram Account**:
    *   A Deepgram account and a **Deepgram API Key** are required for both Speech-to-Text and Text-to-Speech services.
4.  **Audio Hardware**:
    *   A working microphone connected to your system for voice input.
    *   Speakers or headphones for audio output.
5.  **MySQL Database**:
    *   A running MySQL server instance.
    *   You need to have a database created and user credentials with permissions to create tables and read/write data. The application will attempt to create the `appointments` table if it doesn't exist.
6.  **Environment Variables** (to be set, e.g., in a `.env` file):
    *   `GOOGLE_API_KEY`: Your API key for Google Generative AI (e.g., for the Gemini model used by the agent).
    *   `DEEPGRAM_API_KEY`: Your API key for Deepgram services (used for both STT and TTS).
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
    This will install all necessary Python packages, including `deepgram-sdk` (for both STT and TTS), `sounddevice` for audio I/O, and libraries for the agent (e.g., LangChain, Google Generative AI).

4.  **Set Up Environment Variables**:
    Create a `.env` file in the root directory of the project and add your specific configuration:
    ```env
    GOOGLE_API_KEY="your_google_generative_ai_api_key"
    DEEPGRAM_API_KEY="your_deepgram_api_key_for_stt_and_tts"

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

*   When you run `main.py`, the system will greet you using Deepgram TTS.
*   The console will display "Listening (Deepgram)..." when it's ready for your voice input.
*   Speak your command clearly.
*   The system will transcribe your speech using Deepgram.
*   The agent will process your request and respond. The response will be spoken aloud using Deepgram TTS.
*   To quit the application, say "exit".

## Troubleshooting Audio (Linux)
If `sounddevice` has issues on Linux (used for both microphone input and Deepgram audio output), you might need to install system dependencies for PortAudio:
```bash
sudo apt-get update
sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
```
Ensure your microphone is correctly configured in your Linux sound settings.
```
