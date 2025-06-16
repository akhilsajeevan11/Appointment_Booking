# Appointment Booking System with Voice Agent

This project is a Python-based appointment booking system that uses a voice-enabled agent for interaction. Users can speak to the system to book new appointments or view existing ones. The agent utilizes Google's Generative AI. Speech-to-Text is streamed from the user's microphone, and the agent's voice responses are also streamed for lower latency, providing a more real-time conversational experience.

## Prerequisites

1.  **Python**: Python 3.7+ installed.
2.  **pip**: Python package installer.
3.  **Google Cloud Platform (GCP) Project**:
    *   A valid GCP project.
    *   Enable the **Cloud Speech-to-Text API** and **Cloud Text-to-Speech API** for your project.
    *   **Authentication**: Set up Application Default Credentials (ADC). The easiest way for local development is to install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and run:
        ```bash
        gcloud auth application-default login
        ```
        This will store credentials locally that the Python client libraries can automatically pick up.
4.  **Audio Hardware**:
    *   A working microphone connected to your system for voice input.
    *   Speakers or headphones for audio output.
5.  **MySQL Database**:
    *   A running MySQL server instance.
    *   You need to have a database created and user credentials with permissions to create tables and read/write data. The application will attempt to create the `appointments` table if it doesn't exist within the specified database.
6.  **Environment Variables**:
    *   `GOOGLE_API_KEY`: An API key for the Google Generative AI service (e.g., Gemini) used by the appointment agent.
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
    This will install all necessary Python packages, including the Google Cloud client libraries, and `sounddevice` which is used for both microphone input and streaming audio playback. Libraries for the agent are also included. (`playsound` is also included in `requirement.txt` but primary audio I/O is handled by `sounddevice`).

4.  **Set Up Environment Variables**:
    Create a `.env` file in the root directory of the project and add your specific configuration:
    ```env
    GOOGLE_API_KEY="your_google_generative_ai_api_key"
    MYSQL_HOST="localhost"
    MYSQL_USER="your_mysql_user"
    MYSQL_PASSWORD="your_mysql_password"
    MYSQL_DATABASE_NAME="your_database_name"
    MYSQL_PORT="3306"

    # If you are using a specific service account JSON file for GCP authentication
    # (instead of gcloud ADC), you might also set:
    # GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    # However, using `gcloud auth application-default login` is generally simpler for local development.
    ```
    The application uses `python-dotenv` to load these variables.

## Running the Application

1.  Ensure your MySQL server is running and accessible with the configured credentials.
2.  Ensure your microphone is connected and configured as the default input device for your system.
3.  Run the main script:
    ```bash
    python main.py
    ```

## How to Use

*   When you run `main.py`, the system will greet you.
*   The console will display "Listening..." when it's ready for your voice input.
*   Speak your command clearly. For example:
    *   "Book an appointment for John Doe for next Friday at 2 PM for a checkup."
    *   "I want to schedule a meeting."
    *   "View my appointments."
*   The system will transcribe your speech (you'll see "You said: <your_transcribed_text>" and interim STT results in the console).
*   The agent will process your request and respond. The response will be printed to the console and spoken aloud.
*   To quit the application, say "exit".

## Troubleshooting Audio (Linux)
If `sounddevice` or `playsound` have issues on Linux, you might need to install system dependencies:
*   For `sounddevice` (PortAudio):
    ```bash
    sudo apt-get update
    sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
    ```
*   For `playsound` (which might use GStreamer via `pygobject` on Linux):
    ```bash
    sudo apt-get install python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 gir1.2-glib-2.0
    ```
    If `playsound` still has issues, ensure GStreamer plugins are installed (`gstreamer1.0-plugins-good`, `gstreamer1.0-plugins-ugly`, etc.).

```
