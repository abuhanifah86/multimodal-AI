# Multimodal AI Assistant with LLaVA, Whisper, and Gradio

This project is a powerful multimodal AI assistant that can understand both spoken language and images. It uses OpenAI's Whisper for speech-to-text, the LLaVA 7B model for vision-language tasks, and Google's Text-to-Speech (gTTS) for voice responses, all wrapped in an easy-to-use Gradio web interface.

 <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bcf7a133-0c7d-4b74-ae2b-f68df16866ac" />


## ✨ Features

- **🎙️ Speech-to-Text**: Utilizes Whisper for accurate transcription of spoken commands or questions.
- **🖼️ Image Understanding**: Leverages the LLaVA 7B model to analyze and describe images based on your voice prompt.
- **🗣️ Voice Response**: Converts the AI's text response back into speech using gTTS, with automatic language detection based on the user's speech.
- **💪 Robustness**: Includes retry logic for API calls to the Ollama server and smart chunking for long audio files (>30s) to prevent errors.
- **📚 Conversation Logging**: Saves interaction history to a structured JSON file (`conversation_history.json`), laying the groundwork for future Retrieval-Augmented Generation (RAG) capabilities.
- **🖥️ User-Friendly Interface**: Built with Gradio for easy interaction through a web browser.

## ⚙️ How It Works

The application follows a simple yet powerful pipeline:

1.  **Input**: The user records audio and/or uploads an image via the Gradio UI.
2.  **Transcription**: The audio is sent to a local Whisper model, which transcribes it into text and detects the language.
3.  **Analysis**: The transcribed text and the uploaded image are sent to a locally running LLaVA model via the Ollama API.
4.  **Generation**: LLaVA processes the inputs and generates a relevant text response.
5.  **Synthesis**: The text response is converted into an audio file by gTTS, using the language detected by Whisper.
6.  **Output**: The transcribed text, the AI's text response, and the playable audio response are displayed back to the user in the UI.

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.8+
- Ollama installed and running.
- (Optional but Recommended) An NVIDIA GPU with CUDA for significantly faster model inference. The script will fall back to CPU if a GPU is not available.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/Multimodal-AI-App-using-Llava-7B.git
cd Multimodal-AI-App-using-Llava-7B
```

### 3. Install Dependencies

You have two options for installing dependencies: **Conda (recommended)** for robust handling of GPU dependencies, or **Pip** with a standard virtual environment.

#### Option A: Using Conda (Recommended)

This project includes an `environment.yml` file to create a consistent Conda environment with all necessary dependencies, including the correct CUDA toolkit version for PyTorch.

1.  **Prerequisite**: Ensure you have Anaconda or Miniconda installed.

2.  **Create the environment**: Open your terminal in the project root and run the following command. This will create a new environment named `multimodal-rag`.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment**: Before running the application, you must activate the new environment:
    ```bash
    conda activate multimodal-rag
    ```

#### Option B: Using Pip and a Virtual Environment

If you prefer not to use Conda, you can use `pip`. Note that this may require manual installation of CUDA and `ffmpeg` if they are not already on your system.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 4. Download AI Models

The script requires two models: LLaVA (from Ollama) and Whisper (from OpenAI).

*   **LLaVA**: Pull the model using the Ollama CLI.

    ```bash
    ollama pull llava:7b
    ```

*   **Whisper**: The `medium` Whisper model will be downloaded automatically by the script on its first run and cached for future use.

### 5. Run the Application

1.  **Start Ollama**: Make sure the Ollama application is running in the background.
2.  **Run the Python script**:
    ```bash
    python multimodal_rag.py
    ```
3.  **Open the UI**: The console will output a local URL (e.g., `http://127.0.0.1:7860`). Open this link in your web browser to start interacting with the assistant.

## 📝 Logging

The application generates two types of logs:

- **Debug Log**: A timestamped text file (e.g., `2025-08-12_10-30-00_log.txt`) is created for each session. It logs key events, API calls, and errors, which is useful for debugging.
- **Conversation History**: All interactions (inputs, outputs, file paths) are appended to `conversation_history.json`. This structured data can be used for analysis, fine-tuning, or as a knowledge base for a RAG system.

## 🔧 Configuration

You can modify the Ollama configuration directly in the `multimodal_rag.py` script:

```python
OLLAMA_CONFIG = {
    "model": "llava:7b",   # Change to a different model if needed
    "stream": False
}
```
