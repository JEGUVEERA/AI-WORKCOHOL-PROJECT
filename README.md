﻿# AI-WORKCOHOL-PROJECT


## Overview
AI Workcohol Project is an AI-powered application designed to automate tasks using advanced AI models, including LangChain, GPT-4o, Gemini API, and other open-source technologies.

## Project Structure
```
AI WORKCOHOL PROJECT

│   .env                # Environment variables
│   .gitattributes      # Git configuration
│   .gitignore          # Git ignore rules
│   main.py             # Main application script
│   README.md           # Project documentation
│   app.log             # Application log
|
│   file_structure.txt  # File structure reference
│
├── .streamlit          # Streamlit configuration
│   ├── secrets.toml    # Secret keys and credentials
│
├── .venv               # Virtual environment
│   ├── Include         # Python headers
│   ├── Lib             # Installed Python packages
│   ├── Scripts         # Executable scripts
│   ├── pyvenv.cfg      # Virtual environment config
│
├── generate_images     # Directory for AI-generated images
```

## Features
- AI-driven automation
- Integration with multiple AI models
- Real-time processing capabilities
- Customizable configurations via `.env` and `secrets.toml`

## Setup Instructions
### Prerequisites
- Python 3.11+
- Virtual environment setup
- API keys for AI services

### Installation
```sh
# Clone the repository
git clone <repository-url>
cd AI-WORKCOHOL-PROJECT

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory of your project and add the following lines:

```plaintext
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
HF_API_KEY="YOUR_HF_API_KEY"
```

## API Keys Configuration
- Add API keys and settings in `.env`
- Modify `secrets.toml` for additional configurations
- Add API keys `GEMINI_API_KEY` and `HF_API_KEY` to the `secrets.toml` file.

## Installation of Ollama Models
To install Ollama and any model, run the following command:

```bash
ollama install <model_name>
```
## To pull an existing model to use
```bash
ollama pull <model_name>
```
## Using ollama serve

`ollama serve` is a command that allows you to run your models as a web service.

### Installation

1. Make sure you have `ollama` installed. You can install it by following the instructions on the [official website](https://ollama.com/docs/install).

### Usage

1. Open your terminal.
2. Navigate to the directory where your model is located.
3. Run the following command:
   
   ```bash
   ollama serve 
   ```

4. `ollama` will start the server, and you can access your model via the provided endpoint!

## To run the Streamlit application

To run the Streamlit application, use the following command:

```bash
streamlit run README.md
```


##  Deploying with ngrok

`ngrok` is a tool that creates a secure tunnel to your localhost, allowing you to expose a local server to the internet.


### Installation

1. Download `ngrok` from the [official website](https://ngrok.com/download).
2. Unzip the downloaded file and place it in a directory accessible from your command line.


### Usage

1. Open your terminal and navigate to the directory where `ngrok` is located.
2. Start your local server (e.g., using `python -m http.server` or your preferred method).
3. In another terminal window, run the following command:
   
   ```bash
   ngrok http [PORT]
   ```
   Replace `[PORT]` with the port number your local server is running on (e.g., `ngrok http 8000`).
4. `ngrok` will provide you with a public URL that you can use to access your local server from anywhere!



## Contributing
Contributions are welcome! Follow standard Git workflow:
```sh
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```
