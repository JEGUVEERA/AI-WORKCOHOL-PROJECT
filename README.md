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
│   main.ipynb          # Jupyter notebook for experimentation
│   README.md           # Project documentation
│   
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
- Support for Jupyter notebooks

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

## Configuration
- Add API keys and settings in `.env`
- Modify `secrets.toml` for additional configurations
- streamlit/secrets.toml

## API Keys in `streamlit/secrets.toml`

The API keys `GEMINI_API_KEY` and `HF_API_KEY` are typically added to the `streamlit/secrets.toml` file.

## Installation of Ollama Models
To install Ollama and any model, run the following command:

```bash
ollama install <model_name>
```
## To pull an existing model to use
```bash
ollama pull <model_name>
```

 
## Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run main.py
```

## Contributing
Contributions are welcome! Follow standard Git workflow:
```sh
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```

