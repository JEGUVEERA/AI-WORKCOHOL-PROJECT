# AI-WORKCOHOL-PROJECT


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

### Running the Project
```sh
python main.py  # Run the main script
```

## Configuration
- Add API keys and settings in `.env`
- Modify `secrets.toml` for additional configurations

## Contributing
Contributions are welcome! Follow standard Git workflow:
```sh
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```
