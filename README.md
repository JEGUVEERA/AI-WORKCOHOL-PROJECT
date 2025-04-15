# AI-WORKCOHOL-PROJECT


### 🎥 Demo

[Watch Video on Google Drive](https://drive.google.com/file/d/1s74blkI1XZO18ui1F-fs9HnV6yNNOQWn/view?usp=sharing)


## Overview
AI Workcohol Project is an AI-powered marketing and content generation platform that leverages multiple AI models including Gemini API, Ollama, and LangChain for creating diverse marketing content and performing text analysis.



## Overview
AI Workcohol Project is an AI-powered marketing and content generation platform that leverages multiple AI models including Gemini API, Ollama, and LangChain for creating diverse marketing content and performing text analysis.


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


### Required API Keys
```plaintext
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```


## API Keys Configuration
- Add API keys and settings in `.env`
- Modify `secrets.toml` for additional configurations
- Add API keys `GEMINI_API_KEY` to the `secrets.toml` file.



## Installation of Ollama Models
To install Ollama and any model from ollama, run the following command:

```bash
ollama install llama3.2:1b

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
streamlit run main.py
```



## Contributing
Contributions are welcome! Follow standard Git workflow:
```sh
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```
