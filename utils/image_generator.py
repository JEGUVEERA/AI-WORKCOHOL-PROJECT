import os
import requests
import random
import streamlit as st

STYLE_PROMPTS = {
    "Realistic": "photo-realistic, ultra high resolution, detailed lighting, 8k",
    "Anime": "anime style, crisp lines, colorful, dramatic lighting, cel-shading",
    "Digital Art": "digital painting, vivid colors, high contrast, concept art",
    "Fantasy": "fantasy theme, magical lighting, surreal, ethereal atmosphere",
    "Cyberpunk": "cyberpunk aesthetic, neon lights, futuristic, dystopian",
    "No Style": ""
}

HF_API_KEY = os.getenv("HF_API_KEY") or st.secrets.get("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def generate_image(prompt, negative_prompt="", width=512, height=512, steps=30, seed=None):
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
        "parameters": {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
        }
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    response = requests.post(API_URL, headers=headers, json=payload)

    # Check if the response status is 200 and print the response content for debugging
    if response.status_code == 200:
        return response.content  # This is the image content (in bytes)
    else:
        print("Error response:", response.status_code, response.text)  # Print error details
        return None
