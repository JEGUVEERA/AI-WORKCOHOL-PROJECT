import json
import requests
import google.generativeai as genai
import os
import streamlit as st
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv




# --- Configuration ---
load_dotenv()


#client = genai.Client(api_key=GEMINI_API_KEY)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()

if not HF_API_KEY:
    st.warning("Warning: HF_API_KEY is missing. Image and video generation may not work.")

# --- Initialize Gemini ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- The rest of your code remains unchanged ---

# --- Text Generation Function ---

#def generate_text_content(ingredients: str) -> str:
    # Modified prompt instructing the model to output only the content
   # prompt = (
      #  "You are an AI agent for Marketing. "
       # "Generate only marketing content (slogans, ad copy, campaign ideas) based on the following input: "
       # f"{ingredients}. "
        #"Do include any extra commentary only related to prompt details and Do not include codes for any content unless I ask for codes."
    #)
   # try:
      #  resp: ReplayResponse = client.models.generate_content(
        #    model="gemini-2.0-flash",
        #    contents=prompt
      #  )
      #  return resp.text
    #except Exception as e:
       # st.error(f"Error generating text content: {e}")
        #return "Error generating content."
############################################################################################################   

# --- Text Generation Function ---     
    
def generate_text_content(ingredients: str) -> str:
    prompt = (
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas) based on the following input: "
        
        "Do include any extra commentary only related to prompt details and Do not include codes for any content unless I ask for codes."
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information."
    f"{ingredients}."

    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."
  


# --- Concurrent Image Generation Function ---
def generate_images(prompt: str, num_images: int = 4):
    HF_MODEL = "stabilityai/stable-diffusion-2"
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def fetch_image():
        try:
            payload = json.dumps({
                "inputs": prompt,
                "options": {"wait_for_model": True}
            })
            response = requests.post(api_url, headers=headers, data=payload)
            if response.status_code != 200:
                st.error(f"Error generating image: {response.content}")
                return None
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None

    images = []
    # Use a realistic maximum number of workers (e.g., up to 10 or so)
    max_workers = min(num_images, 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image) for _ in range(num_images)]
        for future in as_completed(futures):
            img = future.result()
            if img is not None:
                images.append(img)
    return images




# --- Streamlit UI ---
st.title("AI for Marketing")
st.markdown("**Description:** Generate marketing content and images based on your prompt.")


# --- Voice and Tone Customization UI ---
st.header("Voice and Tone Customization")
st.markdown("Select the tone and voice for your content.")
st.markdown("")

# Tone Selection
tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"], key="tone_selection")

# Voice Selection
voice = st.selectbox("Select Voice", ["Male", "Female", "Neutral"], key="voice_selection")

st.markdown("---")


# --- Marketing Content Generation UI ---


st.header("Marketing Content Generation")
ingredients = st.text_input("Enter ingredients for marketing content:", placeholder="e.g., marketing slogans, ad copy, campaign ideas")
if st.button("Generate Marketing Content"):
    if ingredients:
        # Adjust prompt based on selected tone
        adjusted_prompt = f"Generate {tone.lower()} marketing content based on the following ingredients: {ingredients}"
        content = generate_text_content(ingredients)
        st.subheader("Generated Content")
        st.markdown(content)
        st.download_button("Download Content", content, file_name="content.txt")
    else:
        st.error("Please enter ingredients to generate content.")

st.markdown("---")


# --- Image Generation UI ---
st.header("Image Generation")
image_prompt = st.text_input("Enter an image prompt:", placeholder="e.g., futuristic ad design")
num_images = st.number_input("Number of Images", min_value=0, max_value=100, value=1)
if st.button("Generate Images"):
    if image_prompt:
        with st.spinner("Generating images..."):
            images = generate_images(image_prompt, num_images=int(num_images))
        if images:
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 2].image(img, use_container_width=True)
    else:
        st.error("Please enter an image prompt.")

st.markdown("---")




# --- Chatbot Integration UI ---
st.header("AI Chatbot Integration")

# User inputs for chatbot
chatbot_name = st.text_input("Enter Chatbot Name:")
tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"])
user_message = st.text_area("Type your message to the chatbot:")

if st.button("Send Message"):
    if chatbot_name and user_message:
        # Generate a response using the selected tone
        response = generate_text_content(user_message, tone)
        st.subheader(f"{chatbot_name}'s Response:")
        st.markdown(response)
    else:
        st.error("Please enter both a chatbot name and a message.")

st.markdown("---")






# --- Email Marketing Content Generation ---
def generate_email_content(subject: str, body: str) -> str:
    prompt = f"Generate an email with subject '{subject}' and body: '{body}'"
    try:
        response = model.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
        )
        return response.text

    except Exception as e:
        st.error(f"Error generating email content: {e}")
        return "Error generating email."

# --- Streamlit UI for Email Marketing ---
st.header("Email Marketing Content Generation")
email_subject = st.text_input("Enter email subject:")
email_body = st.text_area("Enter email body:")
if st.button("Generate Email Content"):
    if email_subject and email_body:
         # Adjust prompt based on selected tone
        adjusted_email_prompt = f"Generate a {tone.lower()} email with subject '{email_subject}' and body: '{email_body}'"
        email_content = generate_email_content(email_subject, email_body)
        st.subheader("Generated Email Content")
        st.markdown(email_content)
    else:
        st.error("Please enter both subject and body for the email.")


st.markdown("---")


# --- Social Media Post Generation ---
def generate_social_media_post(platform: str, content: str) -> str:
    prompt = f"Generate a {platform} post for the following content: {content}"
    try:
        response = model.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
        )
        return response.text

    except Exception as e:
        st.error(f"Error generating social media post: {e}")
        return "Error generating post."

# --- Streamlit UI for Social Media Post Generation ---
st.header("Social Media Post Generation")
platform = st.selectbox("Select Platform", ["Twitter", "Instagram", "Facebook"])
social_content = st.text_input("Enter content for social media post:")
if st.button("Generate Social Media Post"):
    if social_content:
        post = generate_social_media_post(platform, social_content)
        st.subheader("Generated Social Media Post")
        st.markdown(post)
    else:
        st.error("Please enter content for the social media post.")

st.markdown("---")


# --- Video Generation ---      

def generate_video(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt}
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json().get('generated_video_url', 'Video generated successfully.')
        else:
            error_msg = response.text
            # Check if error indicates unsupported task variant
            if "unknown variant" in error_msg:
                st.error("The current model endpoint does not support video generation in this format. Please check the model documentation or consider an alternative approach.")
            else:
                st.error(f"Error generating video: {error_msg}")
            return "Error generating video."
    except Exception as e:
        st.error(f"Error generating video: {e}")
        return "Error generating video."

# --- Video Prompt Input ---
video_prompt = st.text_input("Enter video prompt:")

# --- Video Generation Button ---
if st.button("Generate Video", key="generate_video_1"):  
    if video_prompt:
        video = generate_video(video_prompt)
        st.subheader("Generated Video")
        st.video(video)
    else:
        st.error("Please enter a video prompt.")

# --- Video Style Selection ---
video_style = st.selectbox("Select Video Style", ["Animated", "Realistic", "Cartoonish"])

# --- Styled Video Generation Button ---
if st.button("Generate Video", key="generate_video_2"):  
    if video_prompt:
        styled_prompt = f"{video_style} video: {video_prompt}"
        video = generate_video(styled_prompt)
        st.subheader("Generated Video")
        st.video(video)
    else:
        st.error("Please enter a video prompt.")





<<<<<<< HEAD
################## *****************            END               *** *********# ################
=======
################## *****************            END               *** *********# ################
>>>>>>> 118d33c (Removed unnecessary files)
