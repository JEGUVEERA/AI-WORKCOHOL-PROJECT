import streamlit as st
import os
import logging
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_key_configuration():
    """Test API key configuration from both .env and Streamlit secrets"""
    st.title("🧪 API Key Configuration Test")
    
    # Try loading from Streamlit secrets
    api_key_from_secrets = None
    try:
        api_key_from_secrets = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Successfully loaded API key from Streamlit secrets")
        st.write("API key from secrets:", api_key_from_secrets[:5] + "..." if api_key_from_secrets else "None")
    except Exception as e:
        st.warning(f"⚠️ Could not load API key from Streamlit secrets: {str(e)}")
    
    # Try loading from .env file
    load_dotenv()
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    if api_key_from_env:
        st.success("✅ Successfully loaded API key from .env file")
        st.write("API key from .env:", api_key_from_env[:5] + "..." if api_key_from_env else "None")
    else:
        st.warning("⚠️ Could not load API key from .env file")
    
    # Determine which API key to use
    api_key = api_key_from_secrets or api_key_from_env
    
    if not api_key:
        st.error("❌ No API key found in either Streamlit secrets or .env file")
        return None
    
    return api_key

def test_gemini_model(api_key):
    """Test Gemini model initialization and simple generation"""
    st.title("🤖 Gemini Model Test")
    
    if not api_key:
        st.error("❌ Cannot test Gemini model without an API key")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        st.success("✅ Successfully initialized Gemini model")
        
        with st.spinner("Testing model with a simple prompt..."):
            prompt = "Write a single short sentence about marketing."
            response = model.generate_content(prompt)
            
            st.success("✅ Successfully generated content from Gemini model")
            st.write("Test response:", response.text)
        
        return True
    except Exception as e:
        st.error(f"❌ Error testing Gemini model: {str(e)}")
        return False

def test_streamlit_functionality():
    """Test basic Streamlit functionality"""
    st.title("🔍 Streamlit Functionality Test")
    
    try:
        # Test session state
        if "test_value" not in st.session_state:
            st.session_state.test_value = "Test value"
        
        st.success(f"✅ Session state working: {st.session_state.test_value}")
        
        # Test widgets
        st.write("Testing Streamlit widgets...")
        test_input = st.text_input("Test input:", "Hello Streamlit!")
        st.write("Input value:", test_input)
        
        # Test columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("Column 1")
        with col2:
            st.write("Column 2")
            
        st.success("✅ Basic Streamlit functionality is working")
        return True
    except Exception as e:
        st.error(f"❌ Error testing Streamlit functionality: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="AI Marketing App Test", layout="wide")
    
    st.header("📊 AI Marketing App Deployment Test")
    st.write("This script tests essential functionality needed for deployment.")
    
    # Test API key configuration
    api_key = test_api_key_configuration()
    
    # Test Gemini model if API key is available
    if api_key:
        model_success = test_gemini_model(api_key)
    else:
        model_success = False
        st.error("❌ Skipping Gemini model test due to missing API key")
    
    # Test Streamlit functionality
    streamlit_success = test_streamlit_functionality()
    
    # Overall status
    st.header("📋 Overall Test Results")
    if api_key and model_success and streamlit_success:
        st.success("✅ All tests passed! Your app is ready for deployment.")
    else:
        st.error("❌ Some tests failed. Please fix the issues before deployment.")
    
    st.write("Test completed at:", time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()

