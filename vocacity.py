import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai  # Import Google Generative AI
import base64
from transformers import pipeline

# Set up Google Generative AI API key
genai.configure(api_key="AIzaSyBSHJUV79V6gQeoXFY__29tjk0p-h_0u68")  

# Initialize recognizer
recognizer = sr.Recognizer()

# Emotion Detection Model
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Function to detect emotion
def detect_emotion(text):
    emotion = emotion_model(text)[0]['label']
    return emotion

# Function to listen to customer
def listen_to_customer():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Customer said: {text}")
            return text
        except Exception as e:
            st.error(f"Speech Recognition Error: {str(e)}")
            return None

# Function to process text
def process_text(customer_input):
    if customer_input:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
            response = model.generate_content(customer_input)
            return response.text
        except Exception as e:
            return f"Error in AI response: {str(e)}"
    else:
        return "Sorry, I didn't catch that. Could you please repeat?"

# Function to convert text to speech
def text_to_speech(text, voice_option, language):
    lang_code = {"English": "en", "Spanish": "es", "French": "fr", "Hindi": "hi"}.get(language, "en")
    tts = gTTS(text=text, lang=lang_code, tld='com' if voice_option == "Male" else 'co.uk')
    file_path = "response.mp3"
    tts.save(file_path)
    return file_path

# Function to autoplay audio
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(audio_html, unsafe_allow_html=True)

# Main function
def main():
    st.title("Vocacity AI Voice Agent 🎙️")
    st.sidebar.header("Settings")
    
    # User settings
    language = st.sidebar.selectbox("Choose Language:", ["English", "Spanish", "French", "Hindi"])
    voice_option = st.sidebar.selectbox("Choose AI Voice:", ["Male", "Female"])
    clear_chat = st.sidebar.button("🗑️ Clear Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Text Input
    user_text_input = st.text_input("Type your query here:", "")
    
    # Voice Input Button
    if st.button("🎙️ Speak"):
        customer_input = listen_to_customer()
    else:
        customer_input = user_text_input if user_text_input.strip() else None
    
    if customer_input:
        emotion = detect_emotion(customer_input)
        ai_response = process_text(customer_input)
        st.session_state.chat_history.append((customer_input, ai_response))
        
        st.write(f"**AI Response:** {ai_response} (Emotion: {emotion})")
        
        # Convert response to speech and autoplay it
        audio_file = text_to_speech(ai_response, voice_option, language)
        autoplay_audio(audio_file)
        os.remove(audio_file)
    
    # Display chat history
    st.write("### Chat History")
    for user, ai in st.session_state.chat_history[-5:]:
        st.write(f"👤 {user}")
        st.write(f"🤖 {ai}")
    
    # Clear chat
    if clear_chat:
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
