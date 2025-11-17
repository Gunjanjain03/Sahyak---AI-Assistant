import streamlit as st
from PIL import Image
import backend  # Importing your new logic file

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sahayak - IIoT Assistant",
    page_icon="🏭",
    layout="wide"
)

# --- 2. INITIALIZATION ---
# Load the heavy AI components using the backend functions
retriever = backend.load_rag_components()
client = backend.load_groq_client()

# Check if API Key is missing
if not client:
    st.error("🚨 GROQ_API_KEY not found. Please check your .env file or Streamlit Secrets.")
    st.stop()

# --- 3. SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=80)
    st.title("Sahayak IIoT")
    st.caption("Industrial Repair Assistant")
    
    st.markdown("---")
    
    # Status Indicators
    if retriever:
        st.success("Memory (RAG): Online")
    else:
        st.error("Memory (RAG): Offline")
        
    if client:
        st.success("Brain (AI): Online")
    
    st.markdown("---")
    st.markdown("**Knowledge Base:** Arduino, ESP32, BME280 Sensor")
    
    # Reset Button
    if st.button("🔄 Start New Job", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
st.header("🤖 Industrial Diagnostics Interface")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Sahayak System initialized. Ready for visual inspection or technical query."}
    )

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. USER INPUT HANDLING ---
# Create layout for inputs
col1, col2 = st.columns([3, 1])

with col2:
    uploaded_image = st.camera_input("Scan Component")

with col1:
    user_prompt = st.chat_input("Describe the issue or ask for specs...")

# --- 6. PROCESSING LOGIC ---
if user_prompt or uploaded_image:
    
    image_to_process = None
    
    # Handle Image Input
    if uploaded_image:
        image = Image.open(uploaded_image)
        # Display the user's image in chat
        with st.chat_message("user"):
            st.image(image, caption="Visual Scan Uploaded", width=300)
        
        # Set default prompt if user didn't type anything
        if not user_prompt:
            user_prompt = "Analyze this visual scan. Identify the component and any visible anomalies."
        
        image_to_process = image

    # Handle Text Input
    if user_prompt:
        # Only show text if we didn't just show an image caption (avoids duplicates)
        if not uploaded_image: 
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

    # Generate & Display Response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing technical schematics..."):
            # CALL THE BACKEND FUNCTION
            response = backend.query_llm(retriever, client, user_prompt, image=image_to_process)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})