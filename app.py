import streamlit as st
import os
import groq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from PIL import Image
import base64
import io

# --- 1. CONFIGURATION AND SETUP ---

# Load environment variables (your API key)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please create a .env file and add your key.")
    st.stop()

# Paths and model names
VECTOR_STORE_PATH = "vectorstore_faiss"
EMBEDDING_MODEL = "hkunlp/instructor-large"

# --- 2. LOAD "MEMORY" (RAG COMPONENTS) ---
# This part is unchanged and loads your vector store
@st.cache_resource
def load_rag_components():
    """Loads the pre-built vector store and embedding model."""
    print("Loading RAG components...")
    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        if not os.path.exists(VECTOR_STORE_PATH):
            st.error(f"Vector store not found at {VECTOR_STORE_PATH}. Please run setup_rag.py first.")
            return None
            
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Required for FAISS
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 chunks
        print("RAG components loaded successfully.")
        return retriever
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        return None

# --- 3. LOAD "BRAIN" & "EYES" (GROQ CLIENT) ---
# We cache the client to reuse it
@st.cache_resource
def load_groq_client():
    """Loads the Groq client."""
    print("Loading Groq client...")
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        print("Groq client loaded successfully.")
        return client
    except Exception as e:
        st.error(f"Error loading Groq client: {e}")
        return None

# --- 4. HELPER FUNCTIONS ---

def pil_to_base64(image):
    """Converts a PIL image to base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_response_from_ai(retriever, client, user_prompt, image=None):
    """
    This is the main RAG pipeline for Groq.
    It takes a prompt and optional image, gets context, and returns a response.
    """
    if not retriever or not client:
        return "Error: RAG components or Groq client not loaded."

    try:
        # 1. Retrieve (Get "Memory")
        print(f"Retrieving context for: {user_prompt}")
        context_chunks = retriever.invoke(user_prompt)
        context_text = "\n---\n".join([chunk.page_content for chunk in context_chunks])
        
        # 2. Augment (Build the system prompt)
        system_prompt = f"""
        You are 'Sahayak', a helpful AI assistant for hardware repair.
        You are an expert on the Arduino Uno.
        Your job is to guide technicians in diagnosing and repairing issues.
        Be clear, concise, and provide step-by-step instructions.
        NEVER guess. Use the provided context to answer.
        
        Here is relevant information from the technical manual:
        
        ---CONTEXT---
        {context_text}
        ---END CONTEXT---
        """
        
        # 3. Generate (Build the message list for Groq)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add image and text to the message list
        if image:
            print("Image provided. Using vision model.")
            # Convert PIL image to base64
            image_base64 = pil_to_base64(image)
            
            # This is the Groq format for multimodal messages
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            })
            # Use Groq's Llama 4 Scout (Vision) model
            model_to_use = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        else:
            print("No image. Using text model.")
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            # Use Groq's Llama 3.3 (Text) model
            model_to_use = "llama-3.3-70b-versatile"
        
        # 4. Call Groq API
        print(f"Calling Groq with model: {model_to_use}")
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_to_use,
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"An error occurred while generating the response: {e}"

# --- 5. STREAMLIT APP UI ---
# This part is mostly the same as before

# Load all the components
retriever = load_rag_components()
client = load_groq_client()

# Set page title and icon
st.set_page_config(page_title="Sahayak - AI Repair Assistant", page_icon="🤖")
st.title("🤖 Sahayak: AI Repair Assistant")
st.caption("I have read the Arduino manuals. How can I help you?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I can help you with your Arduino. "
                                        "Please describe your problem or upload a photo."}
    )

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Camera input
uploaded_image = st.camera_input("Take a picture of the hardware")

# Text chat input
user_prompt = st.chat_input("What seems to be the problem?")

# Combine image and text if both are provided
if user_prompt or uploaded_image:
    
    image_to_process = None
    if uploaded_image:
        image = Image.open(uploaded_image)
        with st.chat_message("user"):
            st.image(image, caption="You sent this image.", width=250)
        
        if not user_prompt:
            user_prompt = "Analyze this image and tell me what you see. Is anything wrong?"
        
        image_to_process = image

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

    # Generate and Display AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing manuals and schematics..."):
            if retriever and client:
                response = get_response_from_ai(retriever, client, user_prompt, image=image_to_process)
            else:
                response = "I'm sorry, my AI components failed to load. Please restart the app."
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})




