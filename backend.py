import os
import groq
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import base64
import io

# --- CONFIGURATION ---
# Load environment variables
load_dotenv()

# Paths and Models
VECTOR_STORE_PATH = "vectorstore_faiss"
EMBEDDING_MODEL = "hkunlp/instructor-large"

def get_api_key():
    """Retrieves the API Key safely."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        # Check Streamlit secrets if .env fails (for cloud deployment)
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
    return key

# --- COMPONENT LOADING ---
@st.cache_resource
def load_rag_components():
    """Loads the pre-built vector store and embedding model."""
    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        if not os.path.exists(VECTOR_STORE_PATH):
            return None
            
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        print(f"Error loading RAG: {e}")
        return None

@st.cache_resource
def load_groq_client():
    """Loads the Groq client."""
    key = get_api_key()
    if not key:
        return None
    try:
        client = groq.Groq(api_key=key)
        return client
    except Exception as e:
        print(f"Error loading Groq: {e}")
        return None

# --- HELPER FUNCTIONS ---
def pil_to_base64(image):
    """Converts a PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def query_llm(retriever, client, user_prompt, image=None):
    """
    The Core Logic:
    1. RAG Retrieval
    2. Prompt Engineering
    3. API Call
    """
    if not retriever or not client:
        return "System Error: Critical components (Memory or Brain) failed to load."

    try:
        # 1. Retrieve Context from Memory
        context_chunks = retriever.invoke(user_prompt)
        context_text = "\n---\n".join([chunk.page_content for chunk in context_chunks])
        
        # 2. Define the Expert Persona
        system_prompt = f"""
        You are 'Sahayak', an expert industrial AI assistant for IIoT contexts.
        You are analyzing technical manuals for industrial hardware (Arduino, PLCs, Sensors).
        
        Mission: Help the technician diagnose issues, verify specs, or perform repairs.
        Style: Professional, technical, concise, and step-by-step.
        Constraint: Strictly use the provided context. If the answer isn't in the context, state that you don't know.
        
        ---TECHNICAL CONTEXT---
        {context_text}
        ---END CONTEXT---
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 3. Construct Message Payload
        if image:
            # Vision Mode
            image_base64 = pil_to_base64(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            })
            model_to_use = "meta-llama/llama-3.2-11b-vision-preview"
        else:
            # Text Mode
            messages.append({"role": "user", "content": user_prompt})
            model_to_use = "llama-3.2-11b-vision-preview"
        
        # 4. Generate Response
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_to_use,
            temperature=0.2 # Low temperature for factual accuracy
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"An error occurred during processing: {e}"