import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# --- Configuration ---
DOCS_PATH = "docs"
VECTOR_STORE_PATH = "sahayak_memory"
EMBEDDING_MODEL = "hkunlp/instructor-large"

def create_vector_store():
    """Loads PDFs from DOCS_PATH, chunks them, and saves them to a FAISS vector store."""
    
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store already exists at {VECTOR_STORE_PATH}. Skipping creation.")
        return

    print("Loading documents from 'docs' folder...")
    all_docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DOCS_PATH, filename))
            all_docs.extend(loader.load())

    if not all_docs:
        print("Error: No PDF documents found in 'docs' folder.")
        return

    print(f"Loaded {len(all_docs)} document pages.")

    # 2. Split documents into manageable chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # How big each chunk should be
        chunk_overlap=200   # How much chunks overlap
    )
    chunks = text_splitter.split_documents(all_docs)
    
    if not chunks:
        print("Error: Could not create text chunks. Check your PDF content.")
        return
        
    print(f"Created {len(chunks)} text chunks.")

    # 3. Create embeddings (turn text into vectors)
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    print("This may take a few minutes and download ~1.5GB on the first run.")
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a GPU
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create and save the vector store
    print("Creating and saving vector store... (This can also take a few minutes)")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"--- Success! ---")
    print(f"Vector store created and saved at {VECTOR_STORE_PATH}")

def test_rag_query():
    """Loads the saved vector store and performs a test query."""
    
    print("\n--- Running Test Query ---")
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vector store not found at {VECTOR_STORE_PATH}. Run create_vector_store() first.")
        return

    # Load the components needed for the query
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS
    )
    
    # Create a "retriever" to find relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 results

    # --- Test Query ---
    query = "How do I check the power LED on the Arduino Uno?"
    print(f"Test Query: '{query}'")
    
    try:
        relevant_chunks = retriever.invoke(query)
        
        if not relevant_chunks:
            print("No relevant documents found.")
            return

        print(f"\nFound {len(relevant_chunks)} relevant chunks:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i+1} (Source: {chunk.metadata.get('source', 'N/A')}, Page: {chunk.metadata.get('page', 'N/A')}) ---")
            print(chunk.page_content)
            
    except Exception as e:
        print(f"An error occurred during query: {e}")

# This part makes the script run when you call it from the terminal
if __name__ == "__main__":
    # Step 1: Run this to create your database
    create_vector_store()
    
    # Step 2: Test that your database works
    test_rag_query()
    