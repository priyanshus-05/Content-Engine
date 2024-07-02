import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
import streamlit as st
import time

# Replace this with your actual Pinecone API key
api_key = 'bbc1194c-5098-46cc-8a8d-3ec6cec845eb'  # Make sure this is the correct key
pc = Pinecone(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

def generate_embeddings(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    sentences = text.split('\n')
    embeddings = model.encode(sentences)
    return embeddings

def store_embeddings(embeddings, index_name='pdf-embeddings', batch_size=100):
    # Check if index already exists
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]),
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Change to a supported region
            )
        )
    index = pc.Index(index_name)
    
    # Batch the vectors to avoid payload size errors
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        vectors = [(str(i + j), embedding.tolist()) for j, embedding in enumerate(batch)]
        
        retries = 3
        while retries > 0:
            try:
                index.upsert(vectors)
                break
            except Exception as e:
                st.write(f"Error upserting vectors: {e}")
                retries -= 1
                time.sleep(5)
                if retries == 0:
                    st.write(f"Failed to upsert vectors after multiple attempts: {e}")
                    raise

def query_vector_store(query, model_name='all-MiniLM-L6-v2', index_name='pdf-embeddings'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0]
    index = pc.Index(index_name)
    results = index.query(vector=query_embedding.tolist(), top_k=5)
    return results

def generate_insights(query, model_name='gpt2'):  # Changed to a valid model identifier
    generator = pipeline('text-generation', model=model_name)
    response = generator(query, max_length=100)
    return response[0]['generated_text']

def main():
    st.title("Content Engine")
    st.write("Starting PDF processing...")

    pdf_files = [
        r'C:\Users\HP\OneDrive\Desktop\Alemeno\goog-10-k-2023.pdf',
        r'C:\Users\HP\OneDrive\Desktop\Alemeno\tsla-20231231-gen.pdf',
        r'C:\Users\HP\OneDrive\Desktop\Alemeno\uber-10-k-2023.pdf'
    ]
    texts = []
    for pdf in pdf_files:
        st.write(f"Processing {pdf}...")
        text = extract_text_from_pdf(pdf)
        texts.append(text)
        st.write(f"Extracted text from {pdf}")

    embeddings = []
    for i, text in enumerate(texts):
        st.write(f"Generating embeddings for PDF {i+1}...")
        embedding = generate_embeddings(text)
        embeddings.append(embedding)
        st.write(f"Generated embeddings for PDF {i+1}")

    for i, embedding in enumerate(embeddings):
        st.write(f"Storing embeddings for PDF {i+1}...")
        store_embeddings(embedding, index_name=f'pdf-embeddings-{i}')
        st.write(f"Stored embeddings for PDF {i+1}")

    query = st.text_input("Enter your query:")
    if query:
        st.write("Querying vector store...")
        results = [query_vector_store(query, index_name=f'pdf-embeddings-{i}') for i in range(len(pdf_files))]
        st.write("Generating insights...")
        insights = [generate_insights(query) for _ in range(len(pdf_files))]
        st.write("Results:", results)
        st.write("Insights:", insights)

if __name__ == "__main__":
    main()
