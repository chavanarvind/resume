import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json

load_dotenv()

# Load the GROQ And OpenAI API KEY 
groq_api_key = "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"
GOOGLE_API_KEY = "AIzaSyCIsJ2CuMm-6-v2k6raO8wnRpi2we_wA4A"

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
    Context:
    {context}
 
    You are HRBot, an assistant to HR and expert in talent acquisition.
 
    Task: As a helpful and polite HRBot, select the best candidate for the job based on the provided resumes and questions asked. Compulsory follow the instructions.
 
    Instructions:
    Accurately and contextually refer to the provided context.
    Think step-by-step and provide facts and references to sources in the response.
    Answer questions concisely and relevantly and try to keep answers less than a paragraph.
    Evaluate candidates based on relevant work experience, matching skills, education, and other relevant factors.
    Provide a brief justification for selecting a candidate.
    If unsure or lacking information, respond with "I don't know", do not hallucinate or make up information not in the context.
    Do not give duplicate answers and identify candidates by their name
    Provide the output in the following format only:
    1. Name candidate
    2. Years of experience
    3. Key skills
    4. Experience details and industry sector
    5. College tier
    6. Justification of shortlisting
    7. Matching score
   
    Question:
    {input}"""
)

def load_json_files(directory_path):
    """Load JSON files from a directory."""
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), 'r') as file:
                documents.append(json.load(file))
    return documents

def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load JSON documents
        st.session_state.docs = load_json_files("C:/Users/Arvi/Downloads/resume-main/")  # Specify your directory containing JSON files
        if not st.session_state.docs:
            st.error("No JSON documents found. Please check the directory path.")
            return
        
        # Extract text from JSON documents
        all_texts = [json.dumps(doc) for doc in st.session_state.docs]  # Adjust as needed to extract specific fields
        if not all_texts:
            st.error("No text extracted from JSON documents. Please check the JSON structure.")
            return
        
        # Document loading and splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=25)
        st.session_state.final_documents = st.session_state.text_splitter.create_documents(all_texts[:10])
        if not st.session_state.final_documents:
            st.error("No document chunks created. Please check the text splitting configuration.")
            return
        
        # Create vector embeddings
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        except Exception as e:
            st.error(f"Error creating vector embeddings: {str(e)}")
            return

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time :", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    except Exception as e:
        st.error(f"An error occurred during retrieval: {str(e)}")
