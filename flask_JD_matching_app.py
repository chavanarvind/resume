import os
import pandas as pd
from flask import Flask, request, jsonify
import docx
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize LLM and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
    Context:
    {context}

    You are HRBot, an assistant to HR and an expert in talent acquisition.

    Task: Based on the provided resumes, select and rank the top 10 candidates for the job. Follow the instructions below:

    Instructions:
    1. Refer accurately and contextually to the provided context.
    2. Provide step-by-step reasoning and cite specific facts and references from the context.
    3. Answer questions concisely and relevantly. Keep responses brief, ideally under a paragraph.
    4. Evaluate candidates based on:
       - Relevant work experience
       - Matching skills
       - Educational background
       - Any other pertinent factors
    5. Rank the candidates from 1 to 10, with 1 being the best match.
    6. For each candidate, provide:
       - Full Name
       - Years of Experience
       - Key Skills
       - Experience Details and Industry Sector
       - College Tier
       - Justification for Ranking
       - Matching Score (in percentage)
    7. If you lack sufficient information or are unsure, respond with "I don't require profiles." Avoid making assumptions or hallucinating information.
    8. Do not repeat answers. Clearly identify each candidate by their name.

    Output Format:
    1. Rank
    2. Full_Name
    3. Years of Experience
    4. Key Skills
    5. Experience Details and Industry Sector
    6. College/Tier
    7. Justification for Shortlisting
    8. Matching Score (percentage)

    Question:
    {input}"""
)


def load_text_from_docx(file):
    """Extract text from a .docx file."""
    doc = docx.Document(BytesIO(file.read()))
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def remove_columns_from_csv(file_path, columns_to_remove):
    """Removes specified columns from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=columns_to_remove, errors='ignore')  # `errors='ignore'` ensures no error if column not found
        df.to_csv(file_path, index=False)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def prepare_context_from_csv(directory_path):
    """Read CSV files, remove specified columns, and prepare context."""
    all_context = []
    columns_to_remove = ['Address', 'Email_ID', 'Contact_Number']
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory_path, filename)
            
            # Remove specified columns from the CSV file
            remove_columns_from_csv(csv_path, columns_to_remove)
            
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                # Construct context entry by including all columns dynamically
                context_entry = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
                all_context.append(context_entry)

    if not all_context:
        raise ValueError("No context data extracted from CSV documents.")
    
    return "\n".join(all_context)

def vector_embedding(directory_path):
    """Initialize vector embeddings and create a vector store from documents in a directory."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory_path, filename)
            df = pd.read_csv(csv_path)

            # Collect all text data from all columns
            for index, row in df.iterrows():
                row_text = " ".join([str(row[col]) for col in df.columns if pd.notnull(row[col])])
                all_texts.append(row_text)
    
    if not all_texts:
        raise ValueError("No text extracted from CSV documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300000, chunk_overlap=100)
    final_documents = text_splitter.create_documents(all_texts)
    if not final_documents:
        raise ValueError("No document chunks created.")
    
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

@app.route('/process', methods=['POST'])
def process():
    if 'doc' not in request.files:
        return jsonify({"error": "No document file part"}), 400
    
    file = request.files['doc']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({"error": "Invalid file format. Please upload a .docx file."}), 400

    directory_path = 'C:/Users/Arvi/Downloads/resume-main/'  # Update this path to your actual directory
    try:
        vectors = vector_embedding(directory_path)
        
        document_text = load_text_from_docx(file)
        if not document_text:
            return jsonify({"error": "No text extracted from the document."}), 400
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Prepare context from CSV files
        context = prepare_context_from_csv(directory_path)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'context': context, 'input': document_text})
        elapsed_time = time.process_time() - start

        response_data = {
            "response": response.get('answer', ''),
            "response_time": elapsed_time,
            "context": [chunk.page_content for chunk in response.get("context", [])]  # Convert context to serializable format
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
