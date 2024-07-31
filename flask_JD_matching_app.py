from flask import Flask, request, jsonify
import os
import docx
import pandas as pd
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

    You are HRBot, an assistant to HR and expert in talent acquisition.

    Task: As a helpful and polite HRBot, select the top 3 candidates for the job based on the provided resumes and questions asked. Compulsory follow the instructions.

    Instructions:
    Accurately and contextually refer to the provided context.
    Think step-by-step and provide facts and references to sources in the response.
    Answer questions concisely and relevantly and try to keep answers less than a paragraph.
    Evaluate candidates based on relevant work experience, matching skills, education, and other relevant factors.
    Rank the candidates from 1 to 3, with 1 being the best match.
    Provide a brief justification for selecting and ranking each candidate.
    If unsure or lacking information, respond with "I don't require profiles", do not hallucinate or make up information not in the context.
    Do not give duplicate answers and identify candidates by their name.
    matching score shoud be in percentage 
    Provide the output in the following format only:
    1. Rank
    2. Full_Name
    3. Years of experience
    4. Key skills
    5. Experience details and industry sector
    6. College tier
    7. Justification of shortlisting
    8. Matching score

    Question:
    {input}"""
)


def load_text_from_docx(file):
    """Extract text from a .docx file."""
    doc = docx.Document(BytesIO(file.read()))
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def prepare_context_from_csv(directory_path):
    """Read CSV files and prepare context."""
    all_context = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory_path, filename)
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                context_entry = f"""
                Full_Name: {row.get('Full_Name', 'Not provided')}
                Years_of_Experience: {row.get('Years_of_Experience', 'Not provided')}
                Key_Skills: {row.get('Key_Skills', 'Not provided')}
                Experience_Details: {row.get('Experience_Details', 'Not provided')}
                College_Tier: {row.get('College_Tier', 'Not provided')}
                """
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

        # Ensure that response content is serializable
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
