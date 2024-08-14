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

# Initialize Flask apps
app = Flask(__name__)

# Initialize LLM and prompt
llm = ChatGroq(temperature=0.2,groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template("""
#     Context:
#     {context}

#     You are HR and expert in talent acquisition.

#     Task: select all candidates which are matching with given Job discription. Compulsory follow the instructions.

#     Instructions:
#     -Accurately and contextually refer to the provided context.
#     -Answer questions concisely and relevantly and give answer in mentioned JSON format.
#     -Evaluate candidates based on relevant work experience, matching skills, education, and other relevant factors.
#     -Provide a brief justification for selecting and ranking each candidate.
#     -If unsure or lacking information, respond with "No Matching Profile", do not hallucinate or make up information not in the context.
#     -Do not give duplicate answers and identify candidates by their name.
#     -matching score shoud be in percentage.
#     -find name from Full_Name or extract from email.
#     Provide the JSON output in the following format for all resume:
#     1. Rank
#     2. Full_Name
#     3. Years of experience
#     4. Key skills
#     5. Experience details and industry sector
#     6. College name/tier
#     7. Justification of shortlisting
#     8. Matching score
                                      
#     Job discription:
#     {input}"""
# )


prompt = prompt = ChatPromptTemplate.from_template('''
            You are an AI bot designed to shortlist given resumes:{context} based on given Job discription: {input}.
            Please provide following information from resume: 
            Provide the JSON output in the following format for all resume:
            1. Rank
            2. Full_Name
            3. Years of experience
            4. Key skills
            5. Experience details and industry sector
            6. College name/tier
            7. Justification of shortlisting
            8. Matching score
            
            Instruction:
            Only provide the JSON without any explanatory text or notes.
            matching score should be in percentage format
            If any information is not available, include `null` for that field or possible to calculate it.
            Don't provide any notes or explaination on your assumptions.
            Don't add any whitespaces next line 
                        ''' )



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
                Employment_Details: {row.get('Employment_Details', 'Not provided')}
                Industry_sector: {row.get('Industry_sector', 'Not provided')}
                Technical_Skills: {row.get('Technical_Skills', 'Not provided')}
                Education_Degree: {row.get('Education_Degree', 'Not provided')}
                Profile_Summary: {row.get('Profile_Summary', 'Not provided')}
                College_Name: {row.get('College_Name', 'Not provided')}
                Profile_Summary: {row.get('Profile_Summary', 'Not provided')}
                Experience_Details: {row.get('Experience_Details', 'Not provided')}
                Project_Summary: {row.get('Project_Summary', 'Not provided')}
                Certifications: {row.get('Certifications', 'Not provided')}
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500000, chunk_overlap=500)
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

    directory_path = 'C:/Users/arvindchavan/Downloads/RAG/Final_RAG/'  # Update this path to your actual directory
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