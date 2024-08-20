from flask import Flask, request, jsonify
import pandas as pd
import os
import docx
from io import BytesIO
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts using BERT embeddings."""
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return int(round(similarity[0][0] * 100))  # Convert numpy float32 to Python float

def load_text_from_docx(file):
    """Extract text from a .docx file."""
    doc = docx.Document(BytesIO(file.read()))
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def calculate_scores_for_resumes(csv_path, job_description):
    """Calculate similarity scores for each resume in the CSV file."""
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        # Combine relevant fields into a single text for comparison
        resume_text = f"""
        Full_Name: {row.get('Full_Name', 'Not provided')}
        Profile_Summary: {row.get('Profile_Summary', 'Not provided')}
        Key_Skills: {row.get('Key_Skills', 'Not provided')}
        """
        similarity_score = calculate_similarity(resume_text, job_description)
        results.append({
            "Full_Name": row.get('Full_Name', 'Not provided'),
            "similarity_score": similarity_score
        })

    # Sort results by similarity score in descending order
    results_sorted = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    return results_sorted

@app.route('/process', methods=['POST'])
def process():
    """Process the uploaded job description and calculate similarity scores for resumes."""
    if 'doc' not in request.files:
        return jsonify({"error": "No document file part"}), 400

    file = request.files['doc']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.docx'):
        return jsonify({"error": "Invalid file format. Please upload a .docx file."}), 400

    directory_path = 'C:/Users/Arvi/Downloads/resume-main/'  # Update this path to your actual directory

    try:
        document_text = load_text_from_docx(file)
        if not document_text:
            return jsonify({"error": "No text extracted from the document."}), 400

        # Calculate similarity scores for each resume in CSV files
        all_scores = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(directory_path, filename)
                scores = calculate_scores_for_resumes(csv_path, document_text)
                all_scores.extend(scores)

        response_data = {"results": all_scores}

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
