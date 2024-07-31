import os
import json
import zipfile
from flask import Flask, jsonify, request
from groq import Groq
import pandas as pd
import glob

class ResumeProcessor:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.allowed_extensions = {'pdf', 'docx', 'zip'}

    def extract_text_from_docx(self, docx_file):
        try:
            import docx
            doc = docx.Document(docx_file)
            text = [para.text for para in doc.paragraphs]
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from .docx: {e}")

    def extract_text_from_pdf(self, pdf_file):
        try:
            from PyPDF2 import PdfReader
            pdf_text = []
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text.append(page.extract_text())
            return '\n'.join(pdf_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from .pdf: {e}")

    def extract_text_from_resume(self, file_path):
        if file_path.endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        elif file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def process_resumes(self, directory_path):
        resumes_text = {}
        file_pattern = os.path.join(directory_path, '*')
        files = glob.glob(file_pattern)

        for file_path in files:
            if os.path.isfile(file_path) and self.allowed_file(file_path):
                print(f"Processing file: {file_path}")
                if file_path.endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        for zip_info in zip_ref.infolist():
                            if zip_info.filename.endswith(('.docx', '.pdf')):
                                with zip_ref.open(zip_info) as extracted_file:
                                    text = self.extract_text_from_resume(extracted_file)
                                    resumes_text[zip_info.filename] = text
                else:
                    text = self.extract_text_from_resume(file_path)
                    resumes_text[os.path.basename(file_path)] = text

        return self.extract_information(resumes_text)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def extract_information(self, resumes_text):
        extracted_data = []
        for filename, text in resumes_text.items():
            prompt_template = f'''
            You are an AI bot designed to act as a professional for parsing resumes. 
            You are given with resume and your job is to extract the following information from the resume {text} in json just that dont give additional text in the begining and end just this info: 
            1. Full_Name
            2. Address
            2. Email_ID
            3. Contact_Number
            3. Employment_Details
            4. Industry_sector
            5. Technical_Skills
            6. Education_Degree
            7. College_Name
            8. Profile_Summary
            9. Project_Summary
            10. Certifications
            Give the extracted information in json format only. Provide null if information not available in the resume.
            and this is resume {text} and dont add additional text in begining and end just extract json and give complete information and dont include
            Here is the extracted information in json format from resume details above provided:
            '''

            client = Groq(api_key=self.groq_api_key)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt_template,
                    }
                ],
                temperature=0.4,
                model="llama-3.1-8b-instant"
            )

            response_content = chat_completion.choices[0].message.content
            try:
                data = json.loads(response_content.strip())
                extracted_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        
        return extracted_data

app = Flask(__name__)
app.config['GROQ_API_KEY'] = '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'
processor = ResumeProcessor(app.config['GROQ_API_KEY'])

@app.route('/process_directory', methods=['POST'])
def process_directory():
    data = request.get_json()
    directory_path = data.get('directory_path')

    if not directory_path or not os.path.isdir(directory_path):
        return jsonify({"error": "Invalid directory path"}), 400

    try:
        processed_data = processor.process_resumes(directory_path)
        return jsonify(processed_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
