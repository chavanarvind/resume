import os
import json
import zipfile
from flask import Flask, jsonify, request
from groq import Groq
import pandas as pd
import glob
import paramiko
from scp import SCPClient

class ResumeProcessor:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.allowed_extensions = {'pdf', 'docx', 'zip'}
        self.remote_folder = '/home/azureuser/resume_analysis/resume'  # Fixed remote folder
        self.hostname = '13.71.112.51'  # Fixed VM IP address
        self.port = 22  # Default SSH port
        self.username = 'azureuser'  # Fixed SSH username
        self.password = 'Shantabai@12'  # Fixed SSH password

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

        extracted_data = self.extract_information(resumes_text)
        self.save_to_json(extracted_data, 'extracted_data.json')
        self.save_to_csv(extracted_data, 'extracted_data.csv')
        return extracted_data

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def extract_information(self, resumes_text):
        extracted_data = []
        client = Groq(api_key=self.groq_api_key)

        for filename, text in resumes_text.items():
            prompt_template = f'''
            You are an AI bot designed to extract contextual information from resumes.
            Please extract the following information from this resume: 
            1. Full_Name
            2. Year_of_Experience
            3. Address
            4. Email_ID
            5. Contact_Number
            6. Employment_Details
            7. Industry_sector
            8. Technical_Skills
            9. Education_Degree
            10. College_Name
            11. Profile_Summary
            12. Project_Summary
            13. Certifications

            Resume text:
            {text}

            Only provide the JSON without any explanatory text or notes. If any information is not available, include `null` for that field or possible to calculate it.
            Don't provide any notes or explanation on your assumptions.
            Calculate Year_of_Experience using the total duration of work experience if it's not directly available.
            Industry_sector is not directly mentioned in resume, use your knowledge and decide it.
            Profile_Summary must be based on overall work related experience and it should not be more than one paragraph.
            '''

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

            # Debugging step: Print the raw response content
            print(f"Raw API response for file '{filename}': {response_content}")

            try:
                data = json.loads(response_content.strip())
                extracted_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Response content: {response_content}")

        return extracted_data

    def save_to_json(self, data, json_path):
        """Save extracted data to a JSON file."""
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Extracted data saved to {json_path}")

    def save_to_csv(self, data, csv_path):
        """Save extracted data to a CSV file."""
        if data:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Extracted data saved to {csv_path}")
        else:
            print("No data available to save.")

    def scp_upload_folder(self, local_folder):
        """Upload all files from local folder to remote VM folder using SCP."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)
            with SCPClient(ssh.get_transport()) as scp:
                for file_name in os.listdir(local_folder):
                    local_path = os.path.join(local_folder, file_name)
                    if os.path.isfile(local_path):
                        remote_path = os.path.join(self.remote_folder, file_name)
                        scp.put(local_path, remote_path)
                        print(f"File {local_path} uploaded to {remote_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            ssh.close()

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

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    data = request.get_json()
    local_folder = data.get('local_folder')

    if not os.path.isdir(local_folder):
        return jsonify({"error": "Invalid local folder path"}), 400

    try:
        processor.scp_upload_folder(local_folder)
        return jsonify({"message": "Files uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
