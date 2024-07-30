import json
import zipfile
from groq import Groq
import pandas as pd
import glob

def extract_text_from_docx(docx_file):
    try:
        import docx
        doc = docx.Document(docx_file)
        text = [para.text for para in doc.paragraphs]
        return '\n'.join(text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from .docx: {e}")

def extract_text_from_pdf(pdf_file):
    try:
        from PyPDF2 import PdfReader
        pdf_text = []
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text.append(page.extract_text())
        return '\n'.join(pdf_text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from .pdf: {e}")

def extract_text_from_resume(file):
    if file.endswith('.docx'):
        return extract_text_from_docx(file)
    elif file.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        raise ValueError(f"Unsupported file format: {file}")

def process_resumes(resumes_text):
    json_file = 'data.json'
    
    # Load existing data from the JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as jsonfile:
            existing_data = json.load(jsonfile)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
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

        groq_api_key = "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"

        client = Groq(api_key=groq_api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_template,
                }
            ],
            temperature=0.4,
            model = "llama-3.1-8b-instant"
            #model="llama3-70b-8192",
        )

        response_content = chat_completion.choices[0].message.content
        try:
            data = json.loads(response_content.strip())  # Strip whitespace before parsing
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            
        # Check for duplicates
        if (data["Email_ID"] not in [d.get("Email_ID") for d in existing_data]):
            existing_data.append(data)
        else:
            print("Data already exists!!")

    with open(json_file, 'w+', encoding='utf-8') as jsonfile:
        json.dump(existing_data, jsonfile, indent=4)
        jsonfile.seek(0)
        data = json.loads(jsonfile.read())
    
    df = pd.json_normalize(data)
    df.to_csv("data.csv",encoding='utf-8',index=False)
    print("Data added!!")

def main():
    resumes_text = {}
    
    directory_path = 'C:/Users/arvindchavan/Downloads/RAG/Final_RAG/resume'
    file_pattern = (f"{directory_path}\*.pdf")  # Adjust the pattern as needed
    filez = glob.glob(file_pattern)
    
    for file in filez:
        file = file.strip()
        print(file)
        if file.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.endswith(('.docx', '.pdf')):
                        with zip_ref.open(zip_info) as extracted_file:
                            text = extract_text_from_resume(extracted_file)
                            resumes_text[zip_info.filename] = text
        else:
            text = extract_text_from_resume(file)
            resumes_text[file] = text
    process_resumes(resumes_text)

if __name__ == "__main__":
    main()
    

"""from tkinter import Tk
from tkinter.filedialog import askopenfilenames

def get_filenames_from_user(message):
    root = Tk()
    root.withdraw()
    filenames = askopenfilenames(title=message)
    return filenames

# Example usage:
selected_files = get_filenames_from_user('Select files')
print(selected_files)"""