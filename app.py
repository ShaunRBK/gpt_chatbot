import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from flask import Flask, render_template, request, jsonify

# Additional imports for file processing
import PyPDF2
import docx
import pandas as pd
from pptx import Presentation

# Initialize Flask
app = Flask(__name__)

# Set your OpenAI API key (ensure the environment variable is set)

# Global variable to store the latest uploaded file content
uploaded_file_text = None

# --- Helper Functions to Extract Text from Files ---

def extract_text_from_pdf(file):
    file.seek(0)
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    file.seek(0)
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_excel(file):
    file.seek(0)
    try:
        df = pd.read_excel(file)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel file: {e}"

def extract_text_from_pptx(file):
    file.seek(0)
    prs = Presentation(file)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)

# --- Chatbot Endpoints ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global uploaded_file_text

    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Build the conversation context
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        # Include file content if available
        if uploaded_file_text:
            # Optionally, you could summarize or truncate this text if it's too long.
            messages.append({"role": "system", "content": f"File content: {uploaded_file_text}"})
        messages.append({"role": "user", "content": user_message})

        # Use GPT-4o-mini by setting the model accordingly.
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=messages)
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- File Upload Endpoint ---

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_file_text

    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = uploaded_file.filename
    if filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "pptx"}
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type .{extension} not allowed"}), 400

    text = ""
    if extension == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif extension == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif extension == "xlsx":
        text = extract_text_from_excel(uploaded_file)
    elif extension == "pptx":
        text = extract_text_from_pptx(uploaded_file)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Store the extracted text in the global variable for use in the chat context.
    uploaded_file_text = text

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True)
