import os
import uuid
import json
from io import BytesIO

from openai import OpenAI
from flask import Flask, render_template, request, jsonify, send_file

# Additional imports for file processing and document generation
import PyPDF2
import docx
import pandas as pd
from pptx import Presentation
from fpdf import FPDF

# For RAG (retrieval-augmented generation)
import faiss
import numpy as np

# Initialize the OpenAI client (using your migrated code)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask
app = Flask(__name__)

# -----------------------------
# Global Variables
# -----------------------------
faiss_index = None          # FAISS index for vector search
chunk_texts = []            # Mapping: each vector in FAISS corresponds to a text chunk
full_document_text = None   # Full text from an uploaded document (for fallback)
# Note: In production, consider using sessions or a database instead of globals.

# -----------------------------
# Helper Functions for File Extraction and RAG
# -----------------------------

def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.

    Args:
      file (FileStorage): The uploaded PDF file.

    Returns:
      str: The extracted text.
    """
    file.seek(0)
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    """
    Extract text from a DOCX (Word) file.

    Args:
      file (FileStorage): The uploaded DOCX file.

    Returns:
      str: The extracted text.
    """
    file.seek(0)
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_excel(file):
    """
    Extract text from an Excel file by converting its content to a string.

    Args:
      file (FileStorage): The uploaded Excel file.

    Returns:
      str: The content of the Excel file as text.
    """
    file.seek(0)
    try:
        df = pd.read_excel(file)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel file: {e}"

def extract_text_from_pptx(file):
    """
    Extract text from a PowerPoint file.

    Args:
      file (FileStorage): The uploaded PPTX file.

    Returns:
      str: The extracted text.
    """
    file.seek(0)
    prs = Presentation(file)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)

def chunk_text(text, chunk_size=500):
    """
    Splits a long text into chunks of approximately chunk_size words.

    Args:
      text (str): The full text.
      chunk_size (int): Maximum words per chunk.

    Returns:
      list[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def create_vector_store(text):
    """
    Build a vector store (using FAISS) from the given text.

    The text is chunked, each chunk is embedded via OpenAI's embeddings, and then
    stored in a FAISS index. The full text is also saved for fallback.

    Args:
      text (str): The full document text.

    Effects:
      Updates globals: 'faiss_index', 'chunk_texts', 'full_document_text'.
    """
    global faiss_index, chunk_texts, full_document_text
    chunks = chunk_text(text)
    d = 1536  # Embedding dimension for text-embedding-ada-002
    index = faiss.IndexFlatL2(d)
    embeddings = []
    chunk_texts = []  # Reset chunks
    for chunk in chunks:
        try:
            response = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embedding = np.array(response.data[0].embedding, dtype='float32')
            embeddings.append(embedding)
            chunk_texts.append(chunk)
        except Exception as e:
            print(f"Error computing embedding for a chunk: {e}")
    if embeddings:
        embeddings_np = np.vstack(embeddings)
        index.add(embeddings_np)
        faiss_index = index
    full_document_text = text  # Save full text for fallback

def retrieve_relevant_chunks(query, top_k=5):
    """
    Retrieve the top_k text chunks from the vector store most relevant to the query.

    Args:
      query (str): The user query.
      top_k (int): The number of top chunks to return.

    Returns:
      list[str]: Relevant text chunks. If none found, returns the full document text.
    """
    global faiss_index, chunk_texts, full_document_text
    if faiss_index is None:
        return []
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=query)
        query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_k)
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunk_texts):
                relevant_chunks.append(chunk_texts[idx])
        if not relevant_chunks and full_document_text:
            relevant_chunks = [full_document_text]
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []

# -----------------------------
# Helper Functions for Document Generation
# -----------------------------

def generate_docx(text):
    """
    Generate a DOCX document containing the provided text.

    Args:
      text (str): The document content.

    Returns:
      BytesIO: Stream containing the DOCX file.
    """
    document = docx.Document()
    document.add_heading("Generated Document", level=1)
    for paragraph in text.split("\n"):
        document.add_paragraph(paragraph)
    output = BytesIO()
    document.save(output)
    output.seek(0)
    return output

def generate_pptx(text):
    """
    Generate a PPTX presentation containing the provided text.

    Args:
      text (str): The content for the presentation.

    Returns:
      BytesIO: Stream containing the PPTX file.
    """
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[1]
    slide = presentation.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Generated Presentation"
    slide.placeholders[1].text = text
    output = BytesIO()
    presentation.save(output)
    output.seek(0)
    return output

def generate_pdf(text):
    """
    Generate a PDF document containing the provided text.

    Args:
      text (str): The document content.

    Returns:
      BytesIO: Stream containing the PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    output = BytesIO(pdf_bytes)
    output.seek(0)
    return output

def store_file_temporarily(file_stream, ext):
    """
    Store a file stream in temporary storage and return a unique download URL.

    Args:
      file_stream (BytesIO): The file stream.
      ext (str): The file extension (e.g., 'docx', 'pptx', 'pdf').

    Returns:
      str: The relative download URL for the stored file.
    """
    os.makedirs("temp_files", exist_ok=True)
    temp_filename = f"{uuid.uuid4()}.{ext}"
    temp_path = os.path.join("temp_files", temp_filename)
    with open(temp_path, "wb") as f:
        f.write(file_stream.read())
    return f"/download/{temp_filename}"

# -----------------------------
# Flask Endpoints
# -----------------------------

@app.route("/")
def index():
    """
    Renders the main UI page.
    """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles user messages from the main prompt.

    If the LLM detects a document generation request, it returns a function call
    with structured JSON. The backend then generates the document file, stores it temporarily,
    and returns a download link.

    Otherwise, it returns a normal chat answer.

    Returns:
      JSON: A response with either a chat answer or a download link.
    """
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Build conversation context.
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. If a user's request indicates a need to generate a document "
                "(for example, a Word document, PowerPoint, or PDF), respond by calling the function "
                "generate_document with a JSON object containing 'file_type' (one of 'docx', 'pptx', 'pdf') "
                "and 'document_content' (the text for the document). Otherwise, answer normally."
            )}
        ]
        # (Optional) Add retrieval-based context if available.
        relevant_chunks = retrieve_relevant_chunks(user_message, top_k=5)
        context_text = "\n\n".join(relevant_chunks)
        if context_text:
            messages.append({"role": "system", "content": f"Relevant document context:\n{context_text}"})
        messages.append({"role": "user", "content": user_message})

        # Define function for document generation.
        functions = [{
            "name": "generate_document",
            "description": (
                "Generates a document. The output should be a JSON with two keys: "
                "'file_type' (one of ['docx', 'pptx', 'pdf']) and "
                "'document_content' (the text content for the document)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {"type": "string", "enum": ["docx", "pptx", "pdf"]},
                    "document_content": {"type": "string", "description": "The text content for the document."}
                },
                "required": ["file_type", "document_content"]
            }
        }]

        # Call the chat API with function calling enabled.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto"  # Let the model decide if a function call is needed.
        )

        message_response = response.choices[0].message

        # Convert the message to a dictionary if it isn’t one already.
        if hasattr(message_response, "to_dict"):
            message_response = message_response.to_dict()

        # Check if the assistant decided to call the generate_document function.
        if message_response.get("function_call"):
            # Parse function call arguments.
            function_call = message_response["function_call"]
            args = json.loads(function_call.get("arguments", "{}"))
            file_type = args.get("file_type")
            document_content = args.get("document_content")
            if not file_type or not document_content:
                return jsonify({"answer": "Failed to parse document generation parameters."})
            # Generate the document file using the appropriate helper.
            if file_type == "docx":
                file_stream = generate_docx(document_content)
                ext = "docx"
            elif file_type == "pptx":
                file_stream = generate_pptx(document_content)
                ext = "pptx"
            elif file_type == "pdf":
                file_stream = generate_pdf(document_content)
                ext = "pdf"
            else:
                return jsonify({"answer": "Unsupported file type in document generation."})

            # Store the generated file temporarily and get a download URL.
            download_url = store_file_temporarily(file_stream, ext)

            answer = f"Document generated. You can download it here: {download_url}"
            return jsonify({"answer": answer})
        else:
            # Regular chat answer.
            answer = message_response.get("content")
            return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles file uploads. Extracts text from the uploaded file (PDF, DOCX, XLSX, or PPTX),
    builds a vector store (for RAG), and stores the full document text.

    Returns:
      JSON: The extracted text or an error message.
    """
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400
    filename = uploaded_file.filename
    if filename == "":
        return jsonify({"error": "No selected file"}), 400
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

    create_vector_store(text)
    return jsonify({"text": text})

@app.route("/download/<filename>")
def download_file(filename):
    """
    Serves a file from temporary storage for download.

    Args:
      filename (str): The name of the file to download.

    Returns:
      The file as an attachment, or a 404 if not found.
    """
    file_path = os.path.join("temp_files", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
