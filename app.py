import os
from io import BytesIO

from openai import OpenAI
from flask import Flask, render_template, request, jsonify, send_file

# Additional imports for file processing
import PyPDF2
import docx
import pandas as pd
from pptx import Presentation

# Additional import for PDF generation
from fpdf import FPDF

# For vector database (RAG) functionality (from previous steps)
import faiss
import numpy as np

# Initialize the OpenAI client using the migrated approach
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask
app = Flask(__name__)

# -----------------------------
# Global variables used in the app
# -----------------------------
faiss_index = None          # FAISS index for vector search
chunk_texts = []            # List mapping each FAISS vector to its corresponding text chunk
full_document_text = None   # Full text of the uploaded document (for fallback)
# (Note: In a production system, consider using sessions or a database rather than globals.)

# -----------------------------
# Helper Functions for File Extraction and RAG
# -----------------------------

def extract_text_from_pdf(file):
    """
    Extracts and returns text from a PDF file.

    Args:
      file (FileStorage): An uploaded PDF file.

    Returns:
      str: The extracted text from the PDF.
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
    Extracts and returns text from a DOCX (Microsoft Word) file.

    Args:
      file (FileStorage): An uploaded DOCX file.

    Returns:
      str: The extracted text from the Word document.
    """
    file.seek(0)
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_excel(file):
    """
    Extracts and returns text from an Excel file by converting its content to a string.

    Args:
      file (FileStorage): An uploaded Excel file.

    Returns:
      str: The content of the Excel file as a string, or an error message if reading fails.
    """
    file.seek(0)
    try:
        df = pd.read_excel(file)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel file: {e}"

def extract_text_from_pptx(file):
    """
    Extracts and returns text from a PowerPoint (.pptx) file.

    Args:
      file (FileStorage): An uploaded PowerPoint file.

    Returns:
      str: The extracted text from the presentation.
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
    Splits the provided text into smaller chunks based on the specified number of words.

    Args:
      text (str): The full text to be chunked.
      chunk_size (int): The maximum number of words per chunk (default is 500).

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
    Chunks the provided text, computes embeddings for each chunk using OpenAI's embedding model,
    and builds a FAISS vector index for retrieval.

    Also stores the full text for fallback purposes.

    Args:
      text (str): The full text to process.

    Effects:
      Updates global variables: 'faiss_index', 'chunk_texts', and 'full_document_text'.
    """
    global faiss_index, chunk_texts, full_document_text
    chunks = chunk_text(text)
    # The embedding dimension for text-embedding-ada-002 is 1536.
    d = 1536
    index = faiss.IndexFlatL2(d)
    embeddings = []
    chunk_texts = []  # Reset the list of chunks
    for chunk in chunks:
        try:
            response = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embedding = np.array(response.data[0].embedding, dtype='float32')
            embeddings.append(embedding)
            chunk_texts.append(chunk)
        except Exception as e:
            print(f"Error computing embedding for a chunk: {e}")
    if embeddings:
        embeddings_np = np.vstack(embeddings)  # Shape: (num_chunks, d)
        index.add(embeddings_np)
        faiss_index = index
    # Save the full text for fallback if retrieval yields no results.
    full_document_text = text

def retrieve_relevant_chunks(query, top_k=5):
    """
    Computes the embedding for the query and retrieves the top_k most similar text chunks from the FAISS index.

    Args:
      query (str): The user's query.
      top_k (int): Number of top similar chunks to retrieve (default is 5).

    Returns:
      list[str]: A list of relevant text chunks. If no chunks are found, returns the full document text.
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
        # Fallback: if no relevant chunks are found, use the full document text.
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
    Generates a Microsoft Word (.docx) document containing the provided text.

    Args:
      text (str): The content to be included in the document.

    Returns:
      BytesIO: A stream containing the generated DOCX file.
    """
    document = docx.Document()
    document.add_heading("Generated Document", level=1)
    # Add the content as paragraphs.
    for paragraph in text.split("\n"):
        document.add_paragraph(paragraph)
    output = BytesIO()
    document.save(output)
    output.seek(0)
    return output

def generate_pptx(text):
    """
    Generates a PowerPoint (.pptx) presentation containing the provided text.

    Args:
      text (str): The content to be included in the presentation.

    Returns:
      BytesIO: A stream containing the generated PPTX file.
    """
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[1]  # Use the "Title and Content" layout.
    slide = presentation.slides.add_slide(slide_layout)
    # Set the slide title.
    slide.shapes.title.text = "Generated Presentation"
    # Add the content in the content placeholder.
    content_placeholder = slide.placeholders[1]
    content_placeholder.text = text
    output = BytesIO()
    presentation.save(output)
    output.seek(0)
    return output

def generate_pdf(text):
    """
    Generates a PDF document containing the provided text.

    Args:
      text (str): The content to be included in the PDF.

    Returns:
      BytesIO: A stream containing the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Split the text into lines to add via multi_cell.
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    output = BytesIO()
    pdf.output(dest="S").encode("latin1")  # Ensure proper encoding for the PDF output
    # Instead of writing to a file, we get the PDF as a byte string.
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    output.write(pdf_bytes)
    output.seek(0)
    return output

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
    Processes a chat message from the user. Retrieves any relevant document context (via RAG)
    and uses it to build a conversation prompt for the LLM.

    Returns:
      JSON: The LLM's answer.
    """
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        # Retrieve the most relevant document chunks for the query.
        relevant_chunks = retrieve_relevant_chunks(user_message, top_k=5)
        context_text = "\n\n".join(relevant_chunks)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        if context_text:
            messages.append({"role": "system", "content": f"Relevant document context:\n{context_text}"})
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles file uploads. Extracts text from the uploaded file (PDF, DOCX, XLSX, or PPTX),
    then builds a vector store (using FAISS) and stores the full document text for retrieval.

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

    # Build the vector store from the extracted text.
    create_vector_store(text)

    return jsonify({"text": text})

@app.route("/generate_document", methods=["POST"])
def generate_document():
    """
    Generates a document based on the user's prompt and desired file format.

    Expects either JSON data or form data with:
      - prompt (str): The instruction or topic for the write-up.
      - file_type (str): The desired file format ("docx", "pptx", or "pdf").

    Returns:
      A downloadable file (with appropriate MIME type and filename) generated by the LLM.
    """
    # Accept JSON if provided, otherwise fallback to form data.
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    prompt = data.get("prompt")
    file_type = data.get("file_type", "docx").lower()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Call the LLM to generate content based on the prompt.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        document_text = response.choices[0].message.content.strip()

        # Generate the file in the requested format.
        if file_type == "docx":
            file_stream = generate_docx(document_text)
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            file_ext = "docx"
        elif file_type == "pptx":
            file_stream = generate_pptx(document_text)
            mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            file_ext = "pptx"
        elif file_type == "pdf":
            file_stream = generate_pdf(document_text)
            mime_type = "application/pdf"
            file_ext = "pdf"
        else:
            return jsonify({"error": "Unsupported file type requested."}), 400

        # Return the generated file for download.
        return send_file(
            file_stream,
            mimetype=mime_type,
            as_attachment=True,
            download_name=f"generated_document.{file_ext}"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
