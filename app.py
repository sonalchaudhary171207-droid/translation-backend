import os
import io
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from groq import Groq

app = Flask(__name__)

# Allow cross-origin requests from Vercel frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def clean_text(text: str) -> str:
    """Removes excessive whitespace and unwanted special characters from extracted text."""
    if not text:
        return ""
    # Normalize spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Backend API is running"}), 200


@app.route("/summarize", methods=["POST"])
def summarize():
    # 1. Verify Groq API Key
    if not client:
        return jsonify({"error": "GROQ_API_KEY environment variable is not set on server."}), 500

    # 2. Check if a file was provided in the request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a PDF file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Read optional target language from form data (default to English)
    target_language = request.form.get("language", "English")

    try:
        # 3. CRITICAL FIX: Reset file pointer to beginning of stream before reading
        file.seek(0)
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # 4. Extract text with PyPDF from in-memory stream
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        extracted_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

        extracted_text = clean_text(extracted_text)

        # 5. Handle scanned or unreadable PDFs
        if not extracted_text or len(extracted_text) < 5:
            return jsonify({
                "error": "Could not extract text from the provided file. Ensure the PDF contains selectable text and is not an image-based scan."
            }), 400

        # Truncate input text if it's too long to prevent context token overflow (~10k chars)
        max_chars = 10000
        truncated_text = extracted_text[:max_chars]

        # 6. Call Groq API
        prompt_message = f"""
        Analyze the following text and create comprehensive study notes in {target_language}.
        Include:
        1. Overview
        2. Key Points / Takeaways
        3. Important Vocabulary / Terms

        Text content:
        {truncated_text}
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert study assistant. Output all summaries and notes clearly in {target_language}."
                },
                {
                    "role": "user",
                    "content": prompt_message
                }
            ],
            temperature=0.4,
            max_tokens=1200,
        )

        summary_result = completion.choices[0].message.content

        return jsonify({
            "success": True,
            "summary": summary_result
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)