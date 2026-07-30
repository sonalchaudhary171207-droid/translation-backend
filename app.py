import os
import io
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from groq import Groq
import pytesseract
from pdf2image import convert_from_bytes

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
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text using PyPDF first; falls back to Tesseract OCR for scanned PDFs."""
    extracted_text = ""

    # Method 1: PyPDF for selectable text
    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
    except Exception as e:
        print(f"PyPDF extraction error: {e}")

    extracted_text = clean_text(extracted_text)

    # Method 2: OCR Fallback if PyPDF extracted little to no text
    if len(extracted_text) < 15:
        print("Scanned PDF detected. Running OCR...")
        try:
            images = convert_from_bytes(file_bytes)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            extracted_text = clean_text(ocr_text)
        except Exception as e:
            print(f"OCR processing error: {e}")

    return extracted_text


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

    target_language = request.form.get("language", "English")

    try:
        file.seek(0)
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # 3. Extract text (supports both standard and scanned PDFs)
        extracted_text = extract_text_from_pdf(file_bytes)

        if not extracted_text or len(extracted_text) < 10:
            return jsonify({
                "error": "Could not extract text from the file even with OCR. Please make sure the PDF image quality is readable."
            }), 400

        # Limit token size for context length
        max_chars = 12000
        truncated_text = extracted_text[:max_chars]

        # 4. Prompt Groq for Strict JSON response
        prompt_message = f"""
        Analyze the following study material and generate notes in {target_language}.

        You MUST respond ONLY with a raw, valid JSON object (no markdown formatting, no ```json tags, no additional commentary).
        Use this exact schema:

        {{
          "headings": {{
            "overview": "Overview heading translated to {target_language}",
            "key_points": "Key Points heading translated to {target_language}",
            "key_terms": "Key Terms heading translated to {target_language}"
          }},
          "overview": "Detailed overview paragraph summarizing the core concepts in {target_language}.",
          "key_points": [
            "Key takeaway point 1 in {target_language}",
            "Key takeaway point 2 in {target_language}",
            "Key takeaway point 3 in {target_language}"
          ],
          "key_terms": [
            {{
              "term": "Term Name in {target_language}",
              "definition": "Definition or explanation in {target_language}"
            }}
          ]
        }}

        Source text:
        {truncated_text}
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful study assistant. Output strictly valid JSON in {target_language} matching the requested structure."
                },
                {
                    "role": "user",
                    "content": prompt_message
                }
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        raw_response = completion.choices[0].message.content.strip()

        # Clean JSON if any backticks leaked in
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        raw_response = raw_response.strip()

        parsed_json = json.loads(raw_response)
        return jsonify(parsed_json), 200

    except Exception as e:
        print("Server error:", str(e))
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)