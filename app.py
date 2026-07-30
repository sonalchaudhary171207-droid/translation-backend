import os
import io
import re
import json
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from groq import Groq
import pytesseract
from pdf2image import convert_from_bytes

app = Flask(__name__)

# Allow requests from frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def clean_text(text: str):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes):

    extracted_text = ""

    # ---------- PyPDF ----------
    try:

        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        print(f"PDF Pages : {len(reader.pages)}")

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_text += page_text + "\n"

    except Exception as e:
        print("PyPDF ERROR :", e)

    extracted_text = clean_text(extracted_text)

    print("PyPDF Characters :", len(extracted_text))

    # ---------- OCR ----------
    if len(extracted_text) < 15:

        print("Running OCR...")

        try:

            images = convert_from_bytes(file_bytes)

            print("OCR Images :", len(images))

            ocr_text = ""

            for index, image in enumerate(images):

                page_text = pytesseract.image_to_string(image)

                print(f"OCR Page {index+1} : {len(page_text)} chars")

                ocr_text += page_text + "\n"

            extracted_text = clean_text(ocr_text)

            print("OCR Characters :", len(extracted_text))

        except Exception as e:

            print("OCR ERROR")
            traceback.print_exc()

    return extracted_text


@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "message": "Backend API is running"
    })


@app.route("/summarize", methods=["POST"])
def summarize():

    if not client:
        return jsonify({
            "error": "GROQ_API_KEY is missing."
        }), 500

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    language = request.form.get("language", "English")

    try:

        file.seek(0)
        file_bytes = file.read()

        print("=" * 60)
        print("NEW REQUEST")
        print("Filename :", file.filename)
        print("Language :", language)
        print("Size :", len(file_bytes))
        print("=" * 60)

        if len(file_bytes) == 0:

            return jsonify({
                "error": "Uploaded file is empty."
            }), 400

        extracted_text = extract_text_from_pdf(file_bytes)

        print("Final Extracted Characters :", len(extracted_text))

        if len(extracted_text) < 10:

            return jsonify({
                "error": "Could not extract readable text from PDF."
            }), 400

        truncated_text = extracted_text[:12000]

        prompt = f"""
Analyze the following study material.

Generate study notes in {language}.

Return ONLY valid JSON.

Schema:

{{
  "headings": {{
    "overview":"Overview",
    "key_points":"Key Points",
    "key_terms":"Key Terms"
  }},
  "overview":"paragraph",
  "key_points":[
    "point1",
    "point2",
    "point3"
  ],
  "key_terms":[
    {{
      "term":"...",
      "definition":"..."
    }}
  ]
}}

Source Text:

{truncated_text}
"""
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful study assistant. Respond ONLY with valid JSON in {language}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )

        raw_response = completion.choices[0].message.content.strip()

        print("=" * 60)
        print("RAW GROQ RESPONSE")
        print(raw_response)
        print("=" * 60)

        # Remove markdown code fences if present
        raw_response = re.sub(r"^```json\s*", "", raw_response)
        raw_response = re.sub(r"^```\s*", "", raw_response)
        raw_response = re.sub(r"\s*```$", "", raw_response)
        raw_response = raw_response.strip()

        try:
            parsed_json = json.loads(raw_response)
            return jsonify(parsed_json), 200
        except json.JSONDecodeError:
            return jsonify({
                "error": "Groq returned invalid JSON.",
                "raw_response": raw_response
            }), 500

    except Exception as e:
        print("=" * 60)
        print("SERVER ERROR")
        traceback.print_exc()
        print("=" * 60)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)