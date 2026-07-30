from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from PyPDF2 import PdfReader
import os, json
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)
client = Groq()


@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    language = request.form.get("language", "Hindi")
    path = "temp_" + file.filename
    file.save(path)

    filename_lower = file.filename.lower()

    if filename_lower.endswith(".pdf"):
        reader = PdfReader(path)
        text = "".join(page.extract_text() for page in reader.pages)
    else:
        os.remove(path)
        return jsonify({"error": "This deployed version supports PDF files only. Audio summarization is available in the local version."}), 400

    prompt = f"""Summarize the following content for a student in detail. Respond ONLY with valid JSON, no other text, no markdown, in exactly this format:

{{
  "headings": {{
    "overview": "the word 'Overview' translated into {language}",
    "key_points": "the phrase 'Key Points' translated into {language}",
    "key_terms": "the phrase 'Key Terms' translated into {language}"
  }},
  "overview": "4-5 sentence overview here, in {language}",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "key_terms": [
    {{"term": "term name written in {language} script (transliterated phonetically, same word/pronunciation, NOT conceptually translated into a different word, and NOT left in English letters)", "definition": "detailed 1-2 sentence definition in {language}"}}
  ]
}}

Rules:
- ALWAYS include AT LEAST 5 key_points, even if you must break the content into more granular points to reach 5.
- ALWAYS include AT LEAST 4 key_terms, even if you must extract lesser-known terms, names, numbers, or concepts mentioned in the text.
- Never return fewer than these minimums, regardless of how short the source content is.
- For key_terms specifically: write the "term" field using {language}'s native script, transliterating it phonetically (same sound, same word) rather than translating it into a different word or leaving it in English letters.
- For programming/CS-specific keywords with no real-world equivalent (like "Array", "Function"), you may keep them in English letters if transliteration would be confusing.
- Translate all other text (overview, key_points, definitions, headings) into {language}.

Content:
{text[:15000]}"""

    reply = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = reply.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)

    os.remove(path)
    return jsonify(parsed)


if __name__ == "__main__":
    app.run(debug=True)

    @app.route("/")
def home():
    return "Backend is live!", 200