import json
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from groq import Groq

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

def clean_and_parse_json(raw_text):
    """Safely extracts JSON even if the AI includes extra text or backticks."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise Exception("Failed to parse valid JSON from AI response.")

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Get uploaded file and language from frontend
        file = request.files.get('file')
        language = request.form.get('language', 'English')

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # 2. Extract text from PDF or Plain Text file
        extracted_text = ""
        if file.filename.endswith('.pdf'):
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        else:
            extracted_text = file.read().decode('utf-8', errors='ignore')

        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from the provided file."}), 400

        # 3. Construct system and user prompt with JSON enforcement
        system_instructions = (
            "You are a study content extractor. You MUST respond strictly with a valid JSON object. "
            "Do not output markdown block formatting (no ```json)."
        )

        user_prompt = f"""
        Analyze the following text and explain it in {language}.
        You MUST return ONLY a valid JSON object matching this exact structure:

        {{
            "headings": {{
                "overview": "Overview",
                "key_points": "Key points",
                "key_terms": "Key terms"
            }},
            "overview": "Summary paragraph here translated to {language}",
            "key_points": ["Point 1 in {language}", "Point 2 in {language}", "Point 3 in {language}"],
            "key_terms": [
                {{"term": "TERM 1", "definition": "Definition 1 in {language}"}},
                {{"term": "TERM 2", "definition": "Definition 2 in {language}"}}
            ]
        }}

        Text to summarize:
        {extracted_text}
        """

        # 4. Request summary from Groq (using llama-3.3-70b-versatile for high quality)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )

        raw_response_text = chat_completion.choices[0].message.content

        # 5. Clean and parse JSON
        parsed_data = clean_and_parse_json(raw_response_text)

        return jsonify(parsed_data)

    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)