import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- Gemini API Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-flash-lite-latest')

# ✅ Root route (fixes 404 on homepage)
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Gemini Flask API on Vercel!",
        "routes": ["/generate"]
    })

@app.route('/generate', methods=['POST'])
def generate_content():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    try:
        response = model.generate_content(query)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": "Failed to generate content", "details": str(e)}), 500

# ✅ Needed for Vercel to detect the app object
if __name__ == "__main__":
    app.run(debug=True)
