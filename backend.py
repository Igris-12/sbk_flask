import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Gemini API Configuration ---
# Fetches the API key from the environment variables.
# Make sure you have a .env file with GEMINI_API_KEY="YOUR_API_KEY"
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=gemini_api_key)
except ValueError as e:
    print(f"Error: {e}")
    print("Please create a .env file in the 'gemini_server' directory and add your Gemini API key as GEMINI_API_KEY='your_key_here'")
    exit()


# Initialize the Gemini Pro model
model = genai.GenerativeModel('gemini-flash-lite-latest')

# --- API Endpoint Definition ---
@app.route('/generate', methods=['POST'])
def generate_content():
    """
    Receives a query from a client, sends it to the Gemini API,
    and returns the generated content as a JSON response.
    """
    # 1. Validate the incoming request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query', None)

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    # 2. Interact with the Gemini API
    try:
        print(f"Received query: {query}")
        # Generate content using the provided query
        response = model.generate_content(query)
        
        # 3. Format and return the response
        # The response from the API is returned in a structured format.
        # We extract the text part to send back to the Node.js server.
        gemini_response = response.text
        print(f"Sending response: {gemini_response}")
        
        return jsonify({"response": gemini_response})

    except Exception as e:
        # Handle potential errors from the Gemini API call
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to generate content from Gemini API.", "details": str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Runs the Flask app on port 5000.
    # The host '0.0.0.0' makes it accessible from other devices on the same network.
    app.run(host='0.0.0.0', port=5000, debug=True)
