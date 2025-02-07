import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from flask import Flask, render_template, request, jsonify

# Initialize Flask
app = Flask(__name__)

# Set your OpenAI API key (make sure the environment variable is set)

# Home route that serves the chat UI
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint to handle chat messages
@app.route("/chat", methods=["POST"])
def chat():
    # Get the message from the frontend
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Call the OpenAI API (using GPT-4 or your desired model)
        response = client.chat.completions.create(model="gpt-4o-mini",  # or "gpt-3.5-turbo" if preferred/available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ])
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        # Return error details for debugging (in production, you might want to log this instead)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app on http://127.0.0.1:5000/
    app.run(debug=True)
