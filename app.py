from flask import Flask, request, render_template_string
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load credentials and connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("dream-creds.json", scope)
client = gspread.authorize(creds)

# Load sheet
sheet = client.open("Dream Symbol Dictionary").sheet1

# Homepage + form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dream = request.form["dream"]
        interpretation = interpret_dream(dream)
        sheet.append_row([dream, interpretation])
        return render_template_string(PAGE_TEMPLATE, result=interpretation)
    return render_template_string(PAGE_TEMPLATE)

# Fake interpretation logic (replace later with AI)
def interpret_dream(dream_text):
    if "bicycle" in dream_text:
        return "You are progressing slowly but steadily in your spiritual journey."
    elif "car" in dream_text:
        return "You are in control of your life’s direction."
    elif "water" in dream_text:
        return "You are dealing with emotional cleansing or spiritual warfare."
    else:
        return "Your dream may be symbolic of hidden guidance. Pray for confirmation."

# HTML layout
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dream Interpreter</title>
</head>
<body style="font-family: Arial; background-color: #111; color: white; text-align: center; padding: 50px;">
    <h1>🌙 AI Dream Interpreter</h1>
    <form method="POST">
        <textarea name="dream" rows="6" cols="50" placeholder="Describe your dream..." required></textarea><br><br>
        <button type="submit">Interpret Dream</button>
    </form>
    {% if result %}
        <h2>🔮 Interpretation:</h2>
        <p>{{ result }}</p>
    {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
