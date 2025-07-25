from flask import Flask, render_template, request
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import difflib

app = Flask(__name__)

# Google Sheets setup
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Access your sheet
sheet = client.open_by_key("1ToWPpJ_u-Z14eqL9U1oJppXX6f63h_eZcjUVglKr4Zk").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Clean column names
df.columns = [col.strip().lower() for col in df.columns]
df['symbol'] = df['symbol'].str.strip().str.lower()
df['interpretation'] = df['interpretation'].str.strip()

# Matching logic (fuzzy partial match)
def interpret_dream(dream_text):
    dream_text = dream_text.lower()
    matches = []

    for _, row in df.iterrows():
        symbol = row['symbol']
        interpretation = row['interpretation']

        # Partial match OR high similarity
        if symbol in dream_text or difflib.SequenceMatcher(None, symbol, dream_text).ratio() > 0.6:
            matches.append(f"🔹 {interpretation}")

    if matches:
        return "<br><br>".join(matches)
    else:
        return "❌ No known dream symbols matched. Pray for discernment and clarity."

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    interpretation = None
    if request.method == "POST":
        dream = request.form.get("dream", "")
        interpretation = interpret_dream(dream)
    return render_template("index.html", interpretation=interpretation)

if __name__ == "__main__":
    app.run(debug=True, port=10000)
