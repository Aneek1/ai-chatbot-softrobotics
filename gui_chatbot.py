import os
import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import spacy
import google.generativeai as genai
from googleapiclient.discovery import build

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Google API Key & Search Engine ID
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = "43c87e5cec9164abb"  # Replace with your actual Search Engine ID

# Set default CSV path
CSV_PATH = "expanded_soft_robotics_fabrication_methods.csv"

# Load CSV file
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("✅ CSV file loaded successfully!")
else:
    df = None
    print("❌ No CSV file found!")

# Function to process user input and extract meaningful keywords/phrases
def extract_keywords(query):
    doc = nlp(query.lower())
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    print(f"Extracted Keywords: {keywords}")  # Debug print
    return keywords  # Return as list of keywords for better processing

# Function to search CSV for relevant fabrication methods
def search_csv(query):
    if df is None:
        return None

    # Extract keywords from the query
    keywords = extract_keywords(query)
    print(f"Searching for keywords: {keywords}")  # Debug print

    # Try to match the keywords in different columns of the CSV
    matched_rows = []
    for _, row in df.iterrows():
        fabrication_method = str(row['fabrication_method']).lower()
        description = str(row['description']).lower()

        # Check if any of the keywords match the fabrication_method or description
        if any(keyword in fabrication_method or keyword in description for keyword in keywords):
            matched_rows.append(
                f"🛠 **{row['fabrication_method']}**:\n"
                f"➡️ *{row['description']}*\n\n"
                f"📌 **Materials**: {row['materials']}\n"
                f"📊 **Properties**: {row['properties']}\n"
                f"📝 **How it's done**: {row['steps_to_fabricate']}\n"
                f"⏳ **Time Required**: {row['time_taken']}\n"
                f"✅ **Pros**: {row['advantages']}\n"
                f"⚠️ **Cons**: {row['disadvantages']}\n"
                f"🔍 **Used for**: {row['application_example']}\n"
                f"📚 **Source**: {row['notable_source']}\n"
                "----------------------"
            )

    if matched_rows:
        return "\n".join(matched_rows)
    return None

# Function to search Google if CSV has no data
def google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()
        print(f"Google API response: {res}")  # Debug print
        results = res.get("items", [])
        
        if results:
            return "🔍 I found this on Google:\n" + "\n".join([f"🔗 {item['title']}: {item['link']}" for item in results[:3]])
        else:
            return "Hmm... I couldn't find much on Google either!"
    except Exception as e:
        return f"🚨 Error with Google Search: {e}"

# Function to generate Gemini AI response (only as a last resort)
def gemini_response(query):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(query)
        print(f"Gemini response: {response}")  # Debug print
        return response.text if response else "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"🤖 Error with Gemini API: {e}"

# Main chatbot function (ensures CSV is used first)
def chatbot_response(user_input):
    # First, try to search the CSV
    csv_result = search_csv(user_input)

    if csv_result:
        return f"Here's what I found for you! 😊\n\n{csv_result}"

    # If CSV doesn't give results, attempt Google search
    google_result = google_search(user_input)
    if "Error" not in google_result and "Hmm..." not in google_result:
        return google_result

    # If neither CSV nor Google search gives results, fallback to Gemini AI
    return gemini_response(user_input)  # Last fallback

# GUI Setup
def send_message():
    user_input = user_entry.get()
    if user_input.strip():  # Check if input is not empty
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"You: {user_input}\n", "user")
        response = chatbot_response(user_input)
        chat_window.insert(tk.END, f"Bot: {response}\n\n", "bot")
        chat_window.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)
    else:
        print("Input is empty. Please enter a query.")  # Debug message

# Tkinter UI
root = tk.Tk()
root.title("AI Chatbot with CSV + NLP + Google Search")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=25, font=("Arial", 12))
chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

user_entry = tk.Entry(root, width=70, font=("Arial", 12))
user_entry.grid(row=1, column=0, padx=10, pady=10, sticky="w")

send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12))
send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

root.mainloop()