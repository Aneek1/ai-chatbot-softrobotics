import os
import threading
import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import spacy
import google.generativeai as genai
from googleapiclient.discovery import build

# Load Spacy NLP model once
nlp = spacy.load("en_core_web_sm")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = "43c87e5cec9164abb"  # Replace with actual ID

CSV_PATH = "expanded_soft_robotics_fabrication_methods.csv"


class DataProcessor:
    """Handles CSV file loading and searching"""

    def __init__(self, csv_path):
        self.data = self.load_csv(csv_path)

    def load_csv(self, path):
        """Load CSV into a dictionary for faster lookups"""
        if os.path.exists(path):
            df = pd.read_csv(path)
            print("✅ CSV loaded successfully!")
            return df.to_dict(orient="records")  # Store as a list of dicts for quick access
        print("❌ No CSV found!")
        return None

    def extract_keywords(self, query):
        """Extract meaningful keywords from the input"""
        doc = nlp(query.lower())
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    def search(self, query):
        """Fast CSV lookup using preloaded data"""
        if not self.data:
            return None

        keywords = self.extract_keywords(query)
        matched_rows = [
            f"🛠 **{row['fabrication_method']}**:\n"
            f"➡️ *{row['description']}*\n\n"
            f"📌 **Materials**: {row['materials']}\n"
            f"📊 **Properties**: {row['properties']}\n"
            f"📝 **Steps**: {row['steps_to_fabricate']}\n"
            f"⏳ **Time**: {row['time_taken']}\n"
            f"✅ **Pros**: {row['advantages']}\n"
            f"⚠️ **Cons**: {row['disadvantages']}\n"
            f"🔍 **Application**: {row['application_example']}\n"
            "----------------------"
            for row in self.data
            if any(keyword in str(row['fabrication_method']).lower() or keyword in str(row['description']).lower() for keyword in keywords)
        ]

        return "\n".join(matched_rows) if matched_rows else None


class SearchEngine:
    """Handles Google Search and Gemini AI queries"""

    def __init__(self):
        self.google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

    def google_search(self, query):
        """Perform Google search"""
        try:
            res = self.google_service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()
            results = res.get("items", [])
            return "🔍 Google results:\n" + "\n".join([f"🔗 {item['title']}: {item['link']}" for item in results[:3]]) if results else "No Google results found."
        except Exception as e:
            return f"🚨 Google Error: {e}"

    def gemini_response(self, query):
        """Generate response using Gemini AI"""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(query)
            return response.text if response else "No AI response available."
        except Exception as e:
            return f"🤖 AI Error: {e}"


class Chatbot:
    """Main chatbot that integrates CSV search, Google, and Gemini AI"""

    def __init__(self):
        self.data_processor = DataProcessor(CSV_PATH)
        self.search_engine = SearchEngine()
        self.google_result = None  # Used for threading
        self.ai_result = None  # Used for threading

    def fetch_google_result(self, query):
        """Threaded function to fetch Google search results"""
        self.google_result = self.search_engine.google_search(query)

    def fetch_ai_result(self, query):
        """Threaded function to fetch AI response"""
        self.ai_result = self.search_engine.gemini_response(query)

    def get_response(self, user_input):
        """Gets chatbot response with parallel processing"""
        csv_result = self.data_processor.search(user_input)
        if csv_result:
            return f"✅ Here's what I found:\n\n{csv_result}"

        # Run Google & AI searches in parallel
        google_thread = threading.Thread(target=self.fetch_google_result, args=(user_input,))
        ai_thread = threading.Thread(target=self.fetch_ai_result, args=(user_input,))
        google_thread.start()
        ai_thread.start()
        google_thread.join()
        ai_thread.join()

        return self.google_result if "Error" not in self.google_result else self.ai_result


# GUI Class
class ChatbotUI:
    """Handles the Tkinter-based chatbot UI"""

    def __init__(self, root):
        self.bot = Chatbot()

        root.title("AI Chatbot with CSV + Google Search")
        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=25, font=("Arial", 12))
        self.chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_entry = tk.Entry(root, width=70, font=("Arial", 12))
        self.user_entry.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.user_entry.bind("<Return>", self.send_message)  # Bind Enter key to send message

        send_button = tk.Button(root, text="Send", command=self.send_message, font=("Arial", 12))
        send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

    def send_message(self, event=None):
        """Handles user input and bot response"""
        user_input = self.user_entry.get().strip()
        if not user_input:
            return

        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"You: {user_input}\n", "user")
        self.chat_window.insert(tk.END, "Bot: ⏳ Searching...\n", "bot")
        self.chat_window.update_idletasks()  # Refresh UI

        response = self.bot.get_response(user_input)

        self.chat_window.insert(tk.END, f"Bot: {response}\n\n", "bot")
        self.chat_window.config(state=tk.DISABLED)
        self.user_entry.delete(0, tk.END)


# Run the chatbot GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()