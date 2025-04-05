import os
import pandas as pd
import spacy
import threading
import google.generativeai as genai
from googleapiclient.discovery import build
import tkinter as tk
from tkinter import scrolledtext, simpledialog

# Load Spacy NLP model once
nlp = spacy.load("en_core_web_sm")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"  # Replace with actual ID
CSV_PATH = "expanded_soft_robotics_fabrication_methods.csv"

# Keywords to determine relevance to soft robotics fabrication
SOFT_ROBOTICS_KEYWORDS = {"soft robotics", "fabrication", "molding", "casting", "3D printing", "elastomers", "actuators", "silicone", "pneumatics", "inflatable", "deformation", "stretchable", "compliance", "hydrogels"}

# Confidence level mapping for simplification
CONFIDENCE_SIMPLIFICATION = {
    "low": "Provide a very basic overview.",
    "medium": "Provide a moderately detailed explanation.",
    "high": "Provide a comprehensive and detailed explanation."
}

class DataProcessor:
    """Handles CSV file loading and searching"""
    def __init__(self, csv_path):
        self.data = self.load_csv(csv_path)

    def load_csv(self, path):
        """Load CSV into a dictionary for faster lookups"""
        if os.path.exists(path):
            df = pd.read_csv(path)
            print("✅ CSV loaded successfully!")
            return df.to_dict(orient="records")
        print("❌ No CSV found!")
        return None

    def extract_keywords(self, query):
        """Extract meaningful keywords from the input (using token.text)"""
        doc = nlp(query.lower())
        keywords = {token.text for token in doc if token.is_alpha and not token.is_stop}
        print(f"Extracted keywords from query (using token.text): {keywords}")
        return keywords

    def is_relevant_to_soft_robotics(self, query):
        """Check if the query is related to soft robotics fabrication"""
        query_keywords = self.extract_keywords(query)
        relevant = any(keyword in SOFT_ROBOTICS_KEYWORDS for keyword in query_keywords)
        print(f"Is query relevant to soft robotics? {relevant} (Keywords: {query_keywords}, Soft Robotics Keywords: {SOFT_ROBOTICS_KEYWORDS})")
        return relevant

    def search(self, query):
        """Fast CSV lookup using preloaded data"""
        if not self.data:
            return None

        if not self.is_relevant_to_soft_robotics(query):
            return None  # Skip CSV search if the query is not related to soft robotics

        keywords = self.extract_keywords(query)
        matched_rows = []
        print("Searching CSV with keywords:", keywords)
        if self.data:
            print("Sample CSV data (first 5 rows):")
            for i in range(min(5, len(self.data))):
                print(f"  Method: '{self.data[i]['fabrication_method']}'")
                print(f"  Description: '{self.data[i]['description']}'")
            print("-" * 20)

            for row in self.data:
                method_lower = str(row['fabrication_method']).lower()
                description_lower = str(row['description']).lower()
                if any(keyword in method_lower or keyword in description_lower for keyword in keywords):
                    matched_rows.append(
                        f"**{row['fabrication_method']}**:\n"
                        f"{row['description']}\n"
                        f"Materials: {row['materials']}\n"
                        f"Properties: {row['properties']}\n"
                        f"Steps: {row['steps_to_fabricate']}\n"
                        f"Time: {row['time_taken']}\n"
                        f"Pros: {row['advantages']}\n"
                        f"Cons: {row['disadvantages']}\n"
                        f"Application: {row['application_example']}\n"
                        "---"
                    )
                    print(f"Found match in CSV: Method='{row['fabrication_method']}', Description='{row['description']}'")
        else:
            print("CSV data is not loaded.")

        return "\n\n".join(matched_rows) if matched_rows else None

class SearchEngine:
    """Handles Google Search and Gemini AI queries"""
    def __init__(self):
        self.google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)

    def google_search(self, query):
        """Perform Google search"""
        try:
            res = self.google_service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()
            results = res.get("items", [])
            return "\n".join([f"🔗 {item['title']}: {item['link']}" for item in results[:3]]) if results else "No Google results found."
        except Exception as e:
            return f"Google Search Error: {e}"

    def gemini_response(self, query, simplification_prompt=""):
        """Generate response using Gemini AI"""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"{simplification_prompt} {query}"
            response = model.generate_content(prompt)
            return response.text if response and response.text else "No AI response available."
        except Exception as e:
            return f"AI Error: {e}"

    def search_youtube(self, query):
        """Search for YouTube videos using Gemini AI"""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"Search YouTube for videos related to: '{query}'. Provide a concise list of 2-3 relevant video titles and their URLs."
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                return "No YouTube video suggestions found."
        except Exception as e:
            return f"YouTube Search Error: {e}"

class Chatbot:
    """Main chatbot that integrates CSV search, Google, and Gemini AI"""
    def __init__(self):
        self.data_processor = DataProcessor(CSV_PATH)
        self.search_engine = SearchEngine()
        self.google_result = None
        self.ai_result = None
        self.youtube_result = None
        self.greeting_responded = False # Track if the greeting has been responded to

    def fetch_google_result(self, query):
        """Threaded function to fetch Google search results"""
        self.google_result = self.search_engine.google_search(query)

    def fetch_ai_result(self, query, simplification_prompt):
        """Threaded function to fetch AI response"""
        self.ai_result = self.search_engine.gemini_response(query, simplification_prompt)

    def fetch_youtube_result(self, query):
        """Threaded function to fetch YouTube search results"""
        self.youtube_result = self.search_engine.search_youtube(query)

    def get_response(self, user_input, confidence):
        """Gets chatbot response with parallel processing"""
        simplification_prompt = CONFIDENCE_SIMPLIFICATION.get(confidence.lower(), "")
        print(f"User confidence: {confidence.upper()}. Simplification prompt: '{simplification_prompt}'")

        if not self.greeting_responded and user_input.lower() in ["hello", "hi", "hey"]:
            self.greeting_responded = True
            return "Hello! How may I help you today?"

        csv_result = self.data_processor.search(user_input)

        # Start Google & AI searches in parallel
        google_thread = threading.Thread(target=self.fetch_google_result, args=(user_input,))
        ai_thread = threading.Thread(target=self.fetch_ai_result, args=(user_input, simplification_prompt))
        youtube_thread = threading.Thread(target=self.fetch_youtube_result, args=(user_input,))

        google_thread.start()
        ai_thread.start()
        youtube_thread.start()

        google_thread.join()
        ai_thread.join()
        youtube_thread.join()

        combined_response = ""

        if csv_result:
            combined_response += f"{csv_result}\n\n"

        if self.ai_result and "Error" not in self.ai_result:
            combined_response += f"{self.ai_result}\n\n"
        elif self.ai_result and "Error" in self.ai_result:
            print(f"AI Error: {self.ai_result}") # Print error to backend

        if self.google_result and "Error" not in self.google_result:
            combined_response += f"Google Search Results:\n{self.google_result}\n\n"
        elif self.google_result and "Error" in self.google_result:
            print(f"Google Search Error: {self.google_result}") # Print error to backend

        if self.youtube_result and "Error" not in self.youtube_result:
            combined_response += f"YouTube Videos:\n{self.youtube_result}\n\n"
        elif self.youtube_result and "Error" in self.youtube_result:
            print(f"YouTube Search Error: {self.youtube_result}") # Print error to backend

        final_response = combined_response.strip()
        print(f"Final Combined Response: '{final_response}'")
        return final_response if final_response else "No relevant results found."

class ChatbotUI:
    """Handles the Tkinter-based chatbot UI"""
    def __init__(self, root):
        self.bot = Chatbot()
        self.root = root
        root.title("AI Chatbot")
        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=25, font=("Arial", 12))
        self.chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.user_entry = tk.Entry(root, width=70, font=("Arial", 12))
        self.user_entry.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.user_entry.bind("<Return>", self.send_message)
        self.send_button = tk.Button(root, text="Send", command=self.send_message, font=("Arial", 12))
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")
        self.loading_message_index_start = None
        self.loading_message_index_end = None

    def ask_confidence_level(self):
        """Opens a simple dialog to ask for the user's confidence level."""
        level = simpledialog.askstring("Confidence Level", "How confident are you with this topic? (low / medium / high)")
        return level

    def send_message(self, event=None):
        """Handles user input and bot response with loading indicator and confidence prompt"""
        user_input = self.user_entry.get().strip()
        if not user_input:
            return

        # Handle greeting separately before asking for confidence
        if not self.bot.greeting_responded and user_input.lower() in ["hello", "hi", "hey"]:
            self.chat_window.config(state=tk.NORMAL)
            self.chat_window.insert(tk.END, f"You: {user_input}\n", "user")
            self.user_entry.delete(0, tk.END)
            bot_response = self.bot.get_response(user_input, "") # Confidence not needed for greeting
            self.chat_window.insert(tk.END, f"Bot:\n{bot_response}\n\n", "bot")
            self.chat_window.config(state=tk.DISABLED)
            self.chat_window.see(tk.END)
            return

        confidence_level = self.ask_confidence_level()
        if not confidence_level or confidence_level.lower() not in ["low", "medium", "high"]:
            self.chat_window.config(state=tk.NORMAL)
            self.chat_window.insert(tk.END, "Bot: Please enter a valid confidence level: low, medium, or high.\n", "bot_error")
            self.chat_window.config(state=tk.DISABLED)
            return

        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"You: {user_input} (Confidence: {confidence_level})\n", "user")
        self.user_entry.delete(0, tk.END)

        # Disable input and show loading message
        self.user_entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.chat_window.insert(tk.END, "Bot: Thinking...\n\n", "bot_loading")
        # Get the index of the inserted "Thinking..." message
        self.loading_message_index_start = self.chat_window.index(tk.END + '-2l linestart') # Get index of the start of the "Bot: Thinking..." line
        self.loading_message_index_end = self.chat_window.index(tk.END + '-1l linestart')   # Get index of the start of the line after

        self.chat_window.config(state=tk.DISABLED)
        self.chat_window.see(tk.END)

        # Process the message in a separate thread to avoid blocking the UI
        threading.Thread(target=self.process_message, args=(user_input, confidence_level)).start()

    def process_message(self, user_input, confidence_level):
        """Processes the user message and updates the UI with loading indicators"""
        self.chat_window.config(state=tk.NORMAL)
        if self.loading_message_index_start and self.loading_message_index_end:
            try:
                self.chat_window.delete(self.loading_message_index_start, self.loading_message_index_end)
            except tk.TclError as e:
                print(f"Error deleting loading message: {e}")
                self.chat_window.delete(tk.END + '-2l', tk.END + '-1l')

        # Get the full response from the bot, which now handles CSV search internally
        bot_response = self.bot.get_response(user_input, confidence_level)
        self.chat_window.insert(tk.END, f"Bot:\n{bot_response}\n\n", "bot")

        self.chat_window.config(state=tk.DISABLED)
        self.user_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.chat_window.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()