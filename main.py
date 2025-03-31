import os
from google import genai
from google.genai import types
from googlesearch import search  # Google Search
# Function to perform Google Search
def google_search(query, num_results=3):
    try:
        results = [url for url in search(query, num_results=num_results)]
        return results
    except Exception as e:
        return [f"Error during search: {e}"]

# Function to generate content using Gemini API
def generate(user_input):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY is not set. Please set it before running the script.")
        return

    try:
        client = genai.Client(api_key=api_key)

        # Search Google for additional context
        search_results = google_search(user_input)
        search_summary = "\n".join(search_results)

        # Construct prompt with search results
        prompt = f"{user_input}\n\nRelevant Google Search results:\n{search_summary}"

        model = "gemini-2.0-flash"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

        # Generate response
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            print(chunk.text, end="")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main execution
if __name__ == "__main__":
    user_input = input("Please type your query: ")
    generate(user_input)