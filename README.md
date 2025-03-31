## AI Chatbot 🤖

An intelligent chatbot that searches a CSV and uses Gemini AI to answer questions you may have about soft robotics.

## 🚀 Installation

### 1️⃣ Clone the repository:
```sh
git clone https://github.com/Aneek1/ai-chatbot.git
cd ai-chatbot
```

### 2️⃣ Set up a virtual environment:
```sh
python -m venv spacy_env
source spacy_env/bin/activate  # On macOS/Linux
spacy_env\Scripts\activate    # On Windows
```

### 3️⃣ Install dependencies:
```sh
pip install -r requirements.txt
```

### 4️⃣ Obtain and Set Up Your Gemini API Key
#### a) Get the API Key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Gemini API** in **APIs & Services**.
3. Create an **API Key** in **Credentials**.
4. Copy and store your key securely.

#### b) Configure the API Key:
##### **Option 1 (Recommended): Use an `.env` file**
- Create an `.env` file and add your key:
  ```sh
  echo "GOOGLE_API_KEY=your_api_key_here" > .env
  ```
- Load the API key into your environment:
  ```sh
  export $(cat .env | xargs)
  ```

##### **Option 2: Set API Key as an Environment Variable**
- **Linux/macOS:**
  ```sh
  echo 'export GOOGLE_API_KEY="your_api_key_here"' >> ~/.bashrc
  source ~/.bashrc
  ```
  Or for **Zsh** users:
  ```sh
  echo 'export GOOGLE_API_KEY="your_api_key_here"' >> ~/.zshrc
  source ~/.zshrc
  ```

- **Windows (Command Prompt):**
  ```cmd
  setx GOOGLE_API_KEY "your_api_key_here"
  ```

- **Windows (PowerShell):**
  ```powershell
  echo GOOGLE_API_KEY=your_api_key_here > .env
  ```

##### **Option 3: Set API Key Directly in Python (Not Secure)**
```python
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

### 5️⃣ Run the Chatbot
```sh
python3.11 oop_gui_chatbot.py
```

Now your chatbot is ready to use! 🚀

