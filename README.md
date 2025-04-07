# AI Chatbot 🤖

An intelligent chatbot that searches a CSV and uses Gemini AI to answer questions you may have about soft robotics.

## 🚀 Installation
## Please do install GIT and Conda before following the steps below.

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/Aneek1/ai-chatbot-softrobotics.git
cd ai-chatbot-softrobotics
```

### 2️⃣ Set up a Conda environment (Recommended - Cross-Platform):

#### What is Conda?
Conda is a versatile tool for managing software packages and environments. It works similarly to Python's `venv` but is language-agnostic and designed to handle binary dependencies, which can be crucial for scientific computing and machine learning projects. It creates isolated spaces for your projects, preventing conflicts between different software versions and ensuring reproducibility across different operating systems (macOS, Windows, Linux).

#### Create and activate the environment:

```bash
conda create -n spacy_conda python=3.11  # Creates an environment named 'spacy_conda' with Python 3.11
conda activate spacy_conda               # Activates the 'spacy_conda' environment
```

#### Why Conda?
Using Conda is highly recommended for this project due to its robust dependency management, cross-platform compatibility, and ability to handle complex environments. This ensures a smoother installation process and reduces the likelihood of encountering errors related to conflicting libraries or system configurations.

### 3️⃣ Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4️⃣ Obtain and Set Up Your Gemini API Key

#### a) Get the API Key:
1. Go to **Google Cloud Console**.
2. Enable the **Gemini API** in **APIs & Services**.
3. Create an API Key in **Credentials**.
4. Copy and store your key securely.

#### b) Configure the API Key (Cross-Platform):

##### macOS / Linux (Bash):
```bash
echo 'export GOOGLE_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc  # Or ~/.zshrc for Zsh
source ~/.bashrc  # Or source ~/.zshrc
```

##### Windows (Command Prompt):
```cmd
setx GOOGLE_API_KEY "YOUR_API_KEY_HERE"
```

##### Windows (PowerShell):
```powershell
$Env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

### 5️⃣ Run the Bot and Enjoy! 🎉

```bash
python newcodev1.py  # Created using OOP for faster response time
```
### 6. Debugging

```bash
pip install "module name"  # Please install the modules separately if you encounter any errors during installation.
conda install -c conda-forge spacy #Please Run this command before installing the spacy model
python -m spacy download en_core_web_sm  #installs the spacy NLP model 

```
