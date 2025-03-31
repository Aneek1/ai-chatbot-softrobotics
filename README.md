\# AI Chatbot 🤖

An intelligent chatbot that searches a CSV and uses Gemini AI to answer questions you may have about soft robotics.

\#\# 🚀 Installation

\#\#\# 1️⃣ Clone the repository:

\`\`\`bash  
git clone \[https://github.com/Aneek1/ai-chatbot-softrobotics\](https://github.com/Aneek1/ai-chatbot-softrobotics)  
cd ai-chatbot-softrobotics

### **2️⃣ Set up a Conda environment (Recommended \- Cross-Platform):**

**What is Conda?**  
Conda is a versatile tool for managing software packages and environments. It works similarly to Python's venv but is language-agnostic and designed to handle binary dependencies, which can be crucial for scientific computing and machine learning projects. It creates isolated spaces for your projects, preventing conflicts between different software versions and ensuring reproducibility across different operating systems (macOS, Windows, Linux).  
**Create and activate the environment:**  
conda create \-n spacy\_conda python=3.11  \# Creates an environment named 'spacy\_conda' with Python 3.11  
conda activate spacy\_conda                 \# Activates the 'spacy\_conda' environment

**Why Conda?**  
Using Conda is highly recommended for this project due to its robust dependency management, cross-platform compatibility, and ability to handle complex environments. This ensures a smoother installation process and reduces the likelihood of encountering errors related to conflicting libraries or system configurations.

### **3️⃣ Install dependencies:**

pip install \-r requirements.txt

### **4️⃣ Obtain and Set Up Your Gemini API Key**

#### **a) Get the API Key:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/).  
2. Enable the **Gemini API** in **APIs & Services**.  
3. Create an **API Key** in **Credentials**.  
4. Copy and store your key securely.

#### **b) Configure the API Key (Cross-Platform):**

##### **Option 1 (Recommended): Use an .env file**

* Create an .env file in the project's root directory and add your key:  
  echo "GOOGLE\_API\_KEY=YOUR\_API\_KEY\_HERE" \> .env

  (This command works on both macOS/Linux and Windows (PowerShell). For Command Prompt on Windows, use echo GOOGLE\_API\_KEY=YOUR\_API\_KEY\_HERE \> .env)  
* Load the API key into your environment:  
  export $(cat .env | xargs)

  (This command works on macOS/Linux and in Git Bash on Windows. If you are using regular Command Prompt on Windows, you will need to set the environment variable using set or setx, as shown in Option 2.)

##### **Option 2: Set API Key as an Environment Variable (OS Specific \- Less Recommended)**

* **macOS/Linux:**  
  echo 'export GOOGLE\_API\_KEY="YOUR\_API\_KEY\_HERE"' \>\> \~/.bashrc  \# Or \~/.zshrc for Zsh  
  source \~/.bashrc \# Or source \~/.zshrc

* **Windows (Command Prompt):**  
  setx GOOGLE\_API\_KEY "YOUR\_API\_KEY\_HERE"

  (Note: setx sets the variable permanently, but requires opening a new Command Prompt. set sets it only for the current session.)  
* **Windows (PowerShell):**  
  $Env:GOOGLE\_API\_KEY \= "YOUR\_API\_KEY\_HERE"

##### **Option 3: Set API Key Directly in Python (Not Secure \- Cross-Platform, Avoid)**

import os  
os.environ\["GOOGLE\_API\_KEY"\] \= "YOUR\_API\_KEY\_HERE"

### **5️⃣ Run the Chatbot**

python oop\_gui\_chatbot.py

Now your chatbot is ready to use\! 🚀  
**Notes:**

* This README provides cross-platform instructions for setting up the chatbot on both macOS and Windows.  
* We strongly recommend using Conda for environment management, as it simplifies the installation process and ensures consistency across different operating systems. Conda is available on macOS, Windows, and Linux.  
* The Python version is explicitly specified as 3.11 for the Conda environment.  
* The preferred method for setting the Gemini API key is using an .env file (Option 1), as it is cross-platform and generally considered good practice.  
* If you choose to set the API key as an environment variable (Option 2), the instructions are tailored to the specific operating system.  
* Ensure Conda is installed on your system before running the Conda commands. You can download and install it from \[Anaconda
