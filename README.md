# DeepArticle AI

DeepArticle AI fetches full-text articles from arbitrary URLs, builds or loads a vector store for semantic retrieval, and then uses AWS Bedrock and/or Claude to answer user questions based solely on the article’s content. When vector embeddings cannot be created, it seamlessly falls back to direct LLM querying to ensure reliability.

## Table of Contents
1. [Description](#description)  
2. [Features](#features)  
3. [Technology Stack](#technology-stack)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [Configuration](#configuration)  
7. [Running the App](#running-the-app)  
8. [Deployment](#deployment)  
9. [Contact](#contact)  

## Description
DeepArticle AI fetches full-text articles from arbitrary URLs, builds or loads a vector store for semantic retrieval, and then uses AWS Bedrock and/or Claude to answer user questions based solely on the article’s content. When vector embeddings cannot be created, it seamlessly falls back to direct LLM querying to ensure reliability.

## Features
- **URL-based ingestion** of any public article.
- **Vector store creation** with fallback to direct API if embeddings fail.
- **Retrieval-augmented generation** using LangChain’s `RetrievalQA`.
- **Streamlit UI** for interactive querying and real-time status messages.
- **AWS Bedrock** integration for both embedding and generation.

## Technology Stack
- **Frontend:** Streamlit  
- **Backend & LLM integration:** Python, `boto3`, LangChain  
- **Vector store:** Chroma (or equivalent)  
- **Cloud Services:** AWS Bedrock (embeddings + Claude API)  
- **Version Control:** Git & GitHub  

## Prerequisites
- Python 3.12+ installed locally.  
- AWS account with Bedrock access and proper IAM permissions (e.g., `bedrock:InvokeModel`).  
- Streamlit CLI (`pip install streamlit`).  

## Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/rishavkr101/DeepArticle.AI
   cd DeepArticle.AI
   ```  
2. **Create & activate a virtualenv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```  
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  

## Configuration
1. **Streamlit secrets**  
   Create or update `.streamlit/secrets.toml`:  
   ```toml
   [aws]
   aws_access_key_id = "YOUR_KEY_ID"
   aws_secret_access_key = "YOUR_SECRET"
   ```  
2. **Region & Models**  
   Ensure `region_name="us-east-1"` matches where your Bedrock models live.  
3. **IAM Role Permissions**  
   Attach a policy allowing at minimum:
   ```json
   {
     "Effect": "Allow",
     "Action": [
       "bedrock:InvokeModel",
       "bedrock:InvokeModelWithResponseStream",
       "sts:GetCallerIdentity"
     ],
     "Resource": "*"
   }
   ```  

## Running the App
Launch locally with:
```bash
streamlit run app.py
```
Then navigate to http://localhost:8501 to interact with DeepArticle AI.

## Deployment
- **Streamlit Cloud:**  
  Push to GitHub and connect your repo in Streamlit Cloud.  
- **Docker (optional "can be used but i have not used it in this project"):**  
  ```bash
  docker build -t DeepArticle.AI .
  docker run -p 8501:8501 DeepArticle.AI
  ```


## Contact
- **Project Maintainer:** Rishavkumar (<rishavroy39rk@gmail.com>)  
