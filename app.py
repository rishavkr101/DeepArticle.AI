import streamlit as st
import boto3
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import uuid
from urllib.parse import urlparse
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import BedrockChat


# Configure page settings
st.set_page_config(
    page_title="DeepArticle AI",
    layout="wide"
)


import streamlit as st

def setup_aws_credentials():
    try:
        # Use the secrets in your session creation
        session = boto3.Session(
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name='us-east-1'
        )
        
        # Test the credentials
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        st.sidebar.success(f"AWS authenticated as: {identity['Arn']}")
        
        # Create Bedrock runtime client
        bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        return bedrock_runtime
    except Exception as e:
        st.sidebar.error(f"AWS credentials error: {str(e)}")
        st.sidebar.error("Please check your AWS credentials in Streamlit secrets")
        return None



# Initialize the session state variables if they don't exist
if 'article_content' not in st.session_state:
    st.session_state.article_content = ""
if 'article_url' not in st.session_state:
    st.session_state.article_url = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'article_vector_store' not in st.session_state:
    st.session_state.article_vector_store = None
if 'article_chunks' not in st.session_state:
    st.session_state.article_chunks = []

# AWS clients setup
def init_bedrock_client():
    try:
        #getting credentials from config
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        return bedrock_runtime
    except Exception as e:
        st.error(f"Error initializing AWS Bedrock client: {str(e)}")
        st.info("Please ensure your AWS credentials are properly configured.")
        return None

# Initializing bedrock embeddings
def init_bedrock_embeddings():
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v2:0"
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing Bedrock embeddings: {str(e)}")
        return None

# Initializing LLM
def init_llm():
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        # Initializing callback manager for streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initializing the Bedrock LLM
        llm = BedrockChat(
            client=bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            callback_manager=callback_manager,
            model_kwargs={
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0.6,
                "max_tokens": 1000
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Function to extract text from the given URL
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Removeing script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Clean up text with regex
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
        text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
        
        return text
    except Exception as e:
        return f"Error extracting text from URL: {str(e)}"

# Function to chunk text and create vector store
def create_vector_store(text, url):
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Add metadata to chunks
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    "source": url,
                    "chunk": i
                }
            })
        
        # Store chunks for later use
        st.session_state.article_chunks = chunk_docs
        
        # Get embeddings
        embeddings = init_bedrock_embeddings()
        if not embeddings:
            return None
        
        # Create vector store from chunks
        texts = [doc["text"] for doc in chunk_docs]
        metadatas = [doc["metadata"] for doc in chunk_docs]
        
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to query using vector store
def query_with_vector_store(vector_store, question):
    try:
        llm = init_llm()
        if not llm:
            return "Error initializing LLM. Check your AWS credentials."
        
        # Create memory for maintaining conversation context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt template
        template = """
        You are an equity research assistant analyzing financial articles.
        
        Use the following context from an equity-related article to answer the question.
        If you don't know the answer based on the context, say that you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Run the chain
        response = qa_chain({"query": question})
        
        # Extract answer and source documents
        answer = response["result"]
        source_docs = response.get("source_documents", [])
        
        # Format source information if available
        if source_docs:
            source_info = "\n\nSources:\n"
            for i, doc in enumerate(source_docs):
                chunk_num = doc.metadata.get("chunk", "Unknown")
                source_info += f"- Chunk {chunk_num} from article\n"
            answer += source_info
            
        return answer
        
    except Exception as e:
        return f"Error querying vector store: {str(e)}"

# Original Bedrock query function (as fallback)
def query_bedrock(article_content, question, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    client = init_bedrock_client()
    if not client:
        return "AWS Bedrock client initialization failed. Check your credentials."
    
    try:
        # Format the prompt for Claude
        prompt = f"""
        You are an equity research assistant analyzing financial articles.
        
        Here is an article:
        ---
        {article_content[:30000]}  # Limit content length to respect model's token limits
        ---
        
        Question about the article: {question}
        
        Please answer the question based solely on the information in the article. If the article doesn't contain relevant information to answer the question, clearly state that the information is not available in the article.
        """
        
        # Prepare the payload for Claude model
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        # Convert the request body to JSON
        request_body_json = json.dumps(request_body)
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=request_body_json
        )
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        answer = response_body.get('content')[0].get('text')
        return answer
    
    except Exception as e:
        return f"Error querying AWS Bedrock: {str(e)}"

# App header
st.title("🤖 DeepArticle AI")
st.subheader("Upload article links and ask questions powered by AWS Bedrock")



# Sidebar for URL input
with st.sidebar:
    st.header("Article Source")
    article_url = st.text_input("Enter the URL of an article:", key="url_input")
    
    if st.button("Load Article"):
        if article_url:
            with st.spinner("Extracting article content..."):
                
                try:
                    result = urlparse(article_url)
                    if all([result.scheme, result.netloc]):
                        article_text = extract_text_from_url(article_url)
                        if article_text.startswith("Error"):
                            st.error(article_text)
                        else:
                            st.session_state.article_content = article_text
                            st.session_state.article_url = article_url
                            
                            # Creating vector store
                            with st.spinner("Creating vector embeddings for semantic search..."):
                                vector_store = create_vector_store(article_text, article_url)
                                if vector_store:
                                    st.session_state.article_vector_store = vector_store
                                    st.success(f"Article loaded and vectorized successfully! ({len(article_text)} characters, {len(st.session_state.article_chunks)} chunks)")
                                else:
                                    st.warning("Article loaded but vectorization failed. Falling back to direct querying.")
                                    st.success(f"Article loaded successfully! ({len(article_text)} characters)")
                    else:
                        st.error("Please enter a valid URL")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a URL to load an article")
    
    st.divider()
    
    # Vector store info
    if st.session_state.article_vector_store:
        st.success("Vector store active")
        if st.button("Clear Vector Store"):
            st.session_state.article_vector_store = None
            st.session_state.article_chunks = []
            st.experimental_rerun()
    


# Main content area
col1, col2 = st.columns([2, 3])

# Display article preview
with col1:
    st.header("Article Preview")
    if st.session_state.article_content:
        url_display = st.session_state.article_url
        st.markdown(f"**Source:** [{url_display}]({url_display})")
        st.text_area("Content Preview:", 
                    value=st.session_state.article_content[:1000] + 
                    ("..." if len(st.session_state.article_content) > 1000 else ""),
                    height=300, 
                    disabled=True)
        


# Q&A area
with col2:
    st.header("Ask Questions About the Article")
    
    if st.session_state.article_content:
        user_question = st.text_input("Ask a question about the article:", placeholder="What are the key insights about this equity?")
        
        if st.button("Get Answer") and user_question:
            with st.spinner("Analyzing the article..."):
                if st.session_state.article_vector_store:
                    # Use vector store for semantic search and retrieval
                    answer = query_with_vector_store(st.session_state.article_vector_store, user_question)
                else:
                    # Fallback to direct querying
                    answer = query_bedrock(st.session_state.article_content, user_question)
                
                # Add to chat history
                st.session_state.chat_history.append({"question": user_question, "answer": answer})
    else:
        st.warning("Please load an article first before asking questions.")
    
    # Display chat history
    st.subheader("Question History")
    if st.session_state.chat_history:
        for i, exchange in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {exchange['question']}", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Answer:**\n{exchange['answer']}")
    else:
        st.info("Your question history will appear here.")

# Footer
st.divider()
st.caption("Equity Article Analyzer - Powered by AWS Bedrock and Streamlit with Vector Search")
