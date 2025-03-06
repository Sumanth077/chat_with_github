import streamlit as st
import os
import tempfile
from dataclasses import dataclass
from typing import List
import time
import gc
import uuid
from embedchain import App
from embedchain.loaders.github import GithubLoader

# Set environment variables
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]

# Create a temporary directory for the database
DB_PATH = tempfile.mkdtemp()

st.set_page_config(
    page_title="Chat with GitHub Repositories",
    page_icon="</>",
    layout="wide"
)

@dataclass
class QA:
    """A question and answer pair."""
    question: str
    answer: str

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "app_instance" not in st.session_state:
    # Initialize the app instance once at startup
    st.session_state.app_instance = App.from_config(
        config={
            "llm": {
                "provider": "clarifai",
                "config": {
                    "model": "https://clarifai.com/openbmb/miniCPM/models/MiniCPM3-4B",
                    "model_kwargs": {
                        "temperature": 0.5,
                        "max_tokens": 1000
                    }
                }
            },
            "vectordb": {"provider": "chroma", "config": {"dir": DB_PATH}},
            "embedder": {
                "provider": "clarifai",
                "config": {
                    "model": "https://clarifai.com/openai/embed/models/text-embedding-ada",
                }
            }
        }
    )
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""
if "repo_added" not in st.session_state:
    st.session_state.repo_added = False
if "current_repo" not in st.session_state:
    st.session_state.current_repo = ""
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

def get_loader():
    """Get GitHub loader with token."""
    return GithubLoader(config={"token": GITHUB_TOKEN})

def footer(st):
    with open('footer.html', 'r') as file:
        footer = file.read()
        st.write(footer, unsafe_allow_html=True)

def handle_repo_input(repo_url: str):
    """Handle repository addition."""
    if not repo_url:
        st.warning("Please enter a valid GitHub repository URL")
        return
    st.session_state.processing = True
    
    try:
        # Extract owner/repo format from URL if needed
        if "github.com" in repo_url:
            parts = repo_url.strip('/').split('/')
            repo = f"{parts[-2]}/{parts[-1]}"
        else:
            repo = repo_url
            
        with st.spinner(f"Adding repository {repo} to knowledge base..."):
            loader = get_loader()
            st.session_state.app_instance.add(f"repo:{repo} type:repo", data_type="github", loader=loader)
            st.session_state.upload_status = f"✅ Repository loaded successfully!"
            st.session_state.repo_added = True
            st.session_state.current_repo = repo
            # Clear any previous messages
            st.session_state.messages = []
    except Exception as e:
        st.session_state.upload_status = f"❌ Error: {str(e)}"
    finally:
        st.session_state.processing = False

def reset_chat():
    """Clear the chat history."""
    st.session_state.messages = []
    gc.collect()

def clear_repository():
    """Clear the current repository and reset the chat."""
    st.session_state.repo_added = False
    st.session_state.current_repo = ""
    st.session_state.messages = []
    st.session_state.upload_status = ""
    gc.collect()

# Utility function to convert generator to text
def get_generator_text(generator):
    """Convert a generator response to text."""
    try:
        # Attempt to join the generator if it's an iterable
        if hasattr(generator, '__iter__'):
            return ''.join(str(chunk) for chunk in generator)
        # If it's not an iterable, return string representation
        return str(generator)
    except Exception as e:
        return f"Error processing response: {str(e)}"

# Sidebar for repository input
with st.sidebar:
    st.header("Add your GitHub repository!")
    
    github_url = st.text_input("Enter GitHub repository URL or owner/repo", placeholder="https://github.com/owner/repo or owner/repo")

    load_repo = st.button("Load Repository")
    if github_url and load_repo:
        handle_repo_input(github_url)
        
    # Show status message if any
    if st.session_state.upload_status:
        st.markdown(st.session_state.upload_status)

# Main content area
col1, col2 = st.columns([6, 1])
with col1:
    st.header(f"Chat with GitHub 💬")
with col2:
    if st.session_state.repo_added:
        st.button("Clear ↺", on_click=reset_chat)

# Display current repository info if loaded
if st.session_state.repo_added:
    st.info(f"Active Repository: **{st.session_state.current_repo}**")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Process the question
            if st.session_state.processing:
                message_placeholder.markdown("Thinking...")
            else:
                st.session_state.processing = True
                
                try:
                    # Get the answer (streaming isn't supported in this implementation)
                    answer_generator = st.session_state.app_instance.chat(prompt)
                    answer = get_generator_text(answer_generator)
                    print(f"Answer: {answer}")
                    
                    # Display the answer
                    message_placeholder.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    message_placeholder.markdown(f"Error: {str(e)}")
                finally:
                    st.session_state.processing = False
else:
    # Show welcome message when no repository is loaded
    st.markdown("""
    ### Welcome to GitHub Repository Chat! 👋
    
    This app allows you to chat with any GitHub repository using RAG (Retrieval-Augmented Generation).
    
    To get started:
    1. Enter a GitHub repository URL in the sidebar
    2. Click "Load Repository"
    3. Start asking questions about the repository!
    
    The app will process the repository contents and allow you to have a conversation about its code, structure, and functionality.
    """)

# Remove the main() function call and keep the footer
footer(st)