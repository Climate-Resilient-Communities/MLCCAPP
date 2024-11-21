import streamlit as st
import time
import asyncio
from main import MultilingualClimateChatbot
import json

@st.cache_resource
def init_chatbot():
    return MultilingualClimateChatbot(
        index_name="climate-change-adaptation-index-10-24-prod",
        input_jsonl_path="./data/climate_docs.jsonl"
    )

# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Multilingual Climate Chatbot")

def get_citation_details(citation):
    """Safely extract citation details."""
    try:
        # Handle citations from the Cohere response
        if hasattr(citation, 'sources') and citation.sources:
            source = citation.sources[0]
            if hasattr(source, 'document'):
                doc = source.document
                return {
                    'title': doc.get('title', 'Untitled Source'),
                    'url': doc.get('url', ''),
                    'snippet': doc.get('snippet', '')
                }
        # Handle dictionary-style citations
        elif isinstance(citation, dict):
            return {
                'title': citation.get('title', 'Untitled Source'),
                'url': citation.get('url', ''),
                'snippet': citation.get('content', citation.get('snippet', ''))
            }
    except Exception as e:
        st.error(f"Error processing citation: {str(e)}")
    
    return {
        'title': 'Untitled Source',
        'url': '',
        'snippet': ''
    }

def display_source_citations(citations, base_idx=0):
    """Display citations in a visually appealing way."""
    st.write("### Sources")
    
    # Create a dictionary to store unique sources
    unique_sources = {}
    
    for citation in citations:
        details = get_citation_details(citation)
        
        # Use title as key for deduplication
        if details['title'] not in unique_sources:
            unique_sources[details['title']] = details
    
    # Display each unique source
    for idx, (title, source) in enumerate(unique_sources.items()):
        with st.container():
            # Create a unique key using the message index and source index
            unique_key = f"source_{base_idx}_{idx}"
            if st.button(f"📄 {title[:100]}...", key=unique_key):
                st.session_state.selected_source = f"{base_idx}_{title}"
            
            # Show details if selected
            if st.session_state.get('selected_source') == f"{base_idx}_{title}":
                with st.expander("Source Details", expanded=True):
                    st.markdown(f"**Title:** {title}")
                    if source['url']:
                        st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                    if source['snippet']:
                        st.markdown("**Cited Content:**")
                        st.markdown(source['snippet'])

def display_progress(progress_placeholder):
    """Display simple progress bar."""
    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()
    
    stages = [
        ("🔍 Searching...", 0.2),
        ("📚 Retrieving documents...", 0.4),
        ("✍️ Generating response...", 0.7),
        ("✔️ Verifying response...", 0.9),
        ("✨ Complete!", 1.0)
    ]
    
    for stage_text, progress in stages:
        status_text.text(stage_text)
        progress_bar.progress(progress)
        time.sleep(0.5)  # Brief pause between stages
        
    progress_placeholder.empty()

def display_chat_messages():
    """Display chat messages in a custom format."""
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.chat_message("user").markdown(message['content'])
        else:
            assistant_message = st.chat_message("assistant")
            assistant_message.markdown(message['content'])
            if message.get('citations'):
                # Pass the message index to ensure unique keys
                display_source_citations(message['citations'], base_idx=i)

def load_custom_css():
    st.markdown("""
    <style>
    /* Previous CSS styles remain the same */

    /* Add styles for the download button */
    .download-button {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        text-decoration: none;
        margin-left: 10px;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_chat_history_text():
    """Convert chat history to downloadable text format."""
    history_text = "Chat History\n\n"
    for msg in st.session_state.chat_history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"
        if msg.get('citations'):
            history_text += "Sources:\n"
            for citation in msg['citations']:
                details = get_citation_details(citation)
                history_text += f"- {details['title']}\n"
                if details['url']:
                    history_text += f"  URL: {details['url']}\n"
                if details['snippet']:
                    history_text += f"  Content: {details['snippet']}\n"
            history_text += "\n"
    return history_text

def display_chat_history_section():
    """Display chat history with download button."""
    if st.session_state.chat_history:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown("### Chat History")
        with col2:
            # Create download button for chat history
            chat_history_text = generate_chat_history_text()
            st.download_button(
                label="📥",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain",
                help="Download chat history"
            )
        
        messages = st.session_state.chat_history
        for idx in range(0, len(messages), 2):
            if messages[idx]['role'] == 'user':
                q = messages[idx]['content']
                if idx + 1 < len(messages) and messages[idx + 1]['role'] == 'assistant':
                    r = messages[idx + 1]['content']
                else:
                    r = ''
                with st.expander(f"Q: {q[:50]}...", expanded=False):
                    st.write("**Question:**")
                    st.write(q)
                    st.write("**Response:**")
                    st.write(r)

def main():
    load_custom_css()
    
    # Initialize session state
    if 'selected_source' not in st.session_state:
        st.session_state.selected_source = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'has_asked_question' not in st.session_state:
        st.session_state.has_asked_question = False
    if 'language_confirmed' not in st.session_state:
        st.session_state.language_confirmed = False
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'english'
    
    chatbot = init_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="content">', unsafe_allow_html=True)
        
        st.title('Multilingual Climate Chatbot')
        
        # Language selection and confirmation
        st.write("**Please choose your preferred language to get started:**")
        languages = sorted(chatbot.LANGUAGE_NAME_TO_CODE.keys())
        default_index = languages.index(st.session_state.selected_language)
        selected_language = st.selectbox(
            "Select your language",
            options=languages,
            index=default_index
        )
        
        if not st.session_state.language_confirmed:
            if st.button("Confirm"):
                st.session_state.language_confirmed = True
                st.session_state.selected_language = selected_language
        else:
            st.session_state.selected_language = selected_language
        
        # Display About section or Chat History
        if not st.session_state.has_asked_question:
            st.markdown("## About")
            st.markdown('''
                The purpose of this app is to educate individuals about climate change and foster a community of informed citizens. It provides accurate information and resources about climate change and its impacts, and encourages users to take action in their own communities.
            ''')
        else:
            st.markdown("---")
            display_chat_history_section()
        
        # Footer
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        st.markdown('<div>Made by:</div>', unsafe_allow_html=True)
        st.image("tree.ico", width=40)
        st.markdown('<div style="font-size: 18px;">Climate Resilient Communities</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("CCCicon.png", width=80)
    with col2:
        st.title("Multilingual Climate Chatbot")
        st.write("Ask me anything about climate change!")
    
    # Display chat messages
    display_chat_messages()
    
    if st.session_state.language_confirmed:
        query = st.chat_input("Ask Climate Change Bot")
    else:
        st.info("Please select your language and click Confirm to start chatting.")
        query = None
    
    if query:
        # Handle user input and generate response
        st.session_state.chat_history.append({'role': 'user', 'content': query})
        st.session_state.has_asked_question = True
        st.chat_message("user").markdown(query)
        
        typing_message = st.empty()
        with typing_message:
            st.chat_message("assistant").markdown("_Assistant is typing..._")
        
        progress_placeholder = st.empty()
        display_progress(progress_placeholder)
        
        # Process query and get response
        result = asyncio.run(chatbot.process_query(query, st.session_state.selected_language))
        
        typing_message.empty()
        
        if result['success']:
            response_data = {
                'role': 'assistant',
                'content': result['response'],
                'citations': result.get('citations', [])
            }
            st.session_state.chat_history.append(response_data)
            
            # Display the response
            assistant_message = st.chat_message("assistant")
            assistant_message.markdown(result['response'])
            
            if result.get('citations'):
                message_idx = len(st.session_state.chat_history) - 1
                display_source_citations(result['citations'], base_idx=message_idx)
            
            # Force a rerun to update the chat history immediately
            st.rerun()
        else:
            st.error(result['message'])

if __name__ == "__main__":
    main()