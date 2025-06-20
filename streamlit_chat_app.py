import streamlit as st
import os
import json
from dotenv import load_dotenv
from azure_openai_models import TextToSQLPipeline
from snowflake_conn import snowflake_conn

# Load environment variables
load_dotenv()

# Load UI configuration
@st.cache_data
def load_ui_config():
    """Load UI configuration from JSON file."""
    try:
        with open('H1BUI.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("UI configuration file (ui.json) not found!")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error parsing UI configuration: {str(e)}")
        st.stop()

# Load UI configuration
ui_config = load_ui_config()

# Set page configuration using config
st.set_page_config(
    page_title=ui_config["page_config"]["page_title"],
    page_icon=ui_config["page_config"]["page_icon"],
    layout=ui_config["page_config"]["layout"],
    initial_sidebar_state=ui_config["page_config"]["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sql-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_connection():
    """Initialize Snowflake connection with caching."""
    try:
        conn = snowflake_conn()
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

@st.cache_resource
def initialize_pipeline(_conn):
    """Initialize the text-to-SQL pipeline with caching."""
    if _conn is None:
        return None
    try:
        pipeline = TextToSQLPipeline(_conn)
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

def display_chat_message(role, message, sql_query=None):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">{ui_config["chat"]["user_label"]}</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">{ui_config["chat"]["assistant_label"]}</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if sql_query:
            with st.expander(ui_config["chat"]["sql_expander_title"], expanded=False):
                st.code(sql_query, language="sql")

def main():
    # Header
    st.title(ui_config["main"]["title"])
    st.markdown(ui_config["main"]["description"])
    
    # Sidebar with information
    with st.sidebar:
        st.header(ui_config["sidebar"]["about"]["header"])
        
        # About section
        about_content = ui_config["sidebar"]["about"]["description"] + "\n\n**Features:**\n"
        for feature in ui_config["sidebar"]["about"]["features"]:
            about_content += f"- {feature}\n"
        
        about_content += "\n**Example Questions:**\n"
        for question in ui_config["sidebar"]["about"]["example_questions"]:
            about_content += f"- \"{question}\"\n"
        
        st.markdown(about_content)
        
        st.header(ui_config["sidebar"]["system_status"]["header"])
        
        # Check environment variables
        api_key_status = ui_config["icons"]["success"] if os.getenv("AZURE_OPENAI_API_KEY") else ui_config["icons"]["error"]
        endpoint_status = ui_config["icons"]["success"] if os.getenv("AZURE_OPENAI_ENDPOINT") else ui_config["icons"]["error"]
        
        st.markdown(f"""
        - {ui_config["sidebar"]["system_status"]["api_key_label"]}: {api_key_status}
        - {ui_config["sidebar"]["system_status"]["endpoint_label"]}: {endpoint_status}
        """)
    
    # Initialize connections and pipeline
    if "conn" not in st.session_state:
        with st.spinner(ui_config["status_messages"]["connecting_snowflake"]):
            st.session_state.conn = initialize_connection()
    
    if "pipeline" not in st.session_state and st.session_state.conn:
        with st.spinner(ui_config["status_messages"]["initializing_pipeline"]):
            st.session_state.pipeline = initialize_pipeline(st.session_state.conn)
    
    # Check if initialization was successful
    if not st.session_state.conn:
        st.error(ui_config["status_messages"]["snowflake_error"])
        st.stop()
    
    if not st.session_state.pipeline:
        st.error(ui_config["status_messages"]["pipeline_error"])
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": ui_config["main"]["welcome_message"],
            "sql_query": None
        })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("sql_query")
            )
    
    # Chat input
    user_input = st.chat_input(ui_config["chat"]["input_placeholder"])
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sql_query": None
        })
        
        # Process the question
        with st.spinner(ui_config["status_messages"]["processing_question"]):
            try:
                # Use the pipeline to process the question
                result = st.session_state.pipeline.process_question(user_input)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["natural_response"],
                    "sql_query": result["sql_query"]
                })
                
            except Exception as e:
                error_message = f"{ui_config['status_messages']['error_prefix']}{str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sql_query": None
                })
        
        # Rerun to display the new messages
        st.rerun()
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(ui_config["chat"]["clear_button_text"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": ui_config["chat"]["clear_confirmation_message"],
                "sql_query": None
            })
            st.rerun()
    
    # Sample questions
    st.markdown("---")
    st.subheader(ui_config["sample_questions"]["header"])
    
    example_questions = ui_config["sample_questions"]["questions"]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"{ui_config['icons']['chat_prefix']}{question}", key=f"example_{i}", use_container_width=True):
                # Add the example question as user input
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "sql_query": None
                })
                
                # Process the example question
                with st.spinner(ui_config["status_messages"]["processing_example"]):
                    try:
                        result = st.session_state.pipeline.process_question(question)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["natural_response"],
                            "sql_query": result["sql_query"]
                        })
                    except Exception as e:
                        error_message = f"{ui_config['status_messages']['example_error_prefix']}{str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sql_query": None
                        })
                
                st.rerun()

if __name__ == "__main__":
    main() 