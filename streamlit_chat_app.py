import streamlit as st
import os
from dotenv import load_dotenv
from azure_openai_models import TextToSQLPipeline
from snowflake_conn import snowflake_conn

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Vehicle Sales Chat Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
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
            <div class="message-header">🧑‍💻 You:</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">🤖 Assistant:</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if sql_query:
            with st.expander("📊 View SQL Query", expanded=False):
                st.code(sql_query, language="sql")

def main():
    # Header
    st.title("🚗 Vehicle Sales Chat Assistant")
    st.markdown("Ask questions about vehicle sales data for CMX in natural language!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chat assistant helps you explore vehicle sales data for CMX using natural language queries.
        
        **Features:**
        - 🔍 Natural language to SQL conversion
        - 📊 Automatic query execution
        - 💬 Chat-like interface
        - 📈 Data insights and analysis
        
        **Example Questions:**
        - "What's the average price for a used car?"
        - "Show me top 5 models by sales"
        - "How many sales were in 2024?"
        """)
        
        st.header("🔧 System Status")
        
        # Check environment variables
        api_key_status = "✅" if os.getenv("AZURE_OPENAI_API_KEY") else "❌"
        endpoint_status = "✅" if os.getenv("AZURE_OPENAI_ENDPOINT") else "❌"
        
        st.markdown(f"""
        - Azure OpenAI API Key: {api_key_status}
        - Azure OpenAI Endpoint: {endpoint_status}
        """)
    
    # Initialize connections and pipeline
    if "conn" not in st.session_state:
        with st.spinner("🔗 Connecting to Snowflake..."):
            st.session_state.conn = initialize_connection()
    
    if "pipeline" not in st.session_state and st.session_state.conn:
        with st.spinner("🚀 Initializing AI pipeline..."):
            st.session_state.pipeline = initialize_pipeline(st.session_state.conn)
    
    # Check if initialization was successful
    if not st.session_state.conn:
        st.error("❌ Unable to connect to Snowflake. Please check your connection settings.")
        st.stop()
    
    if not st.session_state.pipeline:
        st.error("❌ Unable to initialize AI pipeline. Please check your Azure OpenAI settings.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your vehicle sales assistant for CMX data. Ask me anything about vehicle sales, prices, models, or trends. What would you like to know?",
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
    user_input = st.chat_input("Ask a question about vehicle sales data...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sql_query": None
        })
        
        # Process the question
        with st.spinner("🤔 Thinking and querying the database..."):
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
                error_message = f"I encountered an error while processing your question: {str(e)}"
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
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chat cleared! What would you like to know about vehicle sales data for CMX?",
                "sql_query": None
            })
            st.rerun()
    
    # Sample questions
    st.markdown("---")
    st.subheader("💡 Try these example questions:")
    
    example_questions = [
        "What's the average price for vehicles sold in 2024?",
        "Show me the top 5 vehicle models by sales volume",
        "How many vehicles were sold this year?",
        "What are the price trends for sedans?",
        "Which vehicle models have the highest average selling price?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                # Add the example question as user input
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "sql_query": None
                })
                
                # Process the example question
                with st.spinner("🤔 Processing example question..."):
                    try:
                        result = st.session_state.pipeline.process_question(question)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["natural_response"],
                            "sql_query": result["sql_query"]
                        })
                    except Exception as e:
                        error_message = f"I encountered an error: {str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sql_query": None
                        })
                
                st.rerun()

if __name__ == "__main__":
    main() 