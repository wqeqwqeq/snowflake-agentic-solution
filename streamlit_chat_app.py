import streamlit as st
import os
import json
from dotenv import load_dotenv
from azure_openai_models import TextToSQLPipeline
from snowflake_conn import snowflake_conn
import time

# Load environment variables
load_dotenv()

class ProgressSpinner:
    """Custom progress spinner that can update text in real-time."""
    
    def __init__(self, initial_text="Processing..."):
        self.placeholder = None
        self.current_text = initial_text
    
    def __enter__(self):
        self.placeholder = st.empty()
        self.update_text(self.current_text)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.placeholder:
            self.placeholder.empty()
    
    def update_text(self, text):
        """Update the spinner text."""
        self.current_text = text
        if self.placeholder:
            with self.placeholder:
                st.info(f"⏳ {text}")

def clean_sql_query(sql_query: str) -> str:
    """
    Clean SQL query by removing markdown code block markers.
    
    Args:
        sql_query (str): SQL query that may contain markdown formatting
        
    Returns:
        str: Clean SQL query without markdown markers
    """
    if not sql_query:
        return sql_query
        
    clean_sql = sql_query.strip()
    
    # Remove opening markdown markers
    if clean_sql.startswith("```sql"):
        clean_sql = clean_sql[6:]
    elif clean_sql.startswith("```"):
        clean_sql = clean_sql[3:]
    
    # Remove closing markdown markers
    if clean_sql.endswith("```"):
        clean_sql = clean_sql[:-3]
    
    return clean_sql.strip()

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
    .thinking-step {
        background-color: #f8f9fa;
        border-left: 3px solid #007acc;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 3px;
    }
    .thinking-step-completed {
        border-left-color: #28a745;
        background-color: #f0fff4;
    }
    .thinking-step-processing {
        border-left-color: #ffc107;
        background-color: #fffbf0;
    }
    .agent-name {
        font-weight: 600;
        color: #2c3e50;
    }
    .step-details {
        font-size: 0.9em;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    .token-summary {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .token-metric {
        text-align: center;
        padding: 0.3rem;
        background-color: #f8f9fa;
        border-radius: 4px;
        margin: 0.2rem;
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

def display_thinking_process(thinking_process, token_usage=None, msg_index=None):
    """Display the thinking process with expandable sections and token usage."""
    if not thinking_process:
        return
    
    with st.expander("🧠 AI Thinking Process", expanded=False):
        # High-level view toggle
        if msg_index is None:
            msg_index = len(st.session_state.messages) - 1
        detail_key = f"detail_mode_{msg_index}"
        
        if detail_key not in st.session_state:
            st.session_state[detail_key] = False
        
        # Toggle button for detail level
        col1, col2 = st.columns([0.8, 0.2])
        with col2:
            if st.button("📊 Toggle Detail", key=f"toggle_detail_{msg_index}", help="Switch between high-level and detailed view"):
                st.session_state[detail_key] = not st.session_state[detail_key]
                st.rerun()
        
        with col1:
            if st.session_state[detail_key]:
                st.markdown("**Detailed View** - Full agent inputs and outputs")
            else:
                st.markdown("**High-Level View** - Agent progress summary")
        
        # Map agent names to token usage keys
        agent_token_map = {
            "Table Selection Agent": "table_selection",
            "Column Selection Agent": "column_selection", 
            "SQL Generation Agent": "sql_generation",
            "SQL Execution Agent": "sql_execution"
        }
        
        # Display each step
        for i, step in enumerate(thinking_process):
            step_number = i + 1
            agent_name = step.get("agent", f"Step {step_number}")
            status = step.get("status", "unknown")
            description = step.get("description", "Processing...")
            
            # Get token usage for this agent
            agent_tokens = None
            if token_usage and agent_name in agent_token_map:
                agent_key = agent_token_map[agent_name]
                if agent_key in token_usage:
                    agent_tokens = token_usage[agent_key]
            
            # Status emoji
            status_emoji = "✅" if status == "completed" else "⏳" if status == "processing" else "❓"
            
            # High-level view
            if not st.session_state[detail_key]:
                # Simple progress view with token info
                status_class = "thinking-step-completed" if status == "completed" else "thinking-step-processing"
                token_info = ""
                if agent_tokens and status == "completed":
                    token_info = f'<div class="step-details">🔢 Tokens: {agent_tokens.get("total_tokens", 0)} total ({agent_tokens.get("input_tokens", 0)} in + {agent_tokens.get("output_tokens", 0)} out)</div>'
                
                st.markdown(f"""
                <div class="thinking-step {status_class}">
                    <div class="agent-name">{status_emoji} Step {step_number}: {agent_name}</div>
                    <div class="step-details">{description}</div>
                    {f'<div class="step-details">✨ {step["details"]}</div>' if status == "completed" and "details" in step else ''}
                    {token_info}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed view
            else:
                with st.expander(f"{status_emoji} Step {step_number}: {agent_name}", expanded=True):
                    st.markdown(f"**Description:** {description}")
                    st.markdown(f"**Status:** {status.title()}")
                    
                    # Token usage info for completed steps
                    if agent_tokens and status == "completed":
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Input Tokens", agent_tokens.get("input_tokens", 0))
                        with col2:
                            st.metric("Output Tokens", agent_tokens.get("output_tokens", 0))
                        with col3:
                            st.metric("Total Tokens", agent_tokens.get("total_tokens", 0))
                        with col4:
                            st.metric("API Calls", agent_tokens.get("api_calls", 0))
                    
                    # Input section
                    if "input" in step:
                        st.markdown("**Input:**")
                        input_data = step["input"]
                        if isinstance(input_data, str):
                            st.code(input_data, language="text")
                        else:
                            st.json(input_data)
                    
                    # Output section
                    if "output" in step and status == "completed":
                        st.markdown("**Output:**")
                        output_data = step["output"]
                        if isinstance(output_data, str):
                            # For SQL queries, use SQL syntax highlighting
                            if agent_name == "SQL Generation Agent":
                                st.code(output_data, language="sql")
                            else:
                                st.code(output_data, language="text")
                        elif isinstance(output_data, list):
                            st.json(output_data)
                        elif isinstance(output_data, dict):
                            # Special handling for different output types
                            if "natural_response" in output_data:
                                st.markdown("*Natural Response:*")
                                st.markdown(output_data["natural_response"])
                                if "dataframe_info" in output_data:
                                    st.markdown(f"*Data Result:* {output_data['dataframe_info']}")
                            else:
                                st.json(output_data)
                        else:
                            st.write(output_data)
                    
                    # Details section
                    if "details" in step:
                        st.markdown(f"**Summary:** {step['details']}")
        
        # Total token usage summary
        if token_usage and "total" in token_usage:
            st.markdown("---")
            st.markdown("### 📊 Total Token Usage Summary")
            
            total_tokens = token_usage["total"]
            
            if not st.session_state[detail_key]:
                # High-level summary
                st.markdown(f"""
                <div class="thinking-step thinking-step-completed">
                    <div class="agent-name">🔢 Pipeline Total Usage</div>
                    <div class="step-details">Total: {total_tokens.get('total_tokens', 0)} tokens | API Calls: {total_tokens.get('api_calls', 0)}</div>
                    <div class="step-details">Input: {total_tokens.get('input_tokens', 0)} tokens | Output: {total_tokens.get('output_tokens', 0)} tokens</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Detailed summary with metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Input", f"{total_tokens.get('input_tokens', 0):,}")
                with col2:
                    st.metric("Total Output", f"{total_tokens.get('output_tokens', 0):,}")
                with col3:
                    st.metric("Grand Total", f"{total_tokens.get('total_tokens', 0):,}")
                with col4:
                    st.metric("Total API Calls", total_tokens.get('api_calls', 0))
                
                # Breakdown by agent
                st.markdown("**Breakdown by Agent:**")
                for agent_display_name, agent_key in agent_token_map.items():
                    if agent_key in token_usage:
                        agent_data = token_usage[agent_key]
                        st.markdown(f"- **{agent_display_name}**: {agent_data.get('total_tokens', 0)} tokens ({agent_data.get('api_calls', 0)} calls)")

def display_chat_message(role, message, sql_query=None, dataframe=None, thinking_process=None, token_usage=None, msg_index=None):
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
        
        # Display thinking process if available
        if thinking_process:
            display_thinking_process(thinking_process, token_usage, msg_index)
        
        if sql_query or (dataframe is not None and not dataframe.empty):
            col1, col2 = st.columns(2)
            
            with col1:
                if sql_query:
                    with st.expander(ui_config["chat"]["sql_expander_title"], expanded=False):
                        # Create a unique key for this message's edit state
                        if msg_index is None:
                            msg_index = len(st.session_state.messages) - 1
                        edit_key = f"edit_sql_{msg_index}"
                        
                        # Initialize edit state if not exists
                        if edit_key not in st.session_state:
                            st.session_state[edit_key] = False
                        
                        # Create columns for SQL display and edit button
                        sql_col1, sql_col2 = st.columns([0.85, 0.15])
                        
                        with sql_col2:
                            # Pencil icon button to toggle edit mode
                            if st.button("✏️", key=f"edit_btn_{msg_index}", help="Edit SQL Query"):
                                st.session_state[edit_key] = not st.session_state[edit_key]
                                st.rerun()
                        
                        with sql_col1:
                            if st.session_state[edit_key]:
                                # Edit mode: show text area with clean SQL
                                clean_sql = clean_sql_query(sql_query)
                                
                                edited_sql = st.text_area(
                                    "Edit SQL Query:", 
                                    value=clean_sql, 
                                    height=200,
                                    key=f"sql_editor_{msg_index}"
                                )
                                
                                # Save and Execute, and Cancel buttons
                                btn_col1, btn_col3, btn_col2 = st.columns([1.5, 1, 1])
                                with btn_col1:
                                    if st.button("💾 Save and Execute", key=f"save_sql_{msg_index}", use_container_width=True):
                                        try:
                                            # Clean the edited SQL query before execution
                                            clean_edited_sql = clean_sql_query(edited_sql)
                                            
                                            # Execute the cleaned SQL query
                                            with st.spinner("Executing query..."):
                                                new_dataframe = st.session_state.conn.query_to_pandas(clean_edited_sql)
                                            
                                            # Update the SQL query and dataframe in the message
                                            st.session_state.messages[msg_index]["sql_query"] = clean_edited_sql
                                            st.session_state.messages[msg_index]["dataframe"] = new_dataframe
                                            st.session_state[edit_key] = False
                                            
                                            # Show success message
                                            st.success("Query executed successfully!")
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"Error executing query: {str(e)}")
                                            # Still save the cleaned query even if execution fails
                                            clean_edited_sql = clean_sql_query(edited_sql)
                                            st.session_state.messages[msg_index]["sql_query"] = clean_edited_sql
                                            st.session_state[edit_key] = False
                                
                                with btn_col2:
                                    if st.button("❌ Cancel", key=f"cancel_sql_{msg_index}", use_container_width=True):
                                        st.session_state[edit_key] = False
                                        st.rerun()
                            else:
                                # View mode: show code
                                st.code(sql_query, language="sql")
            
            with col2:
                if dataframe is not None and not dataframe.empty:
                    with st.expander(ui_config["chat"]["sql_result_expander_title"], expanded=False):
                        st.dataframe(dataframe)

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
            "sql_query": None,
            "dataframe": None,
            "thinking_process": None,
            "token_usage": None
        })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("sql_query"),
                message.get("dataframe"),
                message.get("thinking_process"),
                message.get("token_usage"),
                msg_index=i
            )
    
    # Chat input
    user_input = st.chat_input(ui_config["chat"]["input_placeholder"])
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sql_query": None,
            "dataframe": None,
            "thinking_process": None,
            "token_usage": None
        })
        
        # Process the question
        with ProgressSpinner("🤔 Initializing AI agents...") as spinner:
            try:
                # Define progress callback
                def update_progress(message):
                    spinner.update_text(message)
                    time.sleep(0.1)  # Small delay to make progress visible
                
                # Use the pipeline to process the question
                result = st.session_state.pipeline.process_question(user_input, progress_callback=update_progress)
                
                # Final step
                spinner.update_text("✅ Processing complete!")
                time.sleep(0.5)  # Brief pause to show completion
                
                # Clean the SQL query from the pipeline result
                clean_sql = clean_sql_query(result["sql_query"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["natural_response"],
                    "sql_query": clean_sql,
                    "dataframe": result.get("dataframe"),
                    "thinking_process": result.get("thinking_process"),
                    "token_usage": result.get("token_usage")
                })
                
            except Exception as e:
                spinner.update_text("❌ Error occurred during processing")
                error_message = f"{ui_config['status_messages']['error_prefix']}{str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sql_query": None,
                    "dataframe": None,
                    "thinking_process": None,
                    "token_usage": None
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
                "sql_query": None,
                "dataframe": None,
                "thinking_process": None,
                "token_usage": None
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
                    "sql_query": None,
                    "dataframe": None,
                    "thinking_process": None,
                    "token_usage": None
                })
                
                # Process the example question
                with ProgressSpinner("🤔 Initializing AI agents...") as spinner:
                    try:
                        # Define progress callback
                        def update_progress(message):
                            spinner.update_text(message)
                            time.sleep(0.1)  # Small delay to make progress visible
                        
                        result = st.session_state.pipeline.process_question(question, progress_callback=update_progress)
                        
                        # Final step
                        spinner.update_text("✅ Processing complete!")
                        time.sleep(0.5)  # Brief pause to show completion
                        
                        # Clean the SQL query from the pipeline result
                        clean_sql = clean_sql_query(result["sql_query"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["natural_response"],
                            "sql_query": clean_sql,
                            "dataframe": result.get("dataframe"),
                            "thinking_process": result.get("thinking_process"),
                            "token_usage": result.get("token_usage")
                        })
                    except Exception as e:
                        spinner.update_text("❌ Error occurred during processing")
                        error_message = f"{ui_config['status_messages']['example_error_prefix']}{str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sql_query": None,
                            "dataframe": None,
                            "thinking_process": None,
                            "token_usage": None
                        })
                
                st.rerun()

if __name__ == "__main__":
    main() 