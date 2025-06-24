import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from snowflake_conn import snowflake_conn
from llm_tools import get_table_metadata, execute_sql, read_table_descriptions
import yaml

# Load environment variables from .env file
load_dotenv()

def load_agent_config(config_file: str = "agent.yaml") -> Dict[str, Any]:
    """
    Load agent configurations from YAML file.
    
    Args:
        config_file (str): Path to the configuration YAML file
        
    Returns:
        Dict[str, Any]: Agent configurations
    """
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    except yaml.YAMLError:
        raise ValueError(f"Invalid YAML in configuration file {config_file}")

def create_openai_client(use_azure: bool = False):
    """
    Create OpenAI client - either Azure OpenAI or regular OpenAI.
    
    Args:
        use_azure (bool): If True, use Azure OpenAI. If False, use regular OpenAI (default)
        
    Returns:
        OpenAI or AzureOpenAI client instance
    """
    if use_azure:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

class TableSelectionAgent:
    """
    Agent 1: Selects relevant tables and columns based on user question and YAML schema.
    """
    
    def __init__(self, use_azure: bool = False, config_file: str = "agent.yaml"):
        self.client = create_openai_client(use_azure)
        self.config = load_agent_config(config_file)["table_selection_agent"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
    
    def select_tables_and_columns(self, user_question: str) -> str:
        """
        Select relevant tables and columns for answering the user question.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            str: Natural language response with selected tables, columns and reasoning
        """
        # Read the YAML schema directly
        schema_content = self.read_yaml_schema("table_schema.yaml")
        
        # Get system message and user message template from config
        system_message = self.config["system_message"]
        user_message = self.config["user_message_template"].format(
            user_question=user_question,
            schema_content=schema_content
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Single call to generate response with schema already provided
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=self.config["temperature"]
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            # Track cached tokens if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                    self.total_cached_tokens += response.usage.prompt_tokens_details.cached_tokens or 0
            self.total_tokens += response.usage.total_tokens
            self.api_calls += 1
        
        # Return the natural language response
        response_content = response.choices[0].message.content.strip()
        return response_content
    
    def read_yaml_schema(self, file_path: str) -> str:
        """
        Read and return the YAML schema file content.
        
        Args:
            file_path (str): Path to the YAML schema file
            
        Returns:
            str: YAML content as string
        """
        try:
            with open(file_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                return yaml.dump(yaml_content, default_flow_style=False, indent=2)
        except FileNotFoundError:
            return f"Error: YAML schema file {file_path} not found"
        except yaml.YAMLError as e:
            return f"Error parsing YAML file: {str(e)}"
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage statistics for this agent.
        
        Returns:
            Dict[str, int]: Token usage statistics
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls
        }





class SQLGenerationAgent:
    """
    Agent 3: Generates SQL queries based on selected tables and columns.
    """
    
    def __init__(self, use_azure: bool = False, config_file: str = "agent.yaml"):
        self.client = create_openai_client(use_azure)
        self.config = load_agent_config(config_file)["sql_generation_agent"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
    
    def generate_sql(self, user_question: str, selected_tables_columns: str) -> str:
        """
        Generate SQL query based on user question and selected tables/columns.
        
        Args:
            user_question (str): Natural language question from user
            selected_tables_columns (str): Detailed text with selected tables, columns and metadata
            
        Returns:
            str: Generated SQL query
        """
        # Get system message and user message template from config
        system_message = self.config["system_message"]
        user_message = self.config["user_message_template"].format(
            user_question=user_question,
            selected_tables_columns=selected_tables_columns
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Generate SQL query
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=self.config["temperature"]
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            # Track cached tokens if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                    self.total_cached_tokens += response.usage.prompt_tokens_details.cached_tokens or 0
            self.total_tokens += response.usage.total_tokens
            self.api_calls += 1
        
        return response.choices[0].message.content.strip()
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage statistics for this agent.
        
        Returns:
            Dict[str, int]: Token usage statistics
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls
        }


class SQLExecutionAgent:
    """
    Model 2: Executes SQL queries and provides natural language responses.
    Has access to execute_sql tool.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.yaml"):
        self.client = create_openai_client(use_azure)
        self.conn = conn
        self.config = load_agent_config(config_file)["sql_execution_agent"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
    

    
    def generate_response(self, user_question: str, sql_query: str) -> str:
        """
        Execute SQL query and generate natural language response.
        
        Args:
            user_question (str): Original user question
            sql_query (str): SQL query to execute
            
        Returns:
            str: Natural language response
        """
        # Get tools from config
        tools = self.config["tools"]
        
        # Get system message and user message template from config
        system_message = self.config["system_message"]
        user_message = self.config["user_message_template"].format(
            user_question=user_question,
            sql_query=sql_query
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Execute the SQL and get response
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config["temperature"]
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            # Track cached tokens if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                    self.total_cached_tokens += response.usage.prompt_tokens_details.cached_tokens or 0
            self.total_tokens += response.usage.total_tokens
            self.api_calls += 1
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "execute_sql":
                    sql_query = function_args["sql_query"]
                    df, df_string, success = execute_sql(self.conn, sql_query)
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": df_string
                    })
            
            # Continue the conversation
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config["temperature"]
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                # Track cached tokens if available
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                        self.total_cached_tokens += response.usage.prompt_tokens_details.cached_tokens or 0
                self.total_tokens += response.usage.total_tokens
                self.api_calls += 1
        
        return response.choices[0].message.content,df
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage statistics for this agent.
        
        Returns:
            Dict[str, int]: Token usage statistics
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls
        }


class TextToSQLPipeline:
    """
    Complete pipeline that combines all agents for end-to-end text-to-SQL functionality.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.yaml"):
        self.table_selector = TableSelectionAgent(use_azure, config_file)
        self.sql_generator = SQLGenerationAgent(use_azure, config_file)
        self.sql_executor = SQLExecutionAgent(conn, use_azure, config_file)
    
    def process_question(self, user_question: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a natural language question through the complete pipeline.
        
        Args:
            user_question (str): Natural language question from user
            progress_callback (callable): Optional callback to report progress updates
            
        Returns:
            Dict[str, Any]: Results containing SQL query, execution results, natural language response, and thinking process
        """
        print(f"Processing question: {user_question}")
        print("-" * 80)
        
        # Initialize thinking process tracking
        thinking_process = []
        
        # Step 1: Select relevant tables
        if progress_callback:
            progress_callback("🔍 Table Selection Agent - Identifying relevant tables...")
        print("Step 1: Selecting relevant tables...")
        step1_start = {
            "agent": "Table Selection Agent",
            "description": "Analyzing the question to identify relevant tables",
            "input": user_question,
            "status": "processing"
        }
        thinking_process.append(step1_start)
        
        selected_tables_columns = self.table_selector.select_tables_and_columns(user_question)
        
        # Update step 1 with results
        thinking_process[-1].update({
            "output": selected_tables_columns,
            "status": "completed",
            "details": f"Selected tables and columns: {len(selected_tables_columns)} words"
        })
        
        print(f"Selected tables and columns: {selected_tables_columns}")
        
        # Step 2: Generate SQL query
        if progress_callback:
            progress_callback("⚡ SQL Generation Agent - Creating SQL query...")
        print("Step 2: Generating SQL query...")
        step2_start = {
            "agent": "SQL Generation Agent",
            "description": "Generating SQL query based on selected tables and columns",
            "input": {
                "question": user_question,
                "selected_tables_columns": selected_tables_columns
            },
            "status": "processing"
        }
        thinking_process.append(step2_start)
        
        sql_query = self.sql_generator.generate_sql(user_question, selected_tables_columns)
        
        # Update step 2 with results
        thinking_process[-1].update({
            "output": sql_query,
            "status": "completed",
            "details": f"Generated SQL query ({len(sql_query.split())} words)"
        })
        
        print(f"Generated SQL: {sql_query}")
        print()
        
        # Step 3: Execute SQL and generate response
        if progress_callback:
            progress_callback("🚀 SQL Execution Agent - Running query and generating response...")
        print("Step 3: Executing SQL and generating response...")
        step3_start = {
            "agent": "SQL Execution Agent",
            "description": "Executing SQL query and generating natural language response",
            "input": {
                "question": user_question,
                "sql_query": sql_query
            },
            "status": "processing"
        }
        thinking_process.append(step3_start)
        
        natural_response, df = self.sql_executor.generate_response(user_question, sql_query)
        
        # Update step 3 with results
        df_info = f"No data returned" if df is None or df.empty else f"{len(df)} rows, {len(df.columns)} columns"
        thinking_process[-1].update({
            "output": {
                "natural_response": natural_response,
                "dataframe_info": df_info
            },
            "status": "completed",
            "details": f"Executed query successfully. Result: {df_info}"
        })
        
        print(f"Natural Language Response: {natural_response}")
        print()
        
        # Step 4: Collect token usage from all agents
        if progress_callback:
            progress_callback("📊 Collecting token usage statistics...")
        print("Step 4: Token Usage Summary...")
        table_usage = self.table_selector.get_token_usage()
        sql_gen_usage = self.sql_generator.get_token_usage()
        sql_exec_usage = self.sql_executor.get_token_usage()
        
        print(f"Table Selection Agent: {table_usage}")
        print(f"SQL Generation Agent: {sql_gen_usage}")
        print(f"SQL Execution Agent: {sql_exec_usage}")
        
        # Calculate total token usage
        total_input_tokens = (table_usage["input_tokens"] + 
                             sql_gen_usage["input_tokens"] + sql_exec_usage["input_tokens"])
        total_output_tokens = (table_usage["output_tokens"] + 
                              sql_gen_usage["output_tokens"] + sql_exec_usage["output_tokens"])
        total_cached_tokens = (table_usage["cached_tokens"] + 
                              sql_gen_usage["cached_tokens"] + sql_exec_usage["cached_tokens"])
        total_tokens = total_input_tokens + total_output_tokens
        total_api_calls = (table_usage["api_calls"] + 
                          sql_gen_usage["api_calls"] + sql_exec_usage["api_calls"])
        
        print(f"Total Pipeline Usage: Input: {total_input_tokens}, Output: {total_output_tokens}, Cached: {total_cached_tokens}, Total: {total_tokens}, API Calls: {total_api_calls}")
        print()
        
        return {
            "question": user_question,
            "selected_tables_columns": selected_tables_columns,
            "sql_query": sql_query,
            "natural_response": natural_response,
            "dataframe": df,
            "thinking_process": thinking_process,
            "token_usage": {
                "table_selection": table_usage,
                "sql_generation": sql_gen_usage,
                "sql_execution": sql_exec_usage,
                "total": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "cached_tokens": total_cached_tokens,
                    "total_tokens": total_tokens,
                    "api_calls": total_api_calls
                }
            }
        }


def main():
    """
    Main function to demonstrate the two-model text-to-SQL system.
    """
    print("Initializing Snowflake connection...")
    conn = snowflake_conn()
    
    # Set use_azure=True to use Azure OpenAI, or use_azure=False for regular OpenAI (default)
    use_azure = False  # Change this to True to use Azure OpenAI
    provider_name = "Azure OpenAI" if use_azure else "OpenAI"
    print(f"Initializing Text-to-SQL Pipeline with {provider_name}...")
    pipeline = TextToSQLPipeline(conn, use_azure=use_azure)
    
    # Test questions
    test_questions = [
        "Check the pay for a software engineer in Atlanta and compare it with the pay for a software engineer in San Francisco"  
    ]
    
    try:
        for question in test_questions:
            result = pipeline.process_question(question)
            print("=" * 80)
            print()
    finally:
        # Close the connection when done
        print("Closing Snowflake connection...")
        conn.close()


if __name__ == "__main__":
    main() 