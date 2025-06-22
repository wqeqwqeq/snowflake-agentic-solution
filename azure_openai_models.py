import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from snowflake_conn import snowflake_conn
from llm_tools import get_table_metadata, execute_sql, read_table_descriptions

# Load environment variables from .env file
load_dotenv()

def load_agent_config(config_file: str = "agent.json") -> Dict[str, Any]:
    """
    Load agent configurations from JSON file.
    
    Args:
        config_file (str): Path to the configuration JSON file
        
    Returns:
        Dict[str, Any]: Agent configurations
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file {config_file}")

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
    Agent 1: Selects relevant tables based on user question and table descriptions.
    """
    
    def __init__(self, use_azure: bool = False, config_file: str = "agent.json"):
        self.client = create_openai_client(use_azure)
        self.config = load_agent_config(config_file)["table_selection_agent"]
    
    def select_tables(self, user_question: str) -> List[str]:
        """
        Select relevant tables for answering the user question.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            List[str]: List of selected table names
        """
        # Get tools from config
        tools = self.config["tools"]
        
        # Get system message and user message template from config
        system_message = self.config["system_message"]
        user_message = self.config["user_message_template"].format(user_question=user_question)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # First call to read table descriptions
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config["temperature"]
        )
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "read_table_descriptions":
                    file_path = function_args.get("file_path", "table_description.md")
                    function_result = read_table_descriptions(file_path)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result
                    })
            
            # Continue the conversation
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config["temperature"]
            )
        
        # Parse the response to extract table list
        response_content = response.choices[0].message.content.strip()
        try:
            # Try to parse as JSON list
            tables = json.loads(response_content)
            if isinstance(tables, list):
                return tables
            else:
                # If not a list, wrap in list
                return [tables] if tables else []
        except json.JSONDecodeError:
            # If not valid JSON, try to extract table names manually
            print(f"Could not parse table selection response as JSON: {response_content}")
            return ["parsed.combined.h1b_clean"]  # Fallback to default table


class ColumnSelectionAgent:
    """
    Agent 2: Selects relevant columns from selected tables based on user question.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.json"):
        self.client = create_openai_client(use_azure)
        self.conn = conn
        self.config = load_agent_config(config_file)["column_selection_agent"]
    
    def select_columns(self, user_question: str, selected_tables: List[str]) -> Dict[str, List[str]]:
        """
        Select relevant columns for answering the user question.
        
        Args:
            user_question (str): Natural language question from user
            selected_tables (List[str]): List of selected table names
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping table names to selected columns
        """
        # Get tools from config
        tools = self.config["tools"]
        
        # Get system message and user message template from config
        system_message = self.config["system_message"]
        user_message = self.config["user_message_template"].format(
            user_question=user_question,
            selected_tables=selected_tables
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Call to get table metadata for selected tables
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config["temperature"]
        )
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_table_metadata":
                    db_schema_tbl = function_args.get("db_schema_tbl")
                    function_result = get_table_metadata(self.conn, db_schema_tbl)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result
                    })
            
            # Continue the conversation
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config["temperature"]
            )
        
        # Parse the response to extract column mapping
        response_content = response.choices[0].message.content.strip()
        try:
            # Try to parse as JSON dictionary
            columns_mapping = json.loads(response_content)
            if isinstance(columns_mapping, dict):
                return columns_mapping
            else:
                print(f"Unexpected column selection response format: {response_content}")
                return {table: ["*"] for table in selected_tables}
        except json.JSONDecodeError:
            print(f"Could not parse column selection response as JSON: {response_content}")
            return {table: ["*"] for table in selected_tables}


class SQLGenerationAgent:
    """
    Agent 3: Generates SQL queries based on selected tables and columns.
    """
    
    def __init__(self, use_azure: bool = False, config_file: str = "agent.json"):
        self.client = create_openai_client(use_azure)
        self.config = load_agent_config(config_file)["sql_generation_agent"]
    
    def generate_sql(self, user_question: str, selected_tables_columns: Dict[str, List[str]]) -> str:
        """
        Generate SQL query based on user question and selected tables/columns.
        
        Args:
            user_question (str): Natural language question from user
            selected_tables_columns (Dict[str, List[str]]): Selected tables and their columns
            
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
        
        return response.choices[0].message.content.strip()


class SQLExecutionAgent:
    """
    Model 2: Executes SQL queries and provides natural language responses.
    Has access to execute_sql tool.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.json"):
        self.client = create_openai_client(use_azure)
        self.conn = conn
        self.config = load_agent_config(config_file)["sql_execution_agent"]
    

    
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
        
        # Handle tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "execute_sql":
                    sql_query = function_args["sql_query"]
                    df, df_string = execute_sql(self.conn, sql_query)
                    
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
        
        return response.choices[0].message.content,df


class TextToSQLPipeline:
    """
    Complete pipeline that combines all agents for end-to-end text-to-SQL functionality.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.json"):
        self.table_selector = TableSelectionAgent(use_azure, config_file)
        self.column_selector = ColumnSelectionAgent(conn, use_azure, config_file)
        self.sql_generator = SQLGenerationAgent(use_azure, config_file)
        self.sql_executor = SQLExecutionAgent(conn, use_azure, config_file)
    
    def process_question(self, user_question: str) -> Dict[str, Any]:
        """
        Process a natural language question through the complete pipeline.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            Dict[str, Any]: Results containing SQL query, execution results, and natural language response
        """
        print(f"Processing question: {user_question}")
        print("-" * 80)
        
        # Step 1: Select relevant tables
        print("Step 1: Selecting relevant tables...")
        selected_tables = self.table_selector.select_tables(user_question)
        print(f"Selected tables: {selected_tables}")
        print()
        
        # Step 2: Select relevant columns from selected tables
        print("Step 2: Selecting relevant columns...")
        selected_tables_columns = self.column_selector.select_columns(user_question, selected_tables)
        print(f"Selected tables and columns: {selected_tables_columns}")
        print()
        
        # Step 3: Generate SQL query
        print("Step 3: Generating SQL query...")
        sql_query = self.sql_generator.generate_sql(user_question, selected_tables_columns)
        print(f"Generated SQL: {sql_query}")
        print()
        
        # Step 4: Execute SQL and generate response
        print("Step 4: Executing SQL and generating response...")
        natural_response, df = self.sql_executor.generate_response(user_question, sql_query)
        print(f"Natural Language Response: {natural_response}")
        print()
        
        return {
            "question": user_question,
            "selected_tables": selected_tables,
            "selected_tables_columns": selected_tables_columns,
            "sql_query": sql_query,
            "natural_response": natural_response,
            "dataframe": df
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
        "What's the average salary for carmax engineer in 2024?",
        "How many H1B applications were filed for software engineer positions?",
        "Show me the top 5 companies by number of H1B applications"
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