import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from snowflake_conn import snowflake_conn

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

class TextToSQLAgent:
    """
    Model 1: Converts natural language to SQL using OpenAI with function calling.
    Has access to get_table_metadata tool.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.json"):
        self.client = create_openai_client(use_azure)
        self.conn = conn
        self.config = load_agent_config(config_file)["text_to_sql_agent"]
        
    def get_table_metadata(self, table_name: str = "H1B_clean") -> str:
        """
        Tool function to get metadata of a Snowflake table.
        
        Args:
            table_name (str): Name of the table to get metadata for
            
        Returns:
            str: Formatted table schema information
        """
        try:
            self.conn.execute('USE DATABASE parsed')
            self.conn.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM parsed.information_schema.columns 
                WHERE table_name = '{table_name.upper()}'
                ORDER BY ordinal_position
            """)
            schema_data = self.conn.fetch()
            
            if schema_data:
                schema_str = f"Table: parsed.combined.{table_name.lower()}\nColumns:\n"
                for row in schema_data:
                    nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                    default = f", DEFAULT: {row[3]}" if row[3] else ""
                    schema_str += f"- {row[0]} ({row[1]}, {nullable}{default})\n"
                return schema_str
            else:
                return f"No schema information found for table {table_name}"
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"
    
    def generate_sql(self, user_question: str) -> str:
        """
        Convert natural language question to SQL using OpenAI with function calling.
        
        Args:
            user_question (str): Natural language question from user
            
        Returns:
            str: Generated SQL query
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
        
        # First call to get table metadata
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
                    table_name = function_args.get("table_name", "H1B_clean")
                    function_result = self.get_table_metadata(table_name)
                    
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
    
    def execute_sql(self, sql_query: str) -> str:
        """
        Tool function to execute SQL query and fetch results.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            str: Formatted query results
        """
        try:
            self.conn.execute(sql_query)
            results = self.conn.fetch()
            
            if results:
                # Format results as a readable string
                if len(results) == 1 and len(results[0]) == 1:
                    # Single value result
                    return str(results[0][0])
                else:
                    # Multiple rows/columns
                    formatted_results = []
                    for i, row in enumerate(results):
                        if i < 10:  # Limit to first 10 rows for readability
                            formatted_results.append(str(row))
                        else:
                            formatted_results.append(f"... and {len(results) - 10} more rows")
                            break
                    return "\n".join(formatted_results)
            else:
                return "No results returned from the query"
                
        except Exception as e:
            return f"Error executing SQL: {str(e)}"
    
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
                    function_result = self.execute_sql(sql_query)
                    
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
        
        return response.choices[0].message.content


class TextToSQLPipeline:
    """
    Complete pipeline that combines both models for end-to-end text-to-SQL functionality.
    """
    
    def __init__(self, conn, use_azure: bool = False, config_file: str = "agent.json"):
        self.sql_generator = TextToSQLAgent(conn, use_azure, config_file)
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
        print("-" * 60)
        
        # Step 1: Generate SQL using Model 1
        print("Step 1: Generating SQL query...")
        sql_query = self.sql_generator.generate_sql(user_question)
        print(f"Generated SQL: {sql_query}")
        print()
        
        # Step 2: Execute SQL and generate response using Model 2
        print("Step 2: Executing SQL and generating response...")
        natural_response = self.sql_executor.generate_response(user_question, sql_query)
        print(f"Natural Language Response: {natural_response}")
        print()
        
        return {
            "question": user_question,
            "sql_query": sql_query,
            "natural_response": natural_response
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