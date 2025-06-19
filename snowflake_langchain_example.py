import json
import re
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from snowflake_conn import snowflake_conn

# Load environment variables from .env file
load_dotenv()

def extract_sql_query(text):
    """Extract SQL query from LLM response that may contain explanatory text."""
    # First, try to find SQL in markdown code blocks
    sql_pattern = r'```(?:sql)?\s*(.*?)\s*```'
    matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the first SQL block found
        return matches[0].strip()
    
    # If no code blocks found, try to find SQL keywords and extract from there
    # Look for common SQL keywords at the start of a line
    sql_keywords = r'^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER)\b'
    lines = text.split('\n')
    
    sql_lines = []
    capturing = False
    
    for line in lines:
        if re.match(sql_keywords, line, re.IGNORECASE):
            capturing = True
            sql_lines.append(line)
        elif capturing:
            # Continue capturing until we hit an empty line or non-SQL content
            if line.strip() == '' or line.strip().endswith(':') or line.startswith('Explanation:'):
                break
            sql_lines.append(line)
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    # If nothing found, return the original text (fallback)
    return text.strip()

def initialize_database():
    """Initialize and return the database connection using snowflake_conn class."""
    conn = snowflake_conn()
    return conn

def get_schema(conn):
    """Get database schema information using native Snowflake connection."""
    conn.execute('use database parsed')
    conn.execute("select column_name, data_type from parsed.information_schema.columns where table_name = 'H1B_clean' ")
    schema_data = conn.fetch()
    
    # Format the schema information
    if schema_data:
        schema_str = "Table: parsed.combined.h1b_clean\nColumns:\n"
        for row in schema_data:
            schema_str += f"- {row[0]} ({row[1]})\n"
        return schema_str
    else:
        return "No schema information found for table H1B_clean"

def create_sql_chain(conn):
    """Create the SQL generation chain."""
    # Template for generating SQL queries
    template = """Based on the table schema below, write a Snowflake SQL query that would answer the user's question.
The main table is: parsed.combined.h1b_clean. When use where clause, if the column is text, use like operator. for start_date column, use extract(year from start_date) to get the year.

{schema}

Question: {question}

Please provide only the SQL query without any explanation:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM with Azure OpenAI GPT-4o
    llm = AzureChatOpenAI(
        model="gpt-4o",
        azure_endpoint="https://your-azure-openai-resource.openai.azure.com/",  # Replace with your Azure OpenAI endpoint
        api_version="2024-02-15-preview",  # Use the latest API version
        temperature=0
    )
    
    # Create the SQL generation chain with SQL extraction
    sql_chain = (
        RunnablePassthrough.assign(schema=lambda x: get_schema(conn))
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | extract_sql_query
    )
    
    return sql_chain

def create_full_chain(conn, sql_chain):
    """Create the complete chain including natural language response."""
    # Template for natural language response
    template_response = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
    
    prompt_response = ChatPromptTemplate.from_template(template_response)
    
    # Initialize the LLM with Azure OpenAI GPT-4o
    llm = AzureChatOpenAI(
        model="gpt-4o",
        azure_endpoint="https://your-azure-openai-resource.openai.azure.com/",  # Replace with your Azure OpenAI endpoint
        api_version="2024-02-15-preview",  # Use the latest API version
        temperature=0
    )
    
    def run_query(query):
        """Execute SQL query on the database."""
        conn.execute(query)
        return conn.fetch()
    # Create the full chain
    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda x: get_schema(conn),
            response=lambda vars: run_query(vars["query"]),
        )
        | prompt_response
        | llm
        | StrOutputParser()
    )
    
    return full_chain

def test_sql_generation(sql_chain, question):
    """Test the SQL generation chain with a given question."""
    print("Generated SQL Query:")
    sql_query = sql_chain.invoke({"question": question})
    if sql_query:
        print(sql_query)
    else:
        print("Warning: No SQL query generated or SQL query is empty")
    return sql_query

def test_full_chain(full_chain, sql_chain, conn, question):
    """Test the full chain with natural language response."""
    # First generate and print the SQL query
    print("Generated SQL Query:")
    sql_query = sql_chain.invoke({"question": question})
    if sql_query:
        print(sql_query)
    else:
        print("Warning: No SQL query generated or SQL query is empty")
        return None
    
    # Execute and print the SQL results
    print("\nSQL Execution Results:")
    try:
        conn.execute(sql_query)
        sql_results = conn.fetch()
        if sql_results:
            for row in sql_results:
                print(row)
        else:
            print("No results returned from query")
    except Exception as e:
        print(f"Error executing SQL: {e}")
        sql_results = None
    
    print("\nNatural Language Response:")
    response = full_chain.invoke({"question": question})
    
    # Extract content if it's a response object
    if hasattr(response, 'content'):
        print(response.content)
        return response.content
    else:
        print(response)
        return response

def main():
    """Main function to demonstrate the text-to-SQL functionality."""
    print("Initializing database connection...")
    conn = initialize_database()
    
    print("Creating SQL generation chain...")
    sql_chain = create_sql_chain(conn)
    
    print("Creating full chain with natural language response...")
    full_chain = create_full_chain(conn, sql_chain)
    
    print("\n" + "="*60 + "\n")
    
    # Another example query
    user_question2 = "What's the average salary for carmax engineer in 2024?"
    print(f"Question: {user_question2}")
    test_full_chain(full_chain, sql_chain, conn, user_question2)

if __name__ == "__main__":
    main() 