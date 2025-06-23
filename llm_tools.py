import pandas as pd 

def get_table_metadata(conn, db_schema_tbl: str) -> list:
    """
    Tool function to get metadata of a Snowflake table from the metadata table.
    
    Args:
        conn: Database connection object
        db_schema_tbl (str): Full table reference in format database.schema.table
        
    Returns:
        list: List of dictionaries containing column metadata information
    """
    try:
        import os
        metadata_table = os.getenv('METADATA_TABLE')
        if not metadata_table:
            return "METADATA_TABLE environment variable not set"
        
        # Query the metadata table for this specific table
        conn.execute(f"""
            SELECT column_name, data_type, comment, distinct_count, distinct_values
            FROM {metadata_table}
            WHERE table_name = '{db_schema_tbl}'
            ORDER BY column_name
        """)
        metadata_results = conn.fetch()
        
        if metadata_results:
            columns_list = []
            for row in metadata_results:
                column_name, data_type, comment, distinct_count, distinct_values = row
                column_info = {
                    "table_name": db_schema_tbl,
                    "column_name": column_name,
                    "data_type": data_type,
                    "description": comment,
                    "distinct_count": distinct_count,
                    "sample_values": distinct_values
                }
                columns_list.append(column_info)
            return columns_list
        else:
            return []
    except Exception as e:
        return f"Error retrieving metadata: {str(e)}"


def execute_sql(conn, sql_query: str, nrows: int = 10) -> tuple:
    """
    Tool function to execute SQL query and fetch results.
    
    Args:
        conn: Database connection object
        sql_query (str): SQL query to execute
        nrows (int): Number of rows to fetch (default: 10)
        
    Returns:
        tuple: (pandas.DataFrame, str) - DataFrame and string representation of results
    """
    try:
        # Use the query_to_pandas method from the connection object
        df = conn.query_to_pandas(sql_query, nrows=nrows)
        
        if not df.empty:
            # Convert DataFrame to string representation
            df_string = df.to_string(index=False)
            return df, df_string, True
        else:
            # Return empty DataFrame and message
            return df, "No results returned from the query", False
            
    except Exception as e:
        # Return error as single-row DataFrame
        error_df = pd.DataFrame({"Error": [str(e)]})
        error_string = f"Error executing SQL: {str(e)}"
        return error_df, error_string, False


def read_table_descriptions(file_path: str) -> str:
    """
    Tool function to read table descriptions from a markdown file.
    
    Args:
        file_path (str): Path to the table description file
        
    Returns:
        str: Content of the table description file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip():
            return content
        else:
            return "Table description file is empty"
            
    except FileNotFoundError:
        return f"Table description file not found: {file_path}"
    except Exception as e:
        return f"Error reading table description file: {str(e)}" 