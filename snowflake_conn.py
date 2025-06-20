from typing import Optional, Any, List, Tuple, Dict
from snowflake.connector import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()

class snowflake_conn(SnowflakeConnection):
    """
    A Snowflake database connection wrapper class that extends SnowflakeConnection.
    
    This class provides a simplified interface for connecting to and interacting with
    a Snowflake database. It automatically configures the connection with predefined
    credentials and provides convenient methods for executing queries and fetching results.
    
    Attributes:
        cur (SnowflakeCursor): The database cursor for executing queries.
        
    Connection Details:
        - User: STANLEYSNOWFLAKE
        - Account: HTQEADI-ZNA89116
        - Database: parsed
        - Schema: combined
        
    Example:
        >>> conn = snowflake_conn()
        >>> conn.execute("SELECT * FROM my_table LIMIT 10")
        >>> conn.fetch()
        >>> conn.close()
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Snowflake connection with predefined configuration.
        
        Creates a connection to Snowflake using hardcoded credentials and database settings.
        Additional keyword arguments can be passed to override default connection parameters.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the parent SnowflakeConnection.
                     These will override any default connection parameters.
                     
        Raises:
            snowflake.connector.errors.Error: If connection to Snowflake fails.
            
        Note:
            The connection uses hardcoded credentials. In production, consider using
            environment variables or secure credential management.
        """
        super().__init__(
            user="STANLEYSNOWFLAKE",
            account="HTQEADI-ZNA89116",
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database="parsed",
            schema="combined",
            **kwargs
        )

        self.cur: SnowflakeCursor = self.cursor()

    def execute(self, query: str) -> None:
        """
        Execute a SQL query on the Snowflake database.
        
        This method executes the provided SQL query and prints the query to stdout
        for debugging and logging purposes. The results remain in the cursor and
        can be retrieved using the fetch() method.
        
        Args:
            query (str): The SQL query string to execute. Can be any valid SQL statement
                        including SELECT, INSERT, UPDATE, DELETE, CREATE, etc.
                        
        Raises:
            snowflake.connector.errors.ProgrammingError: If the SQL query is invalid.
            snowflake.connector.errors.DatabaseError: If there's a database-related error.
            
        Example:
            >>> conn.execute("SELECT COUNT(*) FROM users WHERE active = true")
            >>> conn.execute("CREATE TABLE temp_table AS SELECT * FROM source_table")
        """
        self.cur.execute(query)

    def fetch(self) -> None:
        """
        Fetch and print all results from the last executed query.
        
        This method retrieves all rows from the result set of the most recently
        executed query and prints them to stdout. The results are fetched as a
        list of tuples, where each tuple represents a row.
        
        Note:
            This method prints results directly and doesn't return them. For programmatic
            access to results, consider using self.cur.fetchall() directly.
            
        Raises:
            snowflake.connector.errors.Error: If there's an error fetching results.
            
        Example:
            >>> conn.execute("SELECT name, age FROM users LIMIT 5")
            >>> conn.fetch()  # Prints: [('John', 25), ('Jane', 30), ...]
        """
        results = self.cur.fetchall()
        return results
    
    def query_to_pandas(self, query: str, nrows: int = 10) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        This method executes the provided SQL query and returns the results as a pandas DataFrame.
        Args:
            query (str): The SQL query string to execute.
            nrows (int): The number of rows to fetch from the result set.
        Returns:
            pd.DataFrame: A pandas DataFrame containing the query results.
        """
        self.execute(query)
        result = self.cur.fetchmany(nrows) 
        result2 = [list(i) for i in result]
        cols = [i[0] for i in self.cur.description]
        return pd.DataFrame(result2, columns=cols)

    def close(self) -> None:
        """
        Close the Snowflake database connection.
        
        This method properly closes the database connection and releases any
        associated resources. It's important to call this method when finished
        with the database operations to prevent connection leaks.
        
        Note:
            After calling this method, the connection object should not be used
            for further database operations.
            
        Example:
            >>> conn = snowflake_conn()
            >>> # ... perform database operations ...
            >>> conn.close()  # Always close when done
        """
        super().close()




