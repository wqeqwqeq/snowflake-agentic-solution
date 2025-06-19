from langchain_core.prompts import ChatPromptTemplate
from snowflake_conn import snowflake_conn

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)


def get_schema(db):
    schema = db.get_table_info()
    return schema


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


user_question = 'how many albums are there in the database?'
sql_chain.invoke({"question": user_question})

# 'SELECT COUNT(*) AS TotalAlbums\nFROM Album;'


template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)


def run_query(query):
    return db.run(query)


full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | model
)
user_question = 'how many albums are there in the database?'
full_chain.invoke({"question": user_question})

# 'There are 347 albums in the database.'
