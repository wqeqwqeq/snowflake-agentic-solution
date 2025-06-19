# 🤖 H1B Data Chat Assistant

An intelligent chat interface for exploring H1B visa data using natural language queries powered by Azure OpenAI and Snowflake.

## 🌟 Features

- **Natural Language Queries**: Ask questions in plain English about H1B data
- **AI-Powered SQL Generation**: Automatically converts your questions to SQL
- **Real-time Data Access**: Connects directly to Snowflake database
- **Interactive Chat Interface**: Modern chat UI with message history
- **SQL Query Transparency**: View the generated SQL queries
- **Example Questions**: Pre-built example queries to get started

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com/
```

### 3. Configure Snowflake Connection

Update the `snowflake_conn.py` file with your Snowflake credentials:

```python
# Update the __init__ method with your actual credentials
def __init__(self, **kwargs):
    super().__init__(
        user='YOUR_SNOWFLAKE_USER',
        password='YOUR_SNOWFLAKE_PASSWORD',
        account='YOUR_SNOWFLAKE_ACCOUNT',
        database='parsed',
        schema='combined',
        **kwargs
    )
```

### 4. Run the Streamlit App

```bash
streamlit run streamlit_chat_app.py
```

The app will open in your browser at `http://localhost:8501`

## 💬 Example Questions

Try asking questions like:

- "What's the average salary for software engineers in 2024?"
- "Show me the top 5 companies by number of H1B applications"
- "How many H1B applications were filed for data scientist positions?"
- "What are the salary trends for engineers at Google?"
- "Which states have the highest H1B application counts?"

## 🏗️ Architecture

The application consists of two main AI models:

### Model 1: TextToSQLAgent
- Converts natural language to SQL queries
- Uses `get_table_metadata` tool to understand table structure
- Generates optimized Snowflake SQL

### Model 2: SQLExecutionAgent
- Executes SQL queries on Snowflake
- Uses `execute_sql` tool to fetch results
- Converts results back to natural language

### TextToSQLPipeline
- Orchestrates both models for end-to-end processing
- Handles the complete flow from question to answer

## 🔧 System Requirements

- Python 3.8+
- Azure OpenAI access with GPT-4o model
- Snowflake database access
- Internet connection for API calls

## 📊 Data Source

The application queries the `parsed.combined.h1b_clean` table in Snowflake, which contains H1B visa application data including:

- Employer information
- Job titles and descriptions
- Salary information
- Application dates
- Approval status
- Geographic data

## 🛠️ Customization

### Adding New Tables
To query additional tables, update the `get_table_metadata` function in `azure_openai_models.py`.

### Modifying UI
The Streamlit interface can be customized by editing `streamlit_chat_app.py`. The app uses custom CSS for styling.

### Extending Functionality
Add new tools or modify existing ones in the respective agent classes.

## 🔒 Security Notes

- Store sensitive credentials in environment variables
- Use Snowflake's security features for database access
- Consider implementing rate limiting for production use
- Monitor Azure OpenAI usage and costs

## 📝 License

This project is for educational and research purposes. Ensure you have proper licenses for all dependencies and data sources.

## 🐛 Troubleshooting

### Connection Issues
- Verify Snowflake credentials and network access
- Check Azure OpenAI API key and endpoint
- Ensure all required environment variables are set

### Performance Issues
- Consider implementing query caching
- Optimize SQL queries for large datasets
- Monitor API rate limits

### UI Issues
- Clear browser cache and restart Streamlit
- Check for conflicts with other Streamlit apps
- Verify all dependencies are correctly installed 