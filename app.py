import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import io
import os
import sqlite3
from sqlalchemy import create_engine
import base64
from io import BytesIO
from data_cleaning import DataCleaner
from data_analyzer import DataAnalyzer
from data_editor import DataEditor
from table_operator import TableOperator

# Set page configuration
st.set_page_config(
    page_title="Data Analytics App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to add author info in sidebar
def add_author_info():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### About the Author")
        
        # Load and display profile image
        import os
        image_path = os.path.join(os.path.dirname(__file__), "karl_siaka_profile.jpg")
        st.image(image_path, width=150)
        
        # Author info
        st.markdown("**Built by Karl Siaka**")
        st.markdown("*IT enthusiast, Data engineer*")
        
        # LinkedIn link
        st.markdown("[Connect on LinkedIn](https://linkedin.com/in/siaka-karl)")
        st.markdown("---")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "upload"
if 'data_cleaner' not in st.session_state:
    st.session_state.data_cleaner = None
if 'data_analyzer' not in st.session_state:
    st.session_state.data_analyzer = None
if 'data_editor' not in st.session_state:
    st.session_state.data_editor = None
if 'table_operator' not in st.session_state:
    st.session_state.table_operator = None
if 'selected_rows' not in st.session_state:
    st.session_state.selected_rows = []

# Function to parse CSV files
def parse_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, f"Error parsing CSV file: {str(e)}"

# Function to parse Excel files
def parse_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df, None
    except Exception as e:
        return None, f"Error parsing Excel file: {str(e)}"

# Function to parse JSON files
def parse_json(uploaded_file):
    try:
        content = uploaded_file.read()
        json_data = json.loads(content)
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            df = pd.json_normalize(json_data)
        elif isinstance(json_data, dict):
            # Check if it's a nested structure
            if any(isinstance(json_data[key], (dict, list)) for key in json_data):
                df = pd.json_normalize(json_data)
            else:
                df = pd.DataFrame([json_data])
        else:
            return None, "Unsupported JSON structure"
        
        return df, None
    except Exception as e:
        return None, f"Error parsing JSON file: {str(e)}"

# Function to execute SQL query
def execute_sql_query(query, connection_string):
    try:
        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return df, None
    except Exception as e:
        return None, f"Error executing SQL query: {str(e)}"

# Function to connect to SQLite database
def connect_sqlite(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn, None
    except Exception as e:
        return None, f"Error connecting to SQLite database: {str(e)}"

# Main app layout
def main():
    # Sidebar navigation
    st.sidebar.title("Data Analytics App")
    
    # Add author info to sidebar
    add_author_info()
    
    # Only show navigation when data is loaded
    if st.session_state.df is not None:
        pages = ["Data Upload", "Data Preview", "Data Cleaning & Validation", 
                "Data Insights", "Row Editing", "Table Operations", "Export"]
        selected_page = st.sidebar.radio("Navigation", pages)
        
        if selected_page == "Data Upload":
            st.session_state.current_page = "upload"
        elif selected_page == "Data Preview":
            st.session_state.current_page = "preview"
        elif selected_page == "Data Cleaning & Validation":
            st.session_state.current_page = "cleaning"
        elif selected_page == "Data Insights":
            st.session_state.current_page = "insights"
        elif selected_page == "Row Editing":
            st.session_state.current_page = "editing"
        elif selected_page == "Table Operations":
            st.session_state.current_page = "operations"
        elif selected_page == "Export":
            st.session_state.current_page = "export"
    else:
        st.session_state.current_page = "upload"
    
    # Display the selected page
    if st.session_state.current_page == "upload":
        display_upload_page()
    elif st.session_state.current_page == "preview":
        display_preview_page()
    elif st.session_state.current_page == "cleaning":
        display_cleaning_page()
    elif st.session_state.current_page == "insights":
        display_insights_page()
    elif st.session_state.current_page == "editing":
        display_editing_page()
    elif st.session_state.current_page == "operations":
        display_operations_page()
    elif st.session_state.current_page == "export":
        display_export_page()

# Upload page
def display_upload_page():
    st.title("Data Analytics App")
    
    # Comprehensive Usage Guide
    with st.expander("📚 How to Use This App", expanded=True):
        st.markdown("""
        ## Comprehensive Guide to Using the Data Analytics App
        
        This app is designed to help you clean, analyze, and transform data from various sources. Here's how to make the most of its features:
        
        ### Getting Started
        
        1. **Upload Data**: Start by uploading a file (CSV, Excel, JSON) or connecting to a database
        2. **Preview Data**: Review your data before processing to understand its structure
        3. **Navigate**: Use the sidebar to move between different app sections
        
        ### Data Cleaning & Validation
        
        * **Data Types**: Convert columns to appropriate data types for accurate analysis
        * **Missing Values**: Detect and handle missing data using various methods (mean, median, custom values)
        * **Duplicates**: Find and remove duplicate entries based on selected columns
        * **Text Standardization**: Normalize text data for consistency
        
        ### Data Insights
        
        * **Missing Data Analysis**: Visualize patterns of missing values
        * **Outlier Detection**: Find anomalies using Z-score or IQR methods
        * **Distributions**: Understand your data's shape through histograms and statistics
        * **Correlations**: Discover relationships between numeric variables
        
        ### Row Editing
        
        * **Row Selection**: Choose specific rows to delete or modify
        * **Filtering**: Apply conditions to focus on relevant data subsets
        * **Custom Filters**: Create complex filters with multiple conditions
        
        ### Table Operations
        
        * **Pivot Tables**: Summarize data with cross-tabulations
        * **Aggregations**: Calculate averages, sums, and other statistics
        * **Conditional Operations**: Count or aggregate based on specific conditions
        
        ### Exporting Results
        
        * **Format Selection**: Choose between Excel or CSV formats
        * **File Naming**: Customize output filenames
        * **Download**: Get your processed data for use in other applications
        
        ### Tips for Best Results
        
        * Clean your data before analysis for more accurate insights
        * Use appropriate visualization types for different data characteristics
        * Export at various stages if you need to track changes
        * For large datasets, consider filtering to relevant subsets first
        """)
    
    st.write("Upload your data file or connect to a database to begin analysis.")
    
    # File upload options
    upload_option = st.radio(
        "Select data source:",
        ["Upload File", "SQL Connection"]
    )
    
    if upload_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    if file_extension == "csv":
                        df, error = parse_csv(uploaded_file)
                    elif file_extension in ["xlsx", "xls"]:
                        df, error = parse_excel(uploaded_file)
                    elif file_extension == "json":
                        df, error = parse_json(uploaded_file)
                    else:
                        df, error = None, "Unsupported file format"
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.session_state.file_info = {
                            "filename": uploaded_file.name,
                            "format": file_extension,
                            "size": uploaded_file.size
                        }
                        # Initialize data modules
                        st.session_state.data_cleaner = DataCleaner(df)
                        st.session_state.data_analyzer = DataAnalyzer(df)
                        st.session_state.data_editor = DataEditor(df)
                        st.session_state.table_operator = TableOperator(df)
                        st.success(f"File processed successfully! {len(df)} rows and {len(df.columns)} columns loaded.")
                        st.session_state.current_page = "preview"
                        st.rerun()
    
    else:  # SQL Connection
        st.subheader("SQL Connection")
        
        # SQL Connection Instructions
        with st.expander("SQL Connection Instructions", expanded=True):
            st.markdown("""
            ### How to Connect to a Database
            
            This app supports connecting to various database types for ETL (Extract, Transform, Load) operations. Follow these steps to connect to your database:
            
            #### General Connection Process:
            1. **Select your database type** from the dropdown below
            2. **Provide connection details** specific to your database type
            3. **Select a table** or write a custom query to extract data
            4. **Click "Load Data"** to import the data for analysis
            
            #### Database-Specific Instructions:
            
            **For SQLite:**
            - Upload your .db, .sqlite, or .sqlite3 file
            - Select a table from the dropdown that appears
            - The app will extract all data from the selected table
            
            **For MySQL/PostgreSQL:**
            - Provide connection details (host, port, database name, username, password)
            - Ensure your database server allows connections from external sources
            - You can use a custom SQL query to filter or join tables
            
            #### ETL Process:
            1. **Extract:** Data is pulled directly from your database tables
            2. **Transform:** Once loaded, use the app's data cleaning and validation features
            3. **Load:** After analysis, export the processed data to your preferred format
            
            #### Security Note:
            - Connection credentials are used only for the current session
            - No credentials are stored permanently
            - For sensitive data, consider using a secure connection
            """)
        
        db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"])
        
        if db_type == "SQLite":
            uploaded_db = st.file_uploader("Upload SQLite Database", type=["db", "sqlite", "sqlite3"])
            
            if uploaded_db is not None:
                # Save the uploaded SQLite file temporarily
                with open("temp_db.sqlite", "wb") as f:
                    f.write(uploaded_db.getbuffer())
                
                # Get tables in the database
                conn, error = connect_sqlite("temp_db.sqlite")
                
                if error:
                    st.error(error)
                else:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type=\'table\';")
                    tables = [table[0] for table in cursor.fetchall()]
                    
                    if tables:
                        selected_table = st.selectbox("Select a table", tables)
                        
                        if st.button("Load Table"):
                            query = f"SELECT * FROM {selected_table}"
                            df, error = execute_sql_query(query, "sqlite:///temp_db.sqlite")
                            
                            if error:
                                st.error(error)
                            else:
                                st.session_state.df = df
                                st.session_state.original_df = df.copy()
                                st.session_state.file_info = {
                                    "filename": uploaded_db.name,
                                    "format": "sqlite",
                                    "table": selected_table
                                }
                                # Initialize data modules
                                st.session_state.data_cleaner = DataCleaner(df)
                                st.session_state.data_analyzer = DataAnalyzer(df)
                                st.session_state.data_editor = DataEditor(df)
                                st.session_state.table_operator = TableOperator(df)
                                st.success(f"Table loaded successfully! {len(df)} rows and {len(df.columns)} columns loaded.")
                                st.session_state.current_page = "preview"
                                st.rerun()
                    else:
                        st.warning("No tables found in the database.")
        
        else:  # MySQL or PostgreSQL
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
            database = st.text_input("Database")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Connect"):
                if not all([host, port, database, username]):
                    st.error("Please fill in all connection details.")
                else:
                    if db_type == "MySQL":
                        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                    else:  # PostgreSQL
                        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                    
                    try:
                        engine = create_engine(connection_string)
                        inspector = engine.inspect()
                        tables = inspector.get_table_names()
                        
                        if tables:
                            st.session_state.db_connection = connection_string
                            st.session_state.db_tables = tables
                            st.success("Connected successfully!")
                            st.session_state.current_page = "sql_query"
                            st.rerun()
                        else:
                            st.warning("No tables found in the database.")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

# Data preview page
def display_preview_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    st.title("Data Preview")
    
    # Display file information
    if st.session_state.file_info:
        st.subheader("File Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"**Filename:** {st.session_state.file_info.get('filename', 'N/A')}")
        with info_col2:
            st.write(f"**Format:** {st.session_state.file_info.get('format', 'N/A')}")
        with info_col3:
            if 'size' in st.session_state.file_info:
                size_kb = st.session_state.file_info['size'] / 1024
                st.write(f"**Size:** {size_kb:.2f} KB")
    
    # Display basic statistics
    st.subheader("Data Summary")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Rows", len(st.session_state.df))
    with stats_col2:
        st.metric("Columns", len(st.session_state.df.columns))
    with stats_col3:
        missing_values = st.session_state.df.isna().sum().sum()
        st.metric("Missing Values", missing_values)
    with stats_col4:
        duplicate_rows = len(st.session_state.df) - len(st.session_state.df.drop_duplicates())
        st.metric("Duplicate Rows", duplicate_rows)
    
    # Display data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame(st.session_state.df.dtypes, columns=["Data Type"])
    dtypes_df = dtypes_df.reset_index().rename(columns={"index": "Column"})
    st.dataframe(dtypes_df)
    
    # Display data preview
    st.subheader("Data Preview")
    rows_to_show = st.slider("Number of rows to display", 5, 100, 10)
    st.dataframe(st.session_state.df.head(rows_to_show))

# Data cleaning and validation page
def display_cleaning_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    if st.session_state.data_cleaner is None:
        st.session_state.data_cleaner = DataCleaner(st.session_state.df)
    
    st.title("Data Cleaning & Validation")
    
    # Create tabs for different cleaning operations
    tabs = st.tabs([
        "Data Types", 
        "Missing Values", 
        "Duplicates", 
        "Text Standardization",
        "Custom Validation",
        "Cleaning History"
    ])
    
    # Data Types tab
    with tabs[0]:
        st.header("Data Type Validation")
        st.write("Convert columns to appropriate data types.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_column = st.selectbox(
                "Select column to convert",
                options=st.session_state.df.columns,
                key="dtype_column"
            )
        
        with col2:
            target_type = st.selectbox(
                "Select target data type",
                options=["int", "float", "str", "datetime", "category", "bool"],
                key="target_dtype"
            )
        
        if st.button("Convert Data Type"):
            success, message = st.session_state.data_cleaner.convert_column_type(selected_column, target_type)
            
            if success:
                st.success(message)
                # Update the session state dataframe
                st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                # Update the data analyzer, editor, and operator with the cleaned data
                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.data_editor = DataEditor(st.session_state.df)
                st.session_state.table_operator = TableOperator(st.session_state.df)
            else:
                st.error(message)
        
        # Display current data types
        st.subheader("Current Data Types")
        dtypes = st.session_state.data_cleaner.get_data_types()
        dtypes_df = pd.DataFrame(list(dtypes.items()), columns=["Column", "Data Type"])
        st.dataframe(dtypes_df)
    
    # Missing Values tab
    with tabs[1]:
        st.header("Missing Values Handling")
        
        # Display missing values summary
        missing_df = st.session_state.data_cleaner.get_missing_data_summary()
        
        if len(missing_df) > 0 and missing_df["Missing Count"].sum() > 0:
            st.subheader("Missing Values Summary")
            st.dataframe(missing_df)
            
            # Missing value handling options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_column = st.selectbox(
                    "Select column with missing values",
                    options=missing_df[missing_df["Missing Count"] > 0]["Column"].tolist(),
                    key="missing_column"
                )
            
            with col2:
                handling_method = st.selectbox(
                    "Select handling method",
                    options=["drop", "fill_value", "mean", "median", "mode", "ffill", "bfill"],
                    key="missing_method"
                )
            
            with col3:
                fill_value = None
                if handling_method == "fill_value":
                    fill_value = st.text_input("Fill value", key="fill_value")
            
            if st.button("Handle Missing Values"):
                success, message = st.session_state.data_cleaner.handle_missing_values(
                    selected_column, handling_method, fill_value
                )
                
                if success:
                    st.success(message)
                    # Update the session state dataframe
                    st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                    # Update the data analyzer, editor, and operator with the cleaned data
                    st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                    st.session_state.data_editor = DataEditor(st.session_state.df)
                    st.session_state.table_operator = TableOperator(st.session_state.df)
                    # Refresh the missing values summary
                    missing_df = st.session_state.data_cleaner.get_missing_data_summary()
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.success("No missing values found in the dataset.")
    
    # Duplicates tab
    with tabs[2]:
        st.header("Duplicate Detection & Removal")
        
        # Get duplicate summary
        duplicate_summary = st.session_state.data_cleaner.get_duplicate_summary()
        
        if duplicate_summary["duplicate_count"] > 0:
            st.subheader("Duplicate Rows Summary")
            st.write(f"Found {duplicate_summary['duplicate_count']} duplicate rows ({duplicate_summary['duplicate_percentage']}% of data).")
            
            # Duplicate removal options
            st.subheader("Remove Duplicates")
            
            # Select columns for considering duplicates
            selected_columns = st.multiselect(
                "Select columns to consider for duplicates (leave empty for all columns)",
                options=st.session_state.df.columns,
                key="duplicate_columns"
            )
            
            if st.button("Remove Duplicate Rows"):
                success, message = st.session_state.data_cleaner.remove_duplicates(
                    subset=selected_columns if selected_columns else None
                )
                
                if success:
                    st.success(message)
                    # Update the session state dataframe
                    st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                    # Update the data analyzer, editor, and operator with the cleaned data
                    st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                    st.session_state.data_editor = DataEditor(st.session_state.df)
                    st.session_state.table_operator = TableOperator(st.session_state.df)
                    # Refresh the duplicate summary
                    duplicate_summary = st.session_state.data_cleaner.get_duplicate_summary()
                    st.rerun()
                else:
                    st.error(message)
            
            # Show duplicate rows
            if st.checkbox("Show duplicate rows"):
                if duplicate_summary["duplicate_indices"]:
                    duplicate_rows = st.session_state.df.loc[duplicate_summary["duplicate_indices"]]
                    st.dataframe(duplicate_rows)
                else:
                    st.write("No duplicate rows to display.")
        else:
            st.success("No duplicate rows found in the dataset.")
    
    # Text Standardization tab
    with tabs[3]:
        st.header("Text Standardization")
        st.write("Standardize text data by converting case and removing whitespace.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_column = st.selectbox(
                "Select text column to standardize",
                options=st.session_state.df.columns,
                key="text_column"
            )
        
        with col2:
            case_option = st.selectbox(
                "Select case conversion",
                options=["lower", "upper", "title", "none"],
                key="case_option"
            )
        
        with col3:
            strip_option = st.checkbox("Strip whitespace", value=True, key="strip_option")
        
        if st.button("Standardize Text"):
            success, message = st.session_state.data_cleaner.standardize_text(
                selected_column, case_option, strip_option
            )
            
            if success:
                st.success(message)
                # Update the session state dataframe
                st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                # Update the data analyzer, editor, and operator with the cleaned data
                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.data_editor = DataEditor(st.session_state.df)
                st.session_state.table_operator = TableOperator(st.session_state.df)
            else:
                st.error(message)
        
        # Preview of text column
        if st.checkbox("Show text column preview"):
            st.subheader(f"Preview of column: {selected_column}")
            st.dataframe(st.session_state.df[[selected_column]].head(10))
    
    # Custom Validation tab
    with tabs[4]:
        st.header("Custom Validation")
        st.write("Apply custom validation rules to your data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_column = st.selectbox(
                "Select column to validate",
                options=st.session_state.df.columns,
                key="validation_column"
            )
        
        with col2:
            condition = st.text_input(
                "Enter validation condition (e.g., '> 0', '== \"Valid\"')",
                key="validation_condition"
            )
        
        if st.button("Validate Data"):
            if condition:
                validation_result = st.session_state.data_cleaner.apply_custom_validation(
                    selected_column, condition
                )
                
                if validation_result["success"]:
                    st.success(validation_result["message"])
                    
                    # Show invalid rows
                    if validation_result["invalid_count"] > 0:
                        st.subheader("Invalid Rows")
                        invalid_rows = st.session_state.df.loc[validation_result["invalid_indices"]]
                        st.dataframe(invalid_rows)
                        
                        # Option to remove invalid rows
                        if st.button("Remove Invalid Rows"):
                            success, message = st.session_state.data_cleaner.remove_rows_by_indices(
                                validation_result["invalid_indices"]
                            )
                            
                            if success:
                                st.success(message)
                                # Update the session state dataframe
                                st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                                # Update the data analyzer, editor, and operator with the cleaned data
                                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                                st.session_state.data_editor = DataEditor(st.session_state.df)
                                st.session_state.table_operator = TableOperator(st.session_state.df)
                                st.rerun()
                            else:
                                st.error(message)
                else:
                    st.error(validation_result["message"])
            else:
                st.warning("Please enter a validation condition.")
    
    # Cleaning History tab
    with tabs[5]:
        st.header("Cleaning History")
        
        cleaning_history = st.session_state.data_cleaner.get_cleaning_history()
        
        if cleaning_history:
            for i, operation in enumerate(cleaning_history):
                with st.expander(f"Operation {i+1}: {operation['operation']}"):
                    for key, value in operation.items():
                        if key != 'operation':
                            st.write(f"**{key}:** {value}")
            
            # Option to reset to original data
            if st.button("Reset to Original Data"):
                st.session_state.data_cleaner.reset_to_original()
                st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                # Update the data analyzer, editor, and operator with the original data
                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.data_editor = DataEditor(st.session_state.df)
                st.session_state.table_operator = TableOperator(st.session_state.df)
                st.success("Data reset to original state.")
                st.rerun()
        else:
            st.info("No cleaning operations have been performed yet.")
    
    # Show current data preview
    st.subheader("Current Data Preview")
    rows_to_show = st.slider("Number of rows to display", 5, 100, 10, key="clean_preview_rows")
    st.dataframe(st.session_state.df.head(rows_to_show))

# Data insights page
def display_insights_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    if st.session_state.data_analyzer is None:
        st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
    
    st.title("Data Insights")
    
    # Create tabs for different insights
    tabs = st.tabs([
        "Summary Statistics", 
        "Missing Data Analysis", 
        "Outlier Detection", 
        "Distributions",
        "Correlations",
        "Visualizations"
    ])
    
    # Summary Statistics tab
    with tabs[0]:
        st.header("Summary Statistics")
        
        # Numeric summary
        st.subheader("Numeric Columns")
        numeric_summary = st.session_state.data_analyzer.get_summary_statistics()
        if not numeric_summary.empty:
            st.dataframe(numeric_summary)
        else:
            st.info("No numeric columns found in the dataset.")
        
        # Categorical summary
        st.subheader("Categorical Columns")
        categorical_summaries = st.session_state.data_analyzer.get_categorical_summary()
        
        if categorical_summaries:
            selected_cat_col = st.selectbox(
                "Select categorical column",
                options=list(categorical_summaries.keys()),
                key="cat_summary_col"
            )
            
            st.dataframe(categorical_summaries[selected_cat_col])
        else:
            st.info("No categorical columns found in the dataset.")
    
    # Missing Data Analysis tab
    with tabs[1]:
        st.header("Missing Data Analysis")
        
        # Get missing data analysis
        missing_analysis = st.session_state.data_analyzer.analyze_missing_data()
        
        # Display overall statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cells", missing_analysis["total_cells"])
        with col2:
            st.metric("Missing Cells", missing_analysis["total_missing"])
        with col3:
            st.metric("Missing Percentage", f"{missing_analysis['overall_missing_percentage']}%")
        
        # Display missing data summary
        if missing_analysis["total_missing"] > 0:
            st.subheader("Missing Data by Column")
            st.dataframe(missing_analysis["missing_summary"])
            
            # Display missing data heatmap
            st.subheader("Missing Data Heatmap")
            missing_heatmap = st.session_state.data_analyzer.create_missing_data_heatmap()
            
            if missing_heatmap:
                st.pyplot(missing_heatmap)
            else:
                st.info("Could not generate missing data heatmap.")
        else:
            st.success("No missing data found in the dataset.")
    
    # Outlier Detection tab
    with tabs[2]:
        st.header("Outlier Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_method = st.selectbox(
                "Select outlier detection method",
                options=["zscore", "iqr"],
                key="outlier_method"
            )
        
        with col2:
            if outlier_method == "zscore":
                threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="zscore_threshold")
            else:  # IQR
                threshold = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_threshold")
        
        if st.button("Detect Outliers"):
            outlier_results = st.session_state.data_analyzer.detect_outliers(outlier_method, threshold)
            
            if outlier_results["outliers_found"]:
                st.success(f"Found outliers in {len(outlier_results['columns_with_outliers'])} columns using {outlier_method.upper()} method.")
                
                # Display outlier summary
                st.subheader("Outlier Summary")
                
                for col in outlier_results["columns_with_outliers"]:
                    col_info = outlier_results["outliers_by_column"][col]
                    st.write(f"**{col}:** {col_info['count']} outliers ({col_info['percentage']}% of data)")
                
                # Option to view outliers
                selected_outlier_col = st.selectbox(
                    "Select column to view outliers",
                    options=outlier_results["columns_with_outliers"],
                    key="outlier_col"
                )
                
                if selected_outlier_col:
                    outlier_indices = outlier_results["outliers_by_column"][selected_outlier_col]["indices"]
                    outlier_rows = st.session_state.df.loc[outlier_indices]
                    
                    st.subheader(f"Outliers in column: {selected_outlier_col}")
                    st.dataframe(outlier_rows)
                    
                    # Option to remove outliers
                    if st.button("Remove Selected Outliers"):
                        if st.session_state.data_cleaner:
                            success, message = st.session_state.data_cleaner.remove_rows_by_indices(outlier_indices)
                            
                            if success:
                                st.success(message)
                                # Update the session state dataframe
                                st.session_state.df = st.session_state.data_cleaner.get_cleaned_data()
                                # Update the data analyzer, editor, and operator with the cleaned data
                                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                                st.session_state.data_editor = DataEditor(st.session_state.df)
                                st.session_state.table_operator = TableOperator(st.session_state.df)
                                st.rerun()
                            else:
                                st.error(message)
                
                # Display boxplots
                st.subheader("Boxplots")
                boxplots = st.session_state.data_analyzer.create_boxplots(outlier_results["columns_with_outliers"])
                
                if boxplots:
                    st.pyplot(boxplots)
            else:
                st.info(f"No outliers found using {outlier_method.upper()} method with threshold {threshold}.")
    
    # Distributions tab
    with tabs[3]:
        st.header("Data Distributions")
        
        # Get numeric columns
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            selected_dist_col = st.selectbox(
                "Select column to view distribution",
                options=numeric_cols,
                key="dist_col"
            )
            
            if selected_dist_col:
                # Create distribution plot
                dist_plots = st.session_state.data_analyzer.create_distribution_plots([selected_dist_col])
                
                if dist_plots:
                    st.pyplot(dist_plots[selected_dist_col])
                    
                    # Display basic statistics for the selected column
                    st.subheader(f"Statistics for {selected_dist_col}")
                    stats = st.session_state.df[selected_dist_col].describe()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['50%']:.2f}")
                    with col3:
                        st.metric("Std Dev", f"{stats['std']:.2f}")
                    with col4:
                        skew = st.session_state.df[selected_dist_col].skew()
                        st.metric("Skewness", f"{skew:.2f}")
        else:
            st.info("No numeric columns found in the dataset.")
    
    # Correlations tab
    with tabs[4]:
        st.header("Correlation Analysis")
        
        # Create correlation heatmap
        corr_heatmap = st.session_state.data_analyzer.create_correlation_heatmap()
        
        if corr_heatmap:
            st.pyplot(corr_heatmap)
            
            # Option to view scatter matrix
            if st.checkbox("Show scatter matrix"):
                # Get numeric columns
                numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                
                if len(numeric_cols) > 1:
                    # Allow user to select columns for scatter matrix
                    selected_scatter_cols = st.multiselect(
                        "Select columns for scatter matrix (max 4 recommended)",
                        options=numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key="scatter_cols"
                    )
                    
                    if selected_scatter_cols and len(selected_scatter_cols) >= 2:
                        scatter_matrix = st.session_state.data_analyzer.create_scatter_matrix(selected_scatter_cols)
                        
                        if scatter_matrix:
                            st.pyplot(scatter_matrix)
                    else:
                        st.info("Please select at least 2 columns for the scatter matrix.")
        else:
            st.info("Could not generate correlation heatmap. Need at least 2 numeric columns.")
    
    # Visualizations tab
    with tabs[5]:
        st.header("Custom Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization type",
            options=["Bar Chart", "Time Series"],
            key="viz_type"
        )
        
        if viz_type == "Bar Chart":
            # Bar chart options
            st.subheader("Bar Chart")
            
            # Get categorical columns for x-axis
            all_cols = st.session_state.df.columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox(
                    "Select X-axis column (categories)",
                    options=all_cols,
                    key="bar_x_col"
                )
            
            with col2:
                y_col_options = ["None"] + all_cols
                y_col = st.selectbox(
                    "Select Y-axis column (values, optional)",
                    options=y_col_options,
                    key="bar_y_col"
                )
                y_col = None if y_col == "None" else y_col
            
            with col3:
                if y_col:
                    agg_func = st.selectbox(
                        "Select aggregation function",
                        options=["count", "sum", "mean", "median"],
                        key="bar_agg_func"
                    )
                else:
                    agg_func = "count"
            
            # Create bar chart
            bar_chart = st.session_state.data_analyzer.create_bar_chart(x_col, y_col, agg_func)
            
            if bar_chart:
                st.pyplot(bar_chart)
            else:
                st.error("Could not create bar chart with the selected columns.")
        
        elif viz_type == "Time Series":
            # Time series options
            st.subheader("Time Series Plot")
            
            # Get all columns
            all_cols = st.session_state.df.columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_col = st.selectbox(
                    "Select date/time column",
                    options=all_cols,
                    key="ts_date_col"
                )
            
            with col2:
                value_col = st.selectbox(
                    "Select value column",
                    options=all_cols,
                    key="ts_value_col"
                )
            
            with col3:
                freq_options = ["None", "D", "W", "M", "Q", "Y"]
                freq = st.selectbox(
                    "Select resampling frequency (optional)",
                    options=freq_options,
                    key="ts_freq"
                )
                freq = None if freq == "None" else freq
            
            # Create time series plot
            ts_plot = st.session_state.data_analyzer.create_time_series_plot(date_col, value_col, freq)
            
            if ts_plot:
                st.pyplot(ts_plot)
            else:
                st.error("Could not create time series plot. Ensure the date column contains valid dates.")

# Row editing page
def display_editing_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    if st.session_state.data_editor is None:
        st.session_state.data_editor = DataEditor(st.session_state.df)
    
    st.title("Row Editing")
    
    # Create tabs for different editing operations
    tabs = st.tabs([
        "Row Selection & Deletion", 
        "Data Filtering", 
        "Advanced Filtering"
    ])
    
    # Row Selection & Deletion tab
    with tabs[0]:
        st.header("Row Selection & Deletion")
        st.write("Select and delete specific rows from your dataset.")
        
        # Display data with row selection
        st.subheader("Select Rows to Delete")
        
        # Create a copy of the dataframe with a reset index for display
        display_df = st.session_state.df.copy().reset_index()
        display_df.rename(columns={'index': 'original_index'}, inplace=True)
        
        # Allow multi-row selection
        selected_rows = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=False,
            key="row_selection"
        )
        
        # Get the selected row indices
        if selected_rows is not None and len(selected_rows) > 0:
            selected_indices = [row['original_index'] for row in selected_rows.values()]
            
            if st.button("Delete Selected Rows"):
                if selected_indices:
                    success, message = st.session_state.data_editor.delete_rows(selected_indices)
                    
                    if success:
                        st.success(message)
                        # Update the session state dataframe
                        st.session_state.df = st.session_state.data_editor.get_edited_data()
                        # Update the data cleaner, analyzer, and operator with the edited data
                        st.session_state.data_cleaner = DataCleaner(st.session_state.df)
                        st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                        st.session_state.table_operator = TableOperator(st.session_state.df)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("No rows selected for deletion.")
    
    # Data Filtering tab
    with tabs[1]:
        st.header("Data Filtering")
        st.write("Filter your data based on column values.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_column = st.selectbox(
                "Select column to filter",
                options=st.session_state.df.columns,
                key="filter_column"
            )
        
        with col2:
            # Determine appropriate condition options based on column data type
            if pd.api.types.is_numeric_dtype(st.session_state.df[filter_column]):
                condition_options = ["==", "!=", ">", ">=", "<", "<="]
            else:
                condition_options = ["==", "!=", "contains", "startswith", "endswith"]
            
            filter_condition = st.selectbox(
                "Select condition",
                options=condition_options,
                key="filter_condition"
            )
        
        with col3:
            # For numeric columns, use a number input
            if pd.api.types.is_numeric_dtype(st.session_state.df[filter_column]):
                filter_value = st.number_input(
                    "Enter filter value",
                    value=float(st.session_state.df[filter_column].mean()) if not pd.isna(st.session_state.df[filter_column].mean()) else 0,
                    key="filter_value_num"
                )
            else:
                # For categorical columns, provide a selectbox with unique values
                if st.session_state.df[filter_column].nunique() < 20:
                    unique_values = st.session_state.df[filter_column].dropna().unique().tolist()
                    filter_value = st.selectbox(
                        "Select filter value",
                        options=unique_values,
                        key="filter_value_cat"
                    )
                else:
                    # For columns with many unique values, use a text input
                    filter_value = st.text_input(
                        "Enter filter value",
                        key="filter_value_text"
                    )
        
        if st.button("Apply Filter"):
            success, message = st.session_state.data_editor.filter_data(
                filter_column, filter_condition, filter_value
            )
            
            if success:
                st.success(message)
                # Update the session state dataframe
                st.session_state.df = st.session_state.data_editor.get_edited_data()
                # Update the data cleaner, analyzer, and operator with the filtered data
                st.session_state.data_cleaner = DataCleaner(st.session_state.df)
                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.table_operator = TableOperator(st.session_state.df)
                st.rerun()
            else:
                st.error(message)
        
        if st.button("Reset Filters"):
            success, message = st.session_state.data_editor.reset_filters()
            
            if success:
                st.success(message)
                # Update the session state dataframe
                st.session_state.df = st.session_state.data_editor.get_edited_data()
                # Update the data cleaner, analyzer, and operator with the original data
                st.session_state.data_cleaner = DataCleaner(st.session_state.df)
                st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.table_operator = TableOperator(st.session_state.df)
                st.rerun()
            else:
                st.error(message)
    
    # Advanced Filtering tab
    with tabs[2]:
        st.header("Advanced Filtering")
        st.write("Apply complex filters using query expressions.")
        
        st.info("""
        Enter a query expression using pandas query syntax. Examples:
        - `Age > 30 and Salary > 50000`
        - `City == "New York" or City == "Los Angeles"`
        - `Name.str.startswith("A") and Age < 40`
        """)
        
        complex_filter = st.text_area(
            "Enter complex filter expression",
            height=100,
            key="complex_filter"
        )
        
        if st.button("Apply Complex Filter"):
            if complex_filter:
                success, message = st.session_state.data_editor.apply_complex_filter(complex_filter)
                
                if success:
                    st.success(message)
                    # Update the session state dataframe
                    st.session_state.df = st.session_state.data_editor.get_edited_data()
                    # Update the data cleaner, analyzer, and operator with the filtered data
                    st.session_state.data_cleaner = DataCleaner(st.session_state.df)
                    st.session_state.data_analyzer = DataAnalyzer(st.session_state.df)
                    st.session_state.table_operator = TableOperator(st.session_state.df)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter a filter expression.")
    
    # Show current data preview
    st.subheader("Current Data Preview")
    rows_to_show = st.slider("Number of rows to display", 5, 100, 10, key="edit_preview_rows")
    st.dataframe(st.session_state.df.head(rows_to_show))
    
    # Display row count
    st.info(f"Current dataset has {len(st.session_state.df)} rows and {len(st.session_state.df.columns)} columns.")

# Table operations page
def display_operations_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    if st.session_state.table_operator is None:
        st.session_state.table_operator = TableOperator(st.session_state.df)
    
    st.title("Table Operations")
    
    # Create tabs for different operations
    tabs = st.tabs([
        "Pivot Table", 
        "Aggregation", 
        "Conditional Operations",
        "Cross-Tabulation"
    ])
    
    # Pivot Table tab
    with tabs[0]:
        st.header("Pivot Table")
        st.write("Create a pivot table to summarize your data.")
        
        all_cols = st.session_state.df.columns.tolist()
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pivot_index = st.multiselect(
                "Select index columns",
                options=all_cols,
                key="pivot_index"
            )
        
        with col2:
            pivot_columns = st.multiselect(
                "Select columns (optional)",
                options=all_cols,
                key="pivot_columns"
            )
        
        with col3:
            pivot_values = st.multiselect(
                "Select value columns",
                options=numeric_cols,
                key="pivot_values"
            )
        
        with col4:
            pivot_aggfunc = st.selectbox(
                "Select aggregation function",
                options=["mean", "sum", "count", "min", "max"],
                key="pivot_aggfunc"
            )
        
        if st.button("Create Pivot Table"):
            if pivot_index and pivot_values:
                success, pivot_table, message = st.session_state.table_operator.create_pivot_table(
                    index=pivot_index,
                    columns=pivot_columns if pivot_columns else None,
                    values=pivot_values,
                    aggfunc=pivot_aggfunc
                )
                
                if success:
                    st.success(message)
                    st.subheader("Pivot Table Result")
                    st.dataframe(pivot_table)
                    
                    # Option to visualize pivot table as heatmap
                    if st.checkbox("Show heatmap visualization"):
                        heatmap = st.session_state.table_operator.create_heatmap(pivot_table)
                        if heatmap:
                            st.pyplot(heatmap)
                        else:
                            st.warning("Could not generate heatmap for this pivot table.")
                else:
                    st.error(message)
            else:
                st.warning("Please select at least one index column and one value column.")
    
    # Aggregation tab
    with tabs[1]:
        st.header("Aggregation")
        st.write("Group data and apply aggregation functions.")
        
        all_cols = st.session_state.df.columns.tolist()
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_by_cols = st.multiselect(
                "Select columns to group by",
                options=all_cols,
                key="group_by_cols"
            )
        
        with col2:
            agg_col = st.selectbox(
                "Select column to aggregate",
                options=numeric_cols,
                key="agg_col"
            )
        
        agg_funcs = st.multiselect(
            "Select aggregation functions",
            options=["mean", "sum", "count", "min", "max", "std", "var", "median"],
            key="agg_funcs"
        )
        
        if st.button("Aggregate Data"):
            if group_by_cols and agg_col and agg_funcs:
                agg_dict = {agg_col: agg_funcs}
                success, aggregated_data, message = st.session_state.table_operator.aggregate_data(
                    group_by=group_by_cols,
                    agg_columns=agg_dict
                )
                
                if success:
                    st.success(message)
                    st.subheader("Aggregation Result")
                    st.dataframe(aggregated_data)
                    
                    # Option to visualize aggregated data
                    if st.checkbox("Show bar chart visualization"):
                        bar_chart = st.session_state.table_operator.create_bar_chart(
                            aggregated_data,
                            title=f"Aggregation of {agg_col} by {', '.join(group_by_cols)}"
                        )
                        if bar_chart:
                            st.pyplot(bar_chart)
                        else:
                            st.warning("Could not generate bar chart for this aggregation.")
                else:
                    st.error(message)
            else:
                st.warning("Please select grouping columns, aggregation column, and functions.")
    
    # Conditional Operations tab
    with tabs[2]:
        st.header("Conditional Operations")
        st.write("Perform counts or aggregations based on conditions.")
        
        operation_type = st.radio(
            "Select operation type",
            options=["Conditional Count", "Conditional Aggregate"],
            key="cond_op_type"
        )
        
        all_cols = st.session_state.df.columns.tolist()
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cond_column = st.selectbox(
                "Select column for condition",
                options=all_cols,
                key="cond_column"
            )
        
        with col2:
            cond_condition = st.text_input(
                "Enter condition (e.g., '> 0', '== \"Value\"')",
                key="cond_condition"
            )
        
        if operation_type == "Conditional Aggregate":
            with col3:
                cond_agg_col = st.selectbox(
                    "Select column to aggregate",
                    options=numeric_cols,
                    key="cond_agg_col"
                )
            
            cond_agg_func = st.selectbox(
                "Select aggregation function",
                options=["mean", "sum", "min", "max", "median"],
                key="cond_agg_func"
            )
        
        if st.button("Perform Operation"):
            if cond_column and cond_condition:
                if operation_type == "Conditional Count":
                    success, count, message = st.session_state.table_operator.conditional_count(
                        cond_column, cond_condition
                    )
                    
                    if success:
                        st.success(message)
                        st.metric("Resulting Count", count)
                    else:
                        st.error(message)
                
                elif operation_type == "Conditional Aggregate":
                    if cond_agg_col and cond_agg_func:
                        success, result, message = st.session_state.table_operator.conditional_aggregate(
                            cond_column, cond_condition, cond_agg_col, cond_agg_func
                        )
                        
                        if success:
                            st.success(message)
                            st.metric(f"Resulting {cond_agg_func.capitalize()}", f"{result:.2f}")
                        else:
                            st.error(message)
                    else:
                        st.warning("Please select aggregation column and function.")
            else:
                st.warning("Please select condition column and enter a condition.")
    
    # Cross-Tabulation tab
    with tabs[3]:
        st.header("Cross-Tabulation")
        st.write("Create a cross-tabulation (contingency table) of two columns.")
        
        all_cols = st.session_state.df.columns.tolist()
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            crosstab_row = st.selectbox(
                "Select row column",
                options=all_cols,
                key="crosstab_row"
            )
        
        with col2:
            crosstab_col = st.selectbox(
                "Select column column",
                options=all_cols,
                key="crosstab_col"
            )
        
        with col3:
            crosstab_values = st.selectbox(
                "Select value column (optional)",
                options=["None"] + numeric_cols,
                key="crosstab_values"
            )
            crosstab_values = None if crosstab_values == "None" else crosstab_values
        
        with col4:
            if crosstab_values:
                crosstab_aggfunc = st.selectbox(
                    "Select aggregation function",
                    options=["mean", "sum", "count", "min", "max"],
                    key="crosstab_aggfunc"
                )
            else:
                crosstab_aggfunc = "count"
        
        if st.button("Create Cross-Tabulation"):
            if crosstab_row and crosstab_col:
                success, crosstab, message = st.session_state.table_operator.create_crosstab(
                    row=crosstab_row,
                    column=crosstab_col,
                    values=crosstab_values,
                    aggfunc=crosstab_aggfunc
                )
                
                if success:
                    st.success(message)
                    st.subheader("Cross-Tabulation Result")
                    st.dataframe(crosstab)
                    
                    # Option to visualize crosstab as heatmap
                    if st.checkbox("Show heatmap visualization", key="crosstab_heatmap_check"):
                        heatmap = st.session_state.table_operator.create_heatmap(crosstab)
                        if heatmap:
                            st.pyplot(heatmap)
                        else:
                            st.warning("Could not generate heatmap for this cross-tabulation.")
                else:
                    st.error(message)
            else:
                st.warning("Please select row and column columns.")

# Export page
def display_export_page():
    if st.session_state.df is None:
        st.warning("No data loaded. Please upload a file first.")
        st.session_state.current_page = "upload"
        st.rerun()
        return
    
    if st.session_state.data_editor is None:
        st.session_state.data_editor = DataEditor(st.session_state.df)
    
    st.title("Export Data")
    st.write("Export your cleaned and processed data to various formats.")
    
    # Export options
    export_format = st.radio(
        "Select export format",
        options=["Excel (.xlsx)", "CSV (.csv)"],
        key="export_format"
    )
    
    # File name input
    default_filename = st.session_state.file_info.get("filename", "data_export").split(".")[0] + "_cleaned"
    export_filename = st.text_input(
        "Enter file name (without extension)",
        value=default_filename,
        key="export_filename"
    )
    
    # Export button
    if st.button("Generate Export"):
        if export_format == "Excel (.xlsx)":
            success, download_link = st.session_state.data_editor.get_download_link("excel")
        else:  # CSV
            success, download_link = st.session_state.data_editor.get_download_link("csv")
        
        if success:
            st.success("Export generated successfully!")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.error(f"Error generating export: {download_link}")
    
    # Data preview
    st.subheader("Data Preview")
    st.write("Preview of the data that will be exported:")
    
    # Show a preview of the data
    preview_rows = min(10, len(st.session_state.df))
    st.dataframe(st.session_state.df.head(preview_rows))
    
    # Display export statistics
    st.subheader("Export Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(st.session_state.df))
    with col2:
        st.metric("Columns", len(st.session_state.df.columns))
    with col3:
        missing_values = st.session_state.df.isna().sum().sum()
        st.metric("Missing Values", missing_values)

if __name__ == "__main__":
    main()
