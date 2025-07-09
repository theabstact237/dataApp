# Data Analytics Streamlit App

A comprehensive data analytics application built with Streamlit that supports multiple file formats, data cleaning, insights generation, and advanced table operations.

## Features

- **Multi-format Data Import**: Support for CSV, Excel (XLS/XLSX), JSON, and SQL databases
- **Data Cleaning & Validation**:
  - Data type conversion
  - Missing value detection and handling
  - Duplicate detection and removal
  - Text standardization
  - Custom validation rules
- **Data Insights**:
  - Summary statistics
  - Missing data analysis
  - Outlier detection (Z-score and IQR methods)
  - Distribution analysis
  - Correlation analysis
  - Custom visualizations
- **Row Editing**:
  - Row selection and deletion
  - Data filtering with various conditions
  - Advanced filtering with complex expressions
- **Table Operations**:
  - Pivot tables
  - Data aggregation
  - Conditional operations
  - Cross-tabulation
- **Export Options**:
  - Excel (XLSX) export
  - CSV export

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly openpyxl sqlalchemy
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

1. **Data Upload**: Start by uploading a file (CSV, Excel, JSON) or connecting to a SQL database
2. **Data Preview**: View basic statistics and preview your data
3. **Data Cleaning**: Clean and validate your data using various tools
4. **Data Insights**: Generate insights and visualizations
5. **Row Editing**: Edit and filter rows based on conditions
6. **Table Operations**: Create pivot tables and perform aggregations
7. **Export**: Export your cleaned and processed data

## Project Structure

- `app.py`: Main Streamlit application
- `data_cleaning.py`: Data cleaning and validation functionality
- `data_analyzer.py`: Data analysis and visualization functionality
- `data_editor.py`: Row editing and filtering functionality
- `table_operator.py`: Table operations functionality
- `generate_sample_data.py`: Script to generate sample data for testing

## Sample Data

The repository includes sample data files in various formats:
- `sample_data.csv`: CSV format
- `sample_data.xlsx`: Excel format
- `sample_data.json`: JSON format
- `sample_data.db`: SQLite database

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- OpenPyXL
- SQLAlchemy

## Deployment

The app can be deployed using Streamlit Sharing, Heroku, or any other platform that supports Python web applications.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
