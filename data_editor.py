import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union
import base64
from io import BytesIO

class DataEditor:
    """
    A class for editing and filtering data in pandas DataFrames.
    Provides methods for row deletion, filtering, and exporting data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataEditor with a DataFrame.
        
        Args:
            df: The pandas DataFrame to edit
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.edit_history = []
    
    def delete_rows(self, indices: List[int]) -> Tuple[bool, str]:
        """
        Delete rows by their indices.
        
        Args:
            indices: List of row indices to delete
            
        Returns:
            A tuple of (success, message)
        """
        try:
            if not indices:
                return False, "No rows selected for deletion"
            
            original_count = len(self.df)
            self.df = self.df.drop(indices)
            removed_count = original_count - len(self.df)
            
            # Record the operation in edit history
            self.edit_history.append({
                'operation': 'delete_rows',
                'indices': indices,
                'removed_count': removed_count
            })
            
            return True, f"Removed {removed_count} rows"
            
        except Exception as e:
            return False, f"Error removing rows: {str(e)}"
    
    def filter_data(self, column: str, condition: str, value: Any) -> Tuple[bool, str]:
        """
        Filter data based on a condition.
        
        Args:
            column: Column to filter on
            condition: Condition operator ('==', '!=', '>', '>=', '<', '<=', 'contains', 'startswith', 'endswith')
            value: Value to compare against
            
        Returns:
            A tuple of (success, message)
        """
        try:
            if column not in self.df.columns:
                return False, f"Column '{column}' not found in the DataFrame"
            
            original_count = len(self.df)
            
            if condition == '==':
                self.df = self.df[self.df[column] == value]
            elif condition == '!=':
                self.df = self.df[self.df[column] != value]
            elif condition == '>':
                self.df = self.df[self.df[column] > value]
            elif condition == '>=':
                self.df = self.df[self.df[column] >= value]
            elif condition == '<':
                self.df = self.df[self.df[column] < value]
            elif condition == '<=':
                self.df = self.df[self.df[column] <= value]
            elif condition == 'contains':
                self.df = self.df[self.df[column].astype(str).str.contains(str(value), na=False)]
            elif condition == 'startswith':
                self.df = self.df[self.df[column].astype(str).str.startswith(str(value), na=False)]
            elif condition == 'endswith':
                self.df = self.df[self.df[column].astype(str).str.endswith(str(value), na=False)]
            else:
                return False, f"Unsupported condition: {condition}"
            
            filtered_count = original_count - len(self.df)
            
            # Record the operation in edit history
            self.edit_history.append({
                'operation': 'filter_data',
                'column': column,
                'condition': condition,
                'value': value,
                'filtered_count': filtered_count
            })
            
            return True, f"Filtered out {filtered_count} rows"
            
        except Exception as e:
            return False, f"Error filtering data: {str(e)}"
    
    def apply_complex_filter(self, filter_expr: str) -> Tuple[bool, str]:
        """
        Apply a complex filter using a query expression.
        
        Args:
            filter_expr: Query expression (e.g., "column1 > 10 and column2 == 'value'")
            
        Returns:
            A tuple of (success, message)
        """
        try:
            original_count = len(self.df)
            self.df = self.df.query(filter_expr)
            filtered_count = original_count - len(self.df)
            
            # Record the operation in edit history
            self.edit_history.append({
                'operation': 'complex_filter',
                'expression': filter_expr,
                'filtered_count': filtered_count
            })
            
            return True, f"Applied complex filter: {filtered_count} rows filtered out"
            
        except Exception as e:
            return False, f"Error applying complex filter: {str(e)}"
    
    def reset_filters(self) -> Tuple[bool, str]:
        """
        Reset all filters and return to the original data.
        
        Returns:
            A tuple of (success, message)
        """
        try:
            self.df = self.original_df.copy()
            
            # Record the operation in edit history
            self.edit_history.append({
                'operation': 'reset_filters'
            })
            
            return True, "Reset all filters and edits"
            
        except Exception as e:
            return False, f"Error resetting filters: {str(e)}"
    
    def export_to_excel(self) -> Tuple[bool, BytesIO, str]:
        """
        Export the current DataFrame to Excel format.
        
        Returns:
            A tuple of (success, BytesIO object, filename)
        """
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                self.df.to_excel(writer, index=False)
            
            output.seek(0)
            filename = "data_export.xlsx"
            
            return True, output, filename
            
        except Exception as e:
            return False, None, f"Error exporting to Excel: {str(e)}"
    
    def export_to_csv(self) -> Tuple[bool, BytesIO, str]:
        """
        Export the current DataFrame to CSV format.
        
        Returns:
            A tuple of (success, BytesIO object, filename)
        """
        try:
            output = BytesIO()
            self.df.to_csv(output, index=False)
            
            output.seek(0)
            filename = "data_export.csv"
            
            return True, output, filename
            
        except Exception as e:
            return False, None, f"Error exporting to CSV: {str(e)}"
    
    def get_edit_history(self) -> List[Dict[str, Any]]:
        """Get the edit history."""
        return self.edit_history.copy()
    
    def get_edited_data(self) -> pd.DataFrame:
        """Get the edited DataFrame."""
        return self.df.copy()
    
    def get_row_by_index(self, index: int) -> Optional[pd.Series]:
        """
        Get a row by its index.
        
        Args:
            index: Row index
            
        Returns:
            A pandas Series or None if index not found
        """
        try:
            return self.df.loc[index]
        except:
            return None
    
    def get_download_link(self, file_format: str = 'excel') -> Tuple[bool, str]:
        """
        Generate a download link for the current DataFrame.
        
        Args:
            file_format: Format to export ('excel' or 'csv')
            
        Returns:
            A tuple of (success, download link HTML)
        """
        try:
            if file_format == 'excel':
                success, buffer, filename = self.export_to_excel()
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif file_format == 'csv':
                success, buffer, filename = self.export_to_csv()
                mime_type = "text/csv"
            else:
                return False, "Unsupported file format"
            
            if not success:
                return False, f"Error generating download link: {buffer}"
            
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {file_format.upper()} file</a>'
            
            return True, href
            
        except Exception as e:
            return False, f"Error generating download link: {str(e)}"
