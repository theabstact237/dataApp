import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union

class DataCleaner:
    """
    A class for cleaning and validating data in pandas DataFrames.
    Provides methods for data type validation, missing value detection,
    duplicate handling, and format standardization.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Args:
            df: The pandas DataFrame to clean and validate
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_history = []
        
    def get_data_types(self) -> Dict[str, str]:
        """
        Get the data types of each column in the DataFrame.
        
        Returns:
            A dictionary mapping column names to their data types
        """
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}
    
    def get_missing_data_summary(self) -> pd.DataFrame:
        """
        Get a summary of missing data in the DataFrame.
        
        Returns:
            A DataFrame with columns for missing count and percentage
        """
        missing_count = self.df.isna().sum()
        missing_percentage = (missing_count / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percentage.values
        })
        
        return missing_df.sort_values('Missing Count', ascending=False)
    
    def get_duplicate_summary(self) -> Dict[str, Any]:
        """
        Get a summary of duplicate rows in the DataFrame.
        
        Returns:
            A dictionary with duplicate statistics
        """
        duplicate_rows = self.df.duplicated()
        duplicate_count = duplicate_rows.sum()
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_count / len(self.df) * 100, 2),
            'duplicate_indices': self.df[duplicate_rows].index.tolist()
        }
    
    def convert_column_type(self, column: str, new_type: str) -> Tuple[bool, str]:
        """
        Convert a column to a new data type.
        
        Args:
            column: The name of the column to convert
            new_type: The new data type ('int', 'float', 'str', 'datetime', 'category', 'bool')
            
        Returns:
            A tuple of (success, message)
        """
        if column not in self.df.columns:
            return False, f"Column '{column}' not found in the DataFrame"
        
        try:
            original_values = self.df[column].copy()
            
            if new_type == 'int':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
            elif new_type == 'float':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            elif new_type == 'str':
                self.df[column] = self.df[column].astype(str)
            elif new_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            elif new_type == 'category':
                self.df[column] = self.df[column].astype('category')
            elif new_type == 'bool':
                self.df[column] = self.df[column].astype(bool)
            else:
                return False, f"Unsupported data type: {new_type}"
            
            # Record the operation in cleaning history
            self.cleaning_history.append({
                'operation': 'convert_type',
                'column': column,
                'old_type': str(original_values.dtype),
                'new_type': new_type
            })
            
            return True, f"Successfully converted column '{column}' to {new_type}"
            
        except Exception as e:
            return False, f"Error converting column '{column}' to {new_type}: {str(e)}"
    
    def handle_missing_values(self, column: str, method: str, fill_value: Any = None) -> Tuple[bool, str]:
        """
        Handle missing values in a column.
        
        Args:
            column: The name of the column to process
            method: The method to use ('drop', 'fill_value', 'mean', 'median', 'mode', 'ffill', 'bfill')
            fill_value: The value to use when method is 'fill_value'
            
        Returns:
            A tuple of (success, message)
        """
        if column not in self.df.columns:
            return False, f"Column '{column}' not found in the DataFrame"
        
        try:
            missing_count = self.df[column].isna().sum()
            
            if missing_count == 0:
                return True, f"No missing values in column '{column}'"
            
            if method == 'drop':
                self.df = self.df.dropna(subset=[column])
                message = f"Dropped {missing_count} rows with missing values in column '{column}'"
            
            elif method == 'fill_value':
                self.df[column] = self.df[column].fillna(fill_value)
                message = f"Filled {missing_count} missing values in column '{column}' with {fill_value}"
            
            elif method == 'mean':
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    mean_value = self.df[column].mean()
                    self.df[column] = self.df[column].fillna(mean_value)
                    message = f"Filled {missing_count} missing values in column '{column}' with mean ({mean_value:.2f})"
                else:
                    return False, f"Cannot use mean for non-numeric column '{column}'"
            
            elif method == 'median':
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    median_value = self.df[column].median()
                    self.df[column] = self.df[column].fillna(median_value)
                    message = f"Filled {missing_count} missing values in column '{column}' with median ({median_value:.2f})"
                else:
                    return False, f"Cannot use median for non-numeric column '{column}'"
            
            elif method == 'mode':
                mode_value = self.df[column].mode()[0]
                self.df[column] = self.df[column].fillna(mode_value)
                message = f"Filled {missing_count} missing values in column '{column}' with mode ({mode_value})"
            
            elif method == 'ffill':
                self.df[column] = self.df[column].ffill()
                message = f"Filled {missing_count} missing values in column '{column}' using forward fill"
            
            elif method == 'bfill':
                self.df[column] = self.df[column].bfill()
                message = f"Filled {missing_count} missing values in column '{column}' using backward fill"
            
            else:
                return False, f"Unsupported method: {method}"
            
            # Record the operation in cleaning history
            self.cleaning_history.append({
                'operation': 'handle_missing',
                'column': column,
                'method': method,
                'fill_value': fill_value if method == 'fill_value' else None,
                'count': missing_count
            })
            
            return True, message
            
        except Exception as e:
            return False, f"Error handling missing values in column '{column}': {str(e)}"
    
    def remove_duplicates(self, subset: List[str] = None) -> Tuple[bool, str]:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset: List of columns to consider for identifying duplicates
            
        Returns:
            A tuple of (success, message)
        """
        try:
            original_count = len(self.df)
            
            if subset:
                # Check if all columns in subset exist in the DataFrame
                missing_cols = [col for col in subset if col not in self.df.columns]
                if missing_cols:
                    return False, f"Columns not found in DataFrame: {', '.join(missing_cols)}"
                
                self.df = self.df.drop_duplicates(subset=subset)
                removed_count = original_count - len(self.df)
                message = f"Removed {removed_count} duplicate rows based on columns: {', '.join(subset)}"
            else:
                self.df = self.df.drop_duplicates()
                removed_count = original_count - len(self.df)
                message = f"Removed {removed_count} duplicate rows"
            
            # Record the operation in cleaning history
            self.cleaning_history.append({
                'operation': 'remove_duplicates',
                'subset': subset,
                'removed_count': removed_count
            })
            
            return True, message
            
        except Exception as e:
            return False, f"Error removing duplicates: {str(e)}"
    
    def standardize_text(self, column: str, case: str = 'lower', strip: bool = True) -> Tuple[bool, str]:
        """
        Standardize text in a column.
        
        Args:
            column: The name of the column to standardize
            case: The case to convert to ('lower', 'upper', 'title', 'none')
            strip: Whether to strip whitespace
            
        Returns:
            A tuple of (success, message)
        """
        if column not in self.df.columns:
            return False, f"Column '{column}' not found in the DataFrame"
        
        try:
            # Convert to string first to ensure text operations work
            self.df[column] = self.df[column].astype(str)
            
            if case == 'lower':
                self.df[column] = self.df[column].str.lower()
            elif case == 'upper':
                self.df[column] = self.df[column].str.upper()
            elif case == 'title':
                self.df[column] = self.df[column].str.title()
            elif case != 'none':
                return False, f"Unsupported case option: {case}"
            
            if strip:
                self.df[column] = self.df[column].str.strip()
            
            # Record the operation in cleaning history
            self.cleaning_history.append({
                'operation': 'standardize_text',
                'column': column,
                'case': case,
                'strip': strip
            })
            
            return True, f"Standardized text in column '{column}'"
            
        except Exception as e:
            return False, f"Error standardizing text in column '{column}': {str(e)}"
    
    def apply_custom_validation(self, column: str, condition: str) -> Dict[str, Any]:
        """
        Apply a custom validation condition to a column.
        
        Args:
            column: The name of the column to validate
            condition: A string representing a condition (e.g., "> 0", "== 'Valid'")
            
        Returns:
            A dictionary with validation results
        """
        if column not in self.df.columns:
            return {
                'success': False,
                'message': f"Column '{column}' not found in the DataFrame",
                'valid_count': 0,
                'invalid_count': 0,
                'invalid_indices': []
            }
        
        try:
            # Create a validation mask using eval
            validation_expr = f"self.df['{column}'] {condition}"
            mask = eval(validation_expr)
            
            valid_count = mask.sum()
            invalid_count = len(mask) - valid_count
            invalid_indices = self.df[~mask].index.tolist()
            
            return {
                'success': True,
                'message': f"Validation complete: {valid_count} valid, {invalid_count} invalid values",
                'valid_count': valid_count,
                'invalid_count': invalid_count,
                'invalid_indices': invalid_indices
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error applying validation: {str(e)}",
                'valid_count': 0,
                'invalid_count': 0,
                'invalid_indices': []
            }
    
    def remove_rows_by_indices(self, indices: List[int]) -> Tuple[bool, str]:
        """
        Remove rows by their indices.
        
        Args:
            indices: List of row indices to remove
            
        Returns:
            A tuple of (success, message)
        """
        try:
            original_count = len(self.df)
            self.df = self.df.drop(indices)
            removed_count = original_count - len(self.df)
            
            # Record the operation in cleaning history
            self.cleaning_history.append({
                'operation': 'remove_rows',
                'indices': indices,
                'removed_count': removed_count
            })
            
            return True, f"Removed {removed_count} rows"
            
        except Exception as e:
            return False, f"Error removing rows: {str(e)}"
    
    def reset_to_original(self) -> None:
        """Reset the DataFrame to its original state and clear cleaning history."""
        self.df = self.original_df.copy()
        self.cleaning_history = []
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Get the cleaned DataFrame."""
        return self.df.copy()
    
    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """Get the cleaning history."""
        return self.cleaning_history.copy()
