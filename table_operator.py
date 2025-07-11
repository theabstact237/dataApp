import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

class TableOperator:
    """
    A class for performing table operations on pandas DataFrames.
    Provides methods for pivot tables, aggregations, and conditional operations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TableOperator with a DataFrame.
        
        Args:
            df: The pandas DataFrame to operate on
        """
        self.df = df.copy()
    
    def create_pivot_table(self, index: List[str], columns: Optional[List[str]], 
                          values: List[str], aggfunc: str) -> Tuple[bool, pd.DataFrame, str]:
        """
        Create a pivot table from the DataFrame.
        
        Args:
            index: Columns to use as index
            columns: Columns to use as columns (optional)
            values: Columns to aggregate
            aggfunc: Aggregation function ('mean', 'sum', 'count', 'min', 'max')
            
        Returns:
            A tuple of (success, pivot_table, message)
        """
        try:
            # Validate inputs
            for col in index + (columns or []) + values:
                if col not in self.df.columns:
                    return False, None, f"Column '{col}' not found in the DataFrame"
            
            # Map string aggfunc to actual function
            if aggfunc == 'mean':
                agg_function = np.mean
            elif aggfunc == 'sum':
                agg_function = np.sum
            elif aggfunc == 'count':
                agg_function = 'count'
            elif aggfunc == 'min':
                agg_function = np.min
            elif aggfunc == 'max':
                agg_function = np.max
            else:
                return False, None, f"Unsupported aggregation function: {aggfunc}"
            
            # Create pivot table
            pivot_table = pd.pivot_table(
                self.df,
                index=index,
                columns=columns,
                values=values,
                aggfunc=agg_function
            )
            
            return True, pivot_table, "Pivot table created successfully"
            
        except Exception as e:
            return False, None, f"Error creating pivot table: {str(e)}"
    
    def aggregate_data(self, group_by: List[str], agg_columns: Dict[str, List[str]]) -> Tuple[bool, pd.DataFrame, str]:
        """
        Aggregate data by grouping and applying aggregation functions.
        
        Args:
            group_by: Columns to group by
            agg_columns: Dictionary mapping columns to aggregation functions
            
        Returns:
            A tuple of (success, aggregated_data, message)
        """
        try:
            # Validate inputs
            for col in group_by:
                if col not in self.df.columns:
                    return False, None, f"Column '{col}' not found in the DataFrame"
            
            for col, funcs in agg_columns.items():
                if col not in self.df.columns:
                    return False, None, f"Column '{col}' not found in the DataFrame"
                for func in funcs:
                    if func not in ['mean', 'sum', 'count', 'min', 'max', 'std', 'var', 'median']:
                        return False, None, f"Unsupported aggregation function: {func}"
            
            # Create aggregation dictionary
            agg_dict = {}
            for col, funcs in agg_columns.items():
                agg_dict[col] = funcs
            
            # Perform aggregation
            aggregated_data = self.df.groupby(group_by).agg(agg_dict)
            
            return True, aggregated_data, "Data aggregated successfully"
            
        except Exception as e:
            return False, None, f"Error aggregating data: {str(e)}"
    
    def conditional_count(self, column: str, condition: str) -> Tuple[bool, int, str]:
        """
        Count rows that meet a specific condition.
        
        Args:
            column: Column to apply condition to
            condition: Condition string (e.g., "> 0", "== 'Value'")
            
        Returns:
            A tuple of (success, count, message)
        """
        try:
            if column not in self.df.columns:
                return False, 0, f"Column '{column}' not found in the DataFrame"
            
            # Create a validation mask using eval
            validation_expr = f"self.df['{column}'] {condition}"
            mask = eval(validation_expr)
            
            count = mask.sum()
            
            return True, count, f"Found {count} rows where {column} {condition}"
            
        except Exception as e:
            return False, 0, f"Error performing conditional count: {str(e)}"
    
    def conditional_aggregate(self, column: str, condition: str, agg_column: str, 
                             agg_func: str) -> Tuple[bool, float, str]:
        """
        Aggregate values for rows that meet a specific condition.
        
        Args:
            column: Column to apply condition to
            condition: Condition string (e.g., "> 0", "== 'Value'")
            agg_column: Column to aggregate
            agg_func: Aggregation function ('mean', 'sum', 'min', 'max', 'median')
            
        Returns:
            A tuple of (success, result, message)
        """
        try:
            if column not in self.df.columns or agg_column not in self.df.columns:
                missing_col = column if column not in self.df.columns else agg_column
                return False, 0, f"Column '{missing_col}' not found in the DataFrame"
            
            # Create a validation mask using eval
            validation_expr = f"self.df['{column}'] {condition}"
            mask = eval(validation_expr)
            
            # Filter data
            filtered_data = self.df[mask][agg_column]
            
            # Apply aggregation
            if agg_func == 'mean':
                result = filtered_data.mean()
            elif agg_func == 'sum':
                result = filtered_data.sum()
            elif agg_func == 'min':
                result = filtered_data.min()
            elif agg_func == 'max':
                result = filtered_data.max()
            elif agg_func == 'median':
                result = filtered_data.median()
            else:
                return False, 0, f"Unsupported aggregation function: {agg_func}"
            
            return True, result, f"{agg_func.capitalize()} of {agg_column} where {column} {condition}: {result}"
            
        except Exception as e:
            return False, 0, f"Error performing conditional aggregation: {str(e)}"
    
    def create_crosstab(self, row: str, column: str, 
                       values: Optional[str] = None, 
                       aggfunc: str = 'count') -> Tuple[bool, pd.DataFrame, str]:
        """
        Create a cross-tabulation (contingency table) of two columns.
        
        Args:
            row: Column to use for rows
            column: Column to use for columns
            values: Column to aggregate (optional, for values other than counts)
            aggfunc: Aggregation function ('mean', 'sum', 'count', 'min', 'max')
            
        Returns:
            A tuple of (success, crosstab, message)
        """
        try:
            if row not in self.df.columns or column not in self.df.columns:
                missing_col = row if row not in self.df.columns else column
                return False, None, f"Column '{missing_col}' not found in the DataFrame"
            
            if values is not None and values not in self.df.columns:
                return False, None, f"Column '{values}' not found in the DataFrame"
            
            # Map string aggfunc to actual function
            if aggfunc == 'mean':
                agg_function = np.mean
            elif aggfunc == 'sum':
                agg_function = np.sum
            elif aggfunc == 'count':
                agg_function = len
            elif aggfunc == 'min':
                agg_function = np.min
            elif aggfunc == 'max':
                agg_function = np.max
            else:
                return False, None, f"Unsupported aggregation function: {aggfunc}"
            
            # Create crosstab
            if values is None:
                crosstab = pd.crosstab(self.df[row], self.df[column])
            else:
                crosstab = pd.crosstab(
                    self.df[row], 
                    self.df[column], 
                    values=self.df[values], 
                    aggfunc=agg_function
                )
            
            return True, crosstab, "Cross-tabulation created successfully"
            
        except Exception as e:
            return False, None, f"Error creating cross-tabulation: {str(e)}"
    
    def create_heatmap(self, pivot_table: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Create a heatmap visualization of a pivot table.
        
        Args:
            pivot_table: Pivot table DataFrame
            
        Returns:
            A matplotlib Figure object or None if error
        """
        try:
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f', ax=ax)
            
            # Set title
            ax.set_title('Pivot Table Heatmap')
            
            return fig
            
        except Exception:
            return None
    
    def create_bar_chart(self, data: pd.DataFrame, title: str = "Bar Chart") -> Optional[plt.Figure]:
        """
        Create a bar chart visualization of aggregated data.
        
        Args:
            data: Aggregated DataFrame
            title: Chart title
            
        Returns:
            A matplotlib Figure object or None if error
        """
        try:
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bar chart
            data.plot(kind='bar', ax=ax)
            
            # Set title and labels
            ax.set_title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception:
            return None
