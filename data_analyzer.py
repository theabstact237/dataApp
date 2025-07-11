import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Union

class DataAnalyzer:
    """
    A class for analyzing data and generating insights from pandas DataFrames.
    Provides methods for missing data analysis, outlier detection, and data visualization.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a DataFrame.
        
        Args:
            df: The pandas DataFrame to analyze
        """
        self.df = df.copy()
    
    def analyze_missing_data(self) -> Dict[str, Any]:
        """
        Analyze missing data in the DataFrame.
        
        Returns:
            A dictionary with missing data analysis results
        """
        # Calculate missing values per column
        missing_count = self.df.isna().sum()
        missing_percentage = (missing_count / len(self.df) * 100).round(2)
        
        # Create a summary DataFrame
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percentage.values
        })
        
        # Sort by missing count in descending order
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        # Calculate overall statistics
        total_cells = self.df.size
        total_missing = missing_count.sum()
        overall_missing_percentage = (total_missing / total_cells * 100).round(2)
        
        # Identify columns with missing values
        columns_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        
        return {
            'missing_summary': missing_df,
            'total_cells': total_cells,
            'total_missing': total_missing,
            'overall_missing_percentage': overall_missing_percentage,
            'columns_with_missing': columns_with_missing
        }
    
    def create_missing_data_heatmap(self) -> Optional[plt.Figure]:
        """
        Create a heatmap visualization of missing data.
        
        Returns:
            A matplotlib Figure object or None if no missing data
        """
        # Check if there's any missing data
        if not self.df.isna().any().any():
            return None
        
        # Create a mask for missing values
        mask = self.df.isna()
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create the heatmap
        sns.heatmap(mask, cmap='viridis', cbar=False, ax=ax)
        
        # Set title and labels
        ax.set_title('Missing Data Heatmap')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        
        return fig
    
    def detect_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns of the DataFrame.
        
        Args:
            method: Method to use for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection (z-score or IQR multiplier)
            
        Returns:
            A dictionary with outlier detection results
        """
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            return {
                'outliers_found': False,
                'message': 'No numeric columns found in the dataset.'
            }
        
        outliers_by_column = {}
        
        for col in numeric_cols:
            # Skip columns with all missing values
            if self.df[col].isna().all():
                continue
            
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_indices = self.df[z_scores > threshold].index.tolist()
            elif method == 'iqr':
                # IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_indices = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index.tolist()
            else:
                return {
                    'outliers_found': False,
                    'message': f'Unsupported outlier detection method: {method}'
                }
            
            if outlier_indices:
                outliers_by_column[col] = {
                    'indices': outlier_indices,
                    'count': len(outlier_indices),
                    'percentage': round(len(outlier_indices) / len(self.df) * 100, 2)
                }
        
        # Summarize results
        total_outliers = sum(info['count'] for info in outliers_by_column.values())
        
        return {
            'outliers_found': total_outliers > 0,
            'method': method,
            'threshold': threshold,
            'total_outliers': total_outliers,
            'columns_with_outliers': list(outliers_by_column.keys()),
            'outliers_by_column': outliers_by_column
        }
    
    def create_boxplots(self, columns: List[str] = None) -> Optional[plt.Figure]:
        """
        Create boxplots for numeric columns to visualize distributions and outliers.
        
        Args:
            columns: List of columns to plot (defaults to all numeric columns)
            
        Returns:
            A matplotlib Figure object or None if no numeric columns
        """
        # Get numeric columns if not specified
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to only include numeric columns from the provided list
            columns = [col for col in columns if col in self.df.select_dtypes(include=np.number).columns]
        
        if not columns:
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
        
        # Handle case with only one column
        if len(columns) == 1:
            axes = [axes]
        
        # Create boxplots
        for i, col in enumerate(columns):
            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(col)
        
        plt.tight_layout()
        return fig
    
    def create_distribution_plots(self, columns: List[str] = None) -> Dict[str, plt.Figure]:
        """
        Create distribution plots (histograms and KDE) for numeric columns.
        
        Args:
            columns: List of columns to plot (defaults to all numeric columns)
            
        Returns:
            A dictionary mapping column names to matplotlib Figure objects
        """
        # Get numeric columns if not specified
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to only include numeric columns from the provided list
            columns = [col for col in columns if col in self.df.select_dtypes(include=np.number).columns]
        
        if not columns:
            return {}
        
        figures = {}
        
        for col in columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            figures[col] = fig
        
        return figures
    
    def create_correlation_heatmap(self) -> Optional[plt.Figure]:
        """
        Create a correlation heatmap for numeric columns.
        
        Returns:
            A matplotlib Figure object or None if fewer than 2 numeric columns
        """
        # Get numeric columns
        numeric_df = self.df.select_dtypes(include=np.number)
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        
        # Set title
        ax.set_title('Correlation Heatmap')
        
        return fig
    
    def create_scatter_matrix(self, columns: List[str] = None, max_cols: int = 4) -> Optional[plt.Figure]:
        """
        Create a scatter matrix for numeric columns.
        
        Args:
            columns: List of columns to include (defaults to all numeric columns)
            max_cols: Maximum number of columns to include
            
        Returns:
            A matplotlib Figure object or None if fewer than 2 numeric columns
        """
        # Get numeric columns if not specified
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to only include numeric columns from the provided list
            columns = [col for col in columns if col in self.df.select_dtypes(include=np.number).columns]
        
        # Limit to max_cols
        if len(columns) > max_cols:
            columns = columns[:max_cols]
        
        if len(columns) < 2:
            return None
        
        # Create scatter matrix
        fig = sns.pairplot(self.df[columns])
        fig.fig.suptitle('Scatter Matrix', y=1.02)
        
        return fig.fig
    
    def create_bar_chart(self, x_col: str, y_col: str = None, agg_func: str = 'count') -> Optional[plt.Figure]:
        """
        Create a bar chart for categorical data.
        
        Args:
            x_col: Column to use for x-axis (categories)
            y_col: Column to use for y-axis (values, optional)
            agg_func: Aggregation function to use ('count', 'sum', 'mean', 'median')
            
        Returns:
            A matplotlib Figure object or None if column not found
        """
        if x_col not in self.df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if y_col is None or y_col not in self.df.columns:
            # Simple count plot
            sns.countplot(x=x_col, data=self.df, ax=ax)
            ax.set_title(f'Count of {x_col}')
        else:
            # Aggregated bar plot
            if agg_func == 'count':
                data = self.df.groupby(x_col)[y_col].count()
            elif agg_func == 'sum':
                data = self.df.groupby(x_col)[y_col].sum()
            elif agg_func == 'mean':
                data = self.df.groupby(x_col)[y_col].mean()
            elif agg_func == 'median':
                data = self.df.groupby(x_col)[y_col].median()
            else:
                return None
            
            data.plot(kind='bar', ax=ax)
            ax.set_title(f'{agg_func.capitalize()} of {y_col} by {x_col}')
        
        ax.set_xlabel(x_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_time_series_plot(self, date_col: str, value_col: str, freq: str = None) -> Optional[plt.Figure]:
        """
        Create a time series plot.
        
        Args:
            date_col: Column with date/time data
            value_col: Column with values to plot
            freq: Frequency for resampling (e.g., 'D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            A matplotlib Figure object or None if columns not found
        """
        if date_col not in self.df.columns or value_col not in self.df.columns:
            return None
        
        # Try to convert to datetime if not already
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                date_series = pd.to_datetime(self.df[date_col])
            else:
                date_series = self.df[date_col]
        except:
            return None
        
        # Create a copy of the data for plotting
        plot_df = pd.DataFrame({
            'date': date_series,
            'value': self.df[value_col]
        }).dropna()
        
        # Set date as index
        plot_df = plot_df.set_index('date')
        
        # Resample if frequency is specified
        if freq is not None:
            plot_df = plot_df.resample(freq).mean()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        plot_df['value'].plot(ax=ax)
        
        # Set title and labels
        ax.set_title(f'Time Series of {value_col}')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        
        plt.tight_layout()
        return fig
    
    def get_summary_statistics(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Args:
            columns: List of columns to include (defaults to all numeric columns)
            
        Returns:
            A DataFrame with summary statistics
        """
        # Get numeric columns if not specified
        if columns is None:
            numeric_df = self.df.select_dtypes(include=np.number)
        else:
            # Filter to only include numeric columns from the provided list
            numeric_df = self.df[columns].select_dtypes(include=np.number)
        
        # Calculate summary statistics
        summary = numeric_df.describe().T
        
        # Add additional statistics
        summary['median'] = numeric_df.median()
        summary['skew'] = numeric_df.skew()
        summary['kurtosis'] = numeric_df.kurtosis()
        summary['missing'] = numeric_df.isna().sum()
        summary['missing_pct'] = (numeric_df.isna().sum() / len(numeric_df) * 100).round(2)
        
        # Reset index to make column names a regular column
        summary = summary.reset_index().rename(columns={'index': 'column'})
        
        return summary
    
    def get_categorical_summary(self, columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for categorical columns.
        
        Args:
            columns: List of columns to include (defaults to all object and category columns)
            
        Returns:
            A dictionary mapping column names to summary DataFrames
        """
        # Get categorical columns if not specified
        if columns is None:
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            # Filter to only include categorical columns from the provided list
            cat_cols = [col for col in columns if col in self.df.select_dtypes(include=['object', 'category']).columns]
        
        if not cat_cols:
            return {}
        
        summaries = {}
        
        for col in cat_cols:
            # Calculate value counts and percentages
            value_counts = self.df[col].value_counts()
            percentages = (value_counts / len(self.df) * 100).round(2)
            
            # Create summary DataFrame
            summary = pd.DataFrame({
                'value': value_counts.index,
                'count': value_counts.values,
                'percentage': percentages.values
            })
            
            summaries[col] = summary
        
        return summaries
