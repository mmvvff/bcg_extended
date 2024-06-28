import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#### Function 3. Feature engineer

# Function to calculate the correlation coefficients for the lower off-diagonal elements
def correlation_matrix(df:pd.DataFrame, numeric_cols:list) -> pd.DataFrame:
    """
    Calculates the correlation matrix for a subset of numeric columns in a DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        cols (list): A list of column names for which to calculate correlations.
    Returns:
        pandas.DataFrame: A DataFrame containing the correlation coefficients.
    """
    # Create a correlation matrix for the numeric columns
    corr_matrix = df[numeric_cols].corr()
    # # Extract column names for the upper triangle of the correlation matrix (excluding redundant correlations)
    upper_tri_cols = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if col1 > col2]
    # Create a DataFrame to store correlation coefficients
    corr_df = pd.DataFrame(columns=["name_col1", "name_col2", "correlation"])
    # Fill the DataFrame with correlation coefficients and column names
    for i, (col1, col2) in enumerate(upper_tri_cols):
        corr_df.loc[i] = [col1, col2, corr_matrix.loc[col1, col2]]
    corr_df = corr_df.loc[corr_df["correlation"] < 1].copy()
    return corr_df

# Function that reduces a set of numerical columns to its most important PC
def extract_principal_components(df, cols):
    """
    Extracts principal components from a DataFrame using selected columns.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        cols (list): A list of column names to use for PCA.
    Returns:
        pandas.DataFrame: A DataFrame containing the principal components.
    """
    
    # Scale the data
    scaled_data = scale(df[cols])
    # Create a PCA object
    pca = PCA()
    # Fit the PCA model to the scaled data
    pca.fit(scaled_data)
    # Transform the scaled data to principal components
    principal_components = pca.transform(scaled_data)
    # Create a DataFrame to store the principal components
    pc_df = pd.DataFrame(principal_components, columns=[f"pc_{i+1}" for i in range(len(cols))])
    # choose number of components that explain 95% of variance 
    pc_explained_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(pc_explained_var >= 0.95) + 1
    # Reduce the number of components to those that explain 95% of the variance
    pc_df = pc_df.iloc[:, :n_components_95]
    return pc_df
