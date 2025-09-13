import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def train_svd_model(user_item_matrix):
    """
    Trains a TruncatedSVD model for collaborative filtering.
    
    Args:
        user_item_matrix (pd.DataFrame): The user-item interaction matrix.
        
    Returns:
        tuple: A tuple containing the SVD model, user factors, and course factors.
    """
    svd = TruncatedSVD(n_components=4, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    course_factors = svd.components_
    
    return svd, user_factors, course_factors

def get_top_n_recommendations(user_id, user_item_matrix, user_factors, course_factors, n=5):
    """
    Generates top N recommendations for a given user based on the SVD model.
    """
    # Get the index of the user in the matrix
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Predict ratings by matrix multiplication
    predicted_ratings = np.dot(user_factors[user_idx], course_factors)
    
    # Get all course IDs and sort them by predicted rating
    all_course_ids = user_item_matrix.columns
    recommended_courses = pd.Series(predicted_ratings, index=all_course_ids).sort_values(ascending=False)
    
    # Filter out courses the user has already interacted with
    user_interacted_courses = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    filtered_recommendations = recommended_courses[~recommended_courses.index.isin(user_interacted_courses)]
    
    return filtered_recommendations.head(n).index.tolist()