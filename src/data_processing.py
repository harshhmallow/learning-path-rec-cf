import pandas as pd
import numpy as np

def load_data(user_activity_path, course_details_path):
    """Loads and merges user activity and course details data."""
    try:
        user_activity_df = pd.read_csv(user_activity_path)
        course_details_df = pd.read_csv(course_details_path)
        
        # Merge dataframes on course_id for a complete view
        df = pd.merge(user_activity_df, course_details_df, on='course_id')
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check file paths.")
        return None

def create_user_item_matrix(df):
    """
    Creates a user-item matrix from the interaction data.
    
    The matrix will have users as rows, courses as columns, and
    numerical ratings based on interaction type.
    """
    rating_map = {'view': 1, 'enroll': 3, 'complete': 5}
    df['rating'] = df['interaction_type'].map(rating_map)
    
    user_item_matrix = df.pivot_table(index='user_id', columns='course_id', values='rating').fillna(0)
    return user_item_matrix