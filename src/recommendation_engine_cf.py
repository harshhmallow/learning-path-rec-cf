import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def train_svd_model(user_item_matrix):
    
    svd = TruncatedSVD(n_components=4, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    course_factors = svd.components_
    
    return svd, user_factors, course_factors #tuple containing the SVD model, user factors, and course factors

def get_top_n_recommendations(user_id, user_item_matrix, user_factors, course_factors, n=5):
    
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    predicted_ratings = np.dot(user_factors[user_idx], course_factors) #predict using matrix multiplication
    
    all_course_ids = user_item_matrix.columns
    recommended_courses = pd.Series(predicted_ratings, index=all_course_ids).sort_values(ascending=False)
    
    user_interacted_courses = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    filtered_recommendations = recommended_courses[~recommended_courses.index.isin(user_interacted_courses)] #filter out courses the user has already interacted with
    
    return filtered_recommendations.head(n).index.tolist()