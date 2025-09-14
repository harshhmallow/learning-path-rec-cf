from src.data_processing import load_data, create_user_item_matrix
from src.recommendation_engine_cf import train_svd_model, get_top_n_recommendations
import pandas as pd

def main():
    
    print("Starting Collaborative Filtering Recommendation System.")
    
    user_activity_file = 'data/user_activity.csv'
    course_details_file = 'data/course_details.csv'
    
    raw_df = load_data(user_activity_file, course_details_file)
    if raw_df is None:
        return
    
    user_item_matrix = create_user_item_matrix(raw_df)
    
    svd_model, user_factors, course_factors = train_svd_model(user_item_matrix)
    print("SVD model trained successfully.")
    
    course_details_df = pd.read_csv(course_details_file)
    user_ids = user_item_matrix.index
    
    print("\n- Generating Recommendations for All Users -")
    for user_id in user_ids:
        top_recommendations = get_top_n_recommendations(
            user_id, user_item_matrix, user_factors, course_factors, n=3
        )
        
        recommended_course_names = course_details_df[
            course_details_df['course_id'].isin(top_recommendations)
        ]['course_name'].tolist()
        
        print(f"User {user_id}: Recommended Courses - {', '.join(recommended_course_names)}")
    
    print("\n- Recommendation Process Complete -")

if __name__ == "__main__":
    main()