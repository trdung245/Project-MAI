import numpy as np
import pandas as pd

# Step 1: Load the Data
df = pd.read_csv('/Users/trdung/Documents/Project-MAI/more_than_100_ratings.csv')

# Step 2: Pivot to create the user-item matrix
matrix = df.pivot_table(index='id', columns='userId', values='rating', aggfunc='mean')
matrix = matrix.fillna(0)

# Convert the matrix to a NumPy array for SVD
ratings = matrix.values

# Step 3: Apply SVD
U, sigma, Vt = np.linalg.svd(ratings, full_matrices=False)
sigma = np.diag(sigma)  # Convert sigma into a diagonal matrix

# Reduce dimensions (selecting top k for simplicity, adjust k as needed)
k = 2  # You can choose a larger k for more accuracy
U_k = U[:, :k]
sigma_k = sigma[:k, :k]
Vt_k = Vt[:k, :]

# Step 4: Reconstruct the Approximate User-Item Matrix
predicted_ratings = np.dot(np.dot(U_k, sigma_k), Vt_k)

# Convert to DataFrame for easier visualization
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=matrix.columns, index=matrix.index)
predicted_ratings_df.index.name = 'Movie ID'

# Step 5: Generate Movie Recommendations
def recommend_movies(user_id, ratings, predicted_ratings, num_recommendations=2):
    # Get movies the user has already rated
    user_ratings = ratings[:, user_id]
    rated_movie_indices = np.where(user_ratings > 0)[0]

    # Filter out rated movies from predictions
    predictions = predicted_ratings[:, user_id]
    predictions[rated_movie_indices] = -np.inf  # Set rated movies to -inf to exclude

    # Get indices of the top N recommended movies
    recommended_movie_indices = np.argsort(predictions)[-num_recommendations:][::-1]
    recommended_movies = [matrix.index[i] for i in recommended_movie_indices]

    return recommended_movies

# Step 6: Display top recommendations for each user
print("\nTop Movie Recommendations:")
for user_id in range(ratings.shape[1]):  # Iterate over users (columns)
    recommended_movies = recommend_movies(user_id, ratings, predicted_ratings)
    movies = df['id'].isin(recommend_movies)['original_title']
    print(f"User {user_id + 1}: {movies}")
    if user_id == 10:
        break