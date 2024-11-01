import numpy as np
import pandas as pd

# Step 1: Sample User-Item Matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Convert to DataFrame for easier handling
ratings_df = pd.DataFrame(ratings, columns=['Toy Story', 'StarWar', 'WarCraft', 'UP'])
ratings_df.index.name = 'User'
print("User-Item Matrix:\n", ratings_df)

# Step 2: Apply SVD
U, sigma, Vt = np.linalg.svd(ratings, full_matrices=False)
sigma = np.diag(sigma)  # Convert sigma into a diagonal matrix

# Reduce dimensions (selecting top k=2 for simplicity)
k = 2
U_k = U[:, :k]
sigma_k = sigma[:k, :k]
Vt_k = Vt[:k, :]

# Print the reduced matrices
print("\nReduced Matrices:")
print("U_k:\n", U_k)
print("Sigma_k:\n", sigma_k)
print("Vt_k:\n", Vt_k)

# Step 3: Reconstruct the Approximate User-Item Matrix
predicted_ratings = np.dot(np.dot(U_k, sigma_k), Vt_k)

# Convert to DataFrame for easier visualization
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=ratings_df.columns)
predicted_ratings_df.index.name = 'User'
print("\nPredicted Ratings:\n", predicted_ratings_df)


# Step 4: Generate Movie Recommendations
def recommend_movies(user_id, ratings, predicted_ratings, num_recommendations=2):
    # Get movies the user has already rated
    user_ratings = ratings[user_id, :]
    rated_movie_indices = np.where(user_ratings > 0)[0]

    # Filter out rated movies from predictions
    predictions = predicted_ratings[user_id, :]
    predictions[rated_movie_indices] = -np.inf  # Set rated movies to -inf to exclude

    # Get indices of the top N recommended movies
    recommended_movie_indices = np.argsort(predictions)[-num_recommendations:][::-1]
    recommended_movies = [ratings_df.columns[i] for i in recommended_movie_indices]

    return recommended_movies


# Display top recommendations for each user
print("\nTop Movie Recommendations:")
for user_id in range(ratings.shape[0]):
    recommended_movies = recommend_movies(user_id, ratings, predicted_ratings)
    print(f"User {user_id + 1}: {recommended_movies}")
