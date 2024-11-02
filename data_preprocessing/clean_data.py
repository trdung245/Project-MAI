import pandas as pd

movie_df = pd.read_csv('/Users/trdung/Documents/Project-MAI/moviesinfo.csv')
rating_df = pd.read_csv('/Users/trdung/Documents/Project-MAI/ratings.csv')

# Remove rows with NaN values in the 'id' column
movie_df = movie_df.dropna(subset=['id'])
rating_df = rating_df.dropna(subset=['id'])

rating_df['id'] = rating_df['id'].astype(str)
# Now, use str.isnumeric() to filter numeric 'id' values
movie_df = movie_df[movie_df['id'].str.isnumeric()]
rating_df = rating_df[rating_df['id'].str.isnumeric()]

movie_df['id'] = movie_df['id'].astype(int)
rating_df['id'] = rating_df['id'].astype(int)

merged_df = movie_df.merge(rating_df, how = 'inner', on = 'id')
merged_df = merged_df[['id', 'original_title', 'userId', 'rating']]
merged_df.to_csv('cleaned_data.csv', index = False)