import pandas as pd

df = pd.read_csv('/Users/trdung/Documents/Project-MAI/final.csv')
print(df.count())
print(df[['rating']].describe())

number_of_users = df['userId'].nunique()
number_of_movies = df['id'].nunique()
print(f'Number of users:{number_of_users}')
print(f'Number_of_movies:{number_of_movies}')

number_of_ratings = df.groupby('id').agg({
    'userId':'count'
}).reset_index()

more_than_10_ratings = number_of_ratings[number_of_ratings['userId']>100]
print(more_than_10_ratings)

df = df[df['id'].isin(more_than_10_ratings['id'])]
print(df['id'].nunique())

df.to_csv('more_than_100_ratings.csv', index = False)