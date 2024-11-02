import pandas as pd

df = pd.read_csv('/Users/trdung/Documents/Project-MAI/cleaned_data.csv')

print(df.count())

# Calculate Q1 and Q3
Q1 = df['rating'].quantile(0.25)
Q3 = df['rating'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['rating'] < lower_bound) | (df['rating'] > upper_bound)]
print(outliers.count())

df = df[(df['rating'] >= lower_bound) & (df['rating'] <= upper_bound)]
print(df.count())

df.to_csv('removed_ouliers.csv', index = False)