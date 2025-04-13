import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

netflix = pd.read_csv(r"C:\Users\shiva\Downloads\netflix_titles.csv")  

netflix['date_added'] = pd.to_datetime(netflix['date_added'].str.strip(), errors='coerce')

netflix['year_added'] = netflix['date_added'].dt.year
netflix['month_added'] = netflix['date_added'].dt.month_name()

print("=== Dataset Overview ===")
print(netflix.info())
print("\nMissing Values:\n", netflix.isnull().sum())

netflix['country'] = netflix['country'].fillna('Unknown')
netflix['rating'] = netflix['rating'].fillna('Not Rated')
netflix['director'] = netflix['director'].fillna('Unknown')

netflix['primary_genre'] = netflix['listed_in'].str.split(',').str[0]
#Content Added Over Time 
plt.figure(figsize=(12, 6))
trend = netflix.groupby(['year_added', 'type']).size().unstack()
trend.plot(kind='line', marker='o', color=['#E50914', '#221F1F'], linewidth=2.5)
plt.title("Netflix Content Added Over Time (Movies vs. TV Shows)")
plt.xlabel("Year")
plt.ylabel("Number of Titles Added")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Type')
plt.show()
#Top Genres (Like Product Categories)
plt.figure(figsize=(12, 6))
top_genres = netflix['primary_genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="magma", legend=False)
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()
#Ratings Distribution (Like Gender Analysis)
plt.figure(figsize=(10, 6))
ratings_order = ['G', 'PG', 'PG-13', 'R', 'TV-14', 'TV-MA', 'NC-17']
sns.countplot(data=netflix, y='rating', hue='rating', order=ratings_order, palette="viridis", legend=False)
plt.title("Distribution of Content Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()
#Content by Country (Geographical Analysis)
plt.figure(figsize=(12, 6))
top_countries = netflix['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette="rocket")
plt.title("Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()
# Movie Durations (Like Price Distribution) 
movies = netflix[netflix['type'] == 'Movie']
movies['duration_min'] = (
    movies['duration']
    .str.extract('(\d+)')  
    .astype(float)         
    .fillna(0)              
    .astype(int)           
)
movies = movies[movies['duration_min'] > 0]

plt.figure(figsize=(10, 6))
sns.histplot(movies['duration_min'], bins=20, kde=True, color='#E50914')
plt.title("Distribution of Movie Durations (Minutes)")
plt.xlabel("Duration (min)")
plt.ylabel("Number of Movies")
plt.show()

#Movie Duration by Rating (Box Plot)
# Convert duration to minutes (for movies only)
movies = netflix[netflix['type'] == 'Movie'].copy()
movies['duration_min'] = movies['duration'].str.extract('(\d+)').astype(float)

# Box plot: Duration by Rating
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=movies, 
    x='rating', 
    y='duration_min',
    order=['G', 'PG', 'PG-13', 'R', 'TV-14', 'TV-MA'],
    palette="Set2"
)
plt.title("Movie Duration Distribution by Rating")
plt.xlabel("Rating")
plt.ylabel("Duration (minutes)")
plt.xticks(rotation=45)
plt.show()

# Box plot: Release year for Movies vs. TV Shows
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=netflix.dropna(subset=['release_year']), 
    x='type', 
    y='release_year',
    palette=['#E50914', '#221F1F']
)
plt.title("Distribution of Release Years by Content Type")
plt.xlabel("Type")
plt.ylabel("Release Year")
plt.show()
#. Movies vs. TV Shows (Pie Chart)
type_counts = netflix['type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(
    type_counts, 
    labels=type_counts.index, 
    autopct='%1.1f%%',
    colors=['#E50914', '#221F1F'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)
plt.title("Netflix Content: Movies vs. TV Shows")
plt.show()

# Pie chart: Top 5 Genres 
plt.figure(figsize=(8, 8))
top_genres = netflix['primary_genre'].value_counts().head(5)
plt.pie(
    top_genres,
    labels=top_genres.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'} 
)
plt.title("Top 5 Genres on Netflix", pad=20)
plt.tight_layout()  
plt.show()

#heatmap Genre Popularity by Country
#(Purpose: Identify which genres dominate in specific countries)

top_countries = netflix['country'].value_counts().head(5).index.tolist()
top_genres = netflix['primary_genre'].value_counts().head(5).index.tolist()
filtered_data = netflix[
    (netflix['country'].isin(top_countries)) & 
    (netflix['primary_genre'].isin(top_genres))
]

cross_tab = pd.crosstab(
    index=filtered_data['country'], 
    columns=filtered_data['primary_genre']
)

plt.figure(figsize=(10, 6))
sns.heatmap(
    cross_tab, 
    cmap="Blues", 
    annot=True, 
    fmt="d", 
    linewidths=0.5,
    cbar_kws={'label': 'Number of Titles'}
)
plt.title("Genre Popularity by Country (Top 5 Each)")
plt.xlabel("Genre")
plt.ylabel("Country")
plt.xticks(rotation=45)
plt.show()

