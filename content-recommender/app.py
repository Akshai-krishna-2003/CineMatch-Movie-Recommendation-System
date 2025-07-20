from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import pickle
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Flask setup
app = Flask(__name__)
app.secret_key = 'A1b2C3d4E5f6G7h8I9j0K_LmNoPqRsTuVwXyZ'

# Load your datasets (adjust the paths if needed)
movies_df = pd.read_csv('models/movies_with_combined_content.csv')  
ratings_df = pd.read_csv('models/rating.csv')  
tags_df = pd.read_csv('models/tag.csv')

# Load models and data
movies_df = pd.read_csv('models/movies_with_combined_content.csv')
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('models/svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

# Recalculate cosine similarity (for TF-IDF model)
tfidf_matrix = tfidf_vectorizer.transform(movies_df['combined_content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Database setup (for user management)
DATABASE = 'users.db'

# Initialize the database (for user management)
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT NOT NULL UNIQUE,
                            password TEXT NOT NULL);''')

# Fetch user by username
def get_user_by_username(username):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        return cursor.fetchone()

# Add new user to the database
def add_user(username, password):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, generate_password_hash(password)))
        conn.commit()


def get_recommendations(input_text, tfidf_vectorizer, tfidf_matrix):
    # Process input text
    processed_text = input_text.lower().strip()
    
    # Transform input text using the TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([processed_text])
    
    # Calculate similarity scores against all movies
    cosine_sim_scores = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get top 5 movie indices
    sim_scores = list(enumerate(cosine_sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:5]]

    # Get full movie data for top indices
    return movies_df.iloc[top_indices].to_dict('records')

# Hybrid model combining collaborative filtering and content-based filtering
def hybrid_recommendations(user_id, movie_title):
    content_recs = get_recommendations(movie_title)
    unseen_movies = movies_df[~movies_df['movieId'].isin(
        ratings_df[ratings_df['userId'] == user_id]['movieId'])]
    collab_recs = []
    for movie in unseen_movies['movieId'].head(10000):  
        pred = svd.predict(user_id, movie)
        collab_recs.append((movie, pred.est))
    top_collab = sorted(collab_recs, key=lambda x: x[1], reverse=True)[:5]
    top_titles = movies_df[movies_df['movieId'].isin([x[0] for x in top_collab])]['title'].tolist()
    return list(set(content_recs + top_titles))[:5]


movie_posters = [
    {
        "title": "The Dark Knight",
        "poster": "https://m.media-amazon.com/images/I/51CbCQNMyiL._AC_.jpg",
        "rating": 4.8,
        "year": 2008
    },
    {
        "title": "Inception",
        "poster": "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg",
        "rating": 4.7,
        "year": 2010
    },
    {
        "title": "The Matrix",
        "poster": "https://m.media-amazon.com/images/M/MV5BN2NmN2VhMTQtMDNiOS00NDlhLTliMjgtODE2ZTY0ODQyNDRhXkEyXkFqcGc@._V1_.jpg",
        "rating": 4.9,
        "year": 1999
    },
    {
        "title": "Fight Club",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFOCd9JPgVLsH7WrjhXxLOb3bhTve666VlAA&s",
        "rating": 4.8,
        "year": 1999
    },
    {
        "title": "Forrest Gump",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRN2PvDX_v4iGJ83u5cHx6W4Sc8tZ7CEAd8ZQ&s",
        "rating": 4.9,
        "year": 1994
    },
    {
        "title": "The Godfather",
        "poster": "https://m.media-amazon.com/images/M/MV5BNGEwYjgwOGQtYjg5ZS00Njc1LTk2ZGEtM2QwZWQ2NjdhZTE5XkEyXkFqcGc@._V1_.jpg",
        "rating": 4.9,
        "year": 1972
    },
    {
        "title": "The Shawshank Redemption",
        "poster": "https://images.moviesanywhere.com/53dd4d73ac5d1dacd2e577550023dab5/429f797f-c4ca-4d27-8fc5-ca552a5d86e7.jpg",
        "rating": 4.9,
        "year": 1994
    },
    {
        "title": "Pulp Fiction",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQeJ5ACuYR0ZriBayXnBIjYUlUIYB1jtpyf3g&s",
        "rating": 4.9,
        "year": 1994
    },
    {
        "title": "Interstellar",
        "poster": "https://play-lh.googleusercontent.com/oVJrPE3AI_X4bTVCKhbzY_2Bekogch9HEfWQ_vCyrS49enzXLZCUIuCTZp-YPT2tcUtjAnXZAXh3WetkHO8",
        "rating": 4.7,
        "year": 2014
    },
    {
        "title": "The Lion King",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRBFpuRxc8jDV90sKBFWNpwSp4ZxFjWCs0XQ&s",
        "rating": 4.8,
        "year": 1994
    },
    {
        "title": "Star Wars: Episode IV - A New Hope",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHMFXaQEwhjUU7ZLposnXXB9sS6Z_6EFhjxg&s",
        "rating": 4.7,
        "year": 1977
    },
    {
        "title": "Back to the Future",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ58pqv8KnfBIAYsCc2Ok3GAj0nq6NeB8IXlw&s",
        "rating": 4.8,
        "year": 1985
    },
    {
        "title": "Goodfellas",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQA4lre3kvvSDkWq0yifXF8NePLgG_5KUI1AQ&s",
        "rating": 4.8,
        "year": 1990
    },
    {
        "title": "The Silence of the Lambs",
        "poster": "https://m.media-amazon.com/images/M/MV5BZTk5NTYzMGEtMDE2OS00ODYxLWJiNjQtNGMyMmM2MTE0M2QxXkEyXkFqcGc@._V1_QL75_UX160_.jpg",
        "rating": 4.8,
        "year": 1991
    },
    {
        "title": "Gladiator",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQC44FWT8AYQVMaRqPbsM1gb7u5ZQTpds3ylA&s",
        "rating": 4.8,
        "year": 2000
    },
    {
        "title": "The Prestige",
        "poster": "https://m.media-amazon.com/images/I/81AdI6L6nAL._AC_UF1000,1000_QL80_.jpg",
        "rating": 4.8,
        "year": 2006
    },
    {
        "title": "The Departed",
        "poster": "https://m.media-amazon.com/images/I/91XLwl9kE8L._AC_UF1000,1000_QL80_.jpg",
        "rating": 4.8,
        "year": 2006
    },
    {
        "title": "Jurassic Park",
        "poster": "https://m.media-amazon.com/images/I/917hANc0Q9L._AC_UF894,1000_QL80_.jpg",
        "rating": 4.7,
        "year": 1993
    },
    {
        "title": "The Green Mile",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4RtL7nwFJxJ996nwOv8IeBeB5XtQSwx18hQ&s",
        "rating": 4.8,
        "year": 1999
    },
    {
        "title": "The Truman Show",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQU0nFDCChC1KBEut1OOxBQzMk-V36vw8dqyw&s",
        "rating": 4.7,
        "year": 1998
    }
]

# Routes
@app.route('/')
def index():
    if not session.get('username'):
        return redirect(url_for('login'))
    
    # Shuffle movies to show different ones on each visit
    import random
    random.shuffle(movie_posters)
    
    return render_template('index.html', 
                         trending_movies=movie_posters[:20])

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_username(username)

        if user and check_password_hash(user[2], password):  # Check if password matches
            session['user_id'] = user[0]
            session['username'] = username
            flash('Login successful', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

# Sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username already exists
        if get_user_by_username(username):
            flash('Username already exists', 'danger')
        else:
            add_user(username, password)
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if not session.get('username'):
        return redirect(url_for('login'))
    
    recommendations = []
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['movie_title']
        recommendations = get_recommendations(input_text, tfidf_vectorizer, tfidf_matrix)

    return render_template(
        'recommend.html',  # Changed from index.html to recommend.html
        recommendations=recommendations,
        input_text=input_text
    )


if __name__ == '__main__':
    init_db()  # Ensure DB is created
    app.run(debug=True)
