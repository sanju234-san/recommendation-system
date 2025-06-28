import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


set = {
    'title': [
        'Inception', 
        'Intersteller', 
        'zindagi na milegi dobara', 
        'stranger things', 
        'fighter', 
        'freinds', 
        'Money heist'
    ]
}
df = pd.DataFrame(set)


v = TfidfVectorizer()
tfidf_m = v.fit_transform(df['title'])
c_sim = cosine_similarity(tfidf_m, tfidf_m)

# Title of the app
st.title("Movie Recommendation System")


def recomended():
    m_name = st.text_input("Enter a movie name:", key="movie_input")  
    if not m_name:
        return []  

    m_name = m_name.lower().strip()

    if m_name not in df['title'].str.lower().values:
        st.error("Movie not found. Please try again with a different movie name.")
        return []

    
    id = df[df['title'].str.lower() == m_name].index[0]


    s_scores = list(enumerate(c_sim[id]))
    s_scores = sorted(s_scores, key=lambda x: x[1], reverse=True)


    top_indices = [i[0] for i in s_scores[1:6]]

    return df['title'].iloc[top_indices].tolist()


recomended_movies = recomended()

if recomended_movies:
    st.success("You may also like:")
    for movie in recomended_movies:
        st.write(f"- {movie}")
