# ============================================================
#   PROJECT 3 – Simple Recommendation System
#   SoftGrowTech Internship | Task 2
#   Uses: Scikit-learn (TF-IDF + Cosine Similarity)
#         Content-Based Filtering on a built-in movie dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
#  Built-in Movie Dataset  (no download needed)
# ─────────────────────────────────────────────
MOVIES = [
    # id, title, genre, description
    (1,  "The Dark Knight",       "Action|Crime|Thriller",
     "Batman battles the Joker to save Gotham from chaos and crime"),
    (2,  "Inception",             "Action|Sci-Fi|Thriller",
     "A thief enters dreams to steal secrets from deep within the subconscious"),
    (3,  "Interstellar",          "Adventure|Drama|Sci-Fi",
     "Astronauts travel through a wormhole in search of a new home for humanity"),
    (4,  "The Matrix",            "Action|Sci-Fi",
     "A hacker discovers reality is a simulation and joins a rebellion"),
    (5,  "Avengers: Endgame",     "Action|Adventure|Sci-Fi",
     "The Avengers assemble to reverse Thanos' snap and save the universe"),
    (6,  "The Shawshank Redemption","Drama",
     "A banker is wrongfully convicted and finds hope inside a prison"),
    (7,  "Forrest Gump",          "Drama|Romance",
     "A slow-witted but kind man witnesses key historical events across decades"),
    (8,  "Titanic",               "Drama|Romance",
     "A romance blooms between a rich girl and a poor artist on the doomed ship"),
    (9,  "The Lion King",         "Animation|Adventure|Drama",
     "A young lion prince flees his kingdom and learns the true meaning of responsibility"),
    (10, "Toy Story",             "Animation|Adventure|Comedy",
     "Toys come alive and go on adventures when humans are not watching"),
    (11, "Finding Nemo",          "Animation|Adventure|Comedy",
     "A clownfish crosses the ocean to rescue his captured son"),
    (12, "The Silence of the Lambs","Crime|Drama|Thriller",
     "An FBI trainee seeks help from a cannibalistic genius to catch a serial killer"),
    (13, "Se7en",                 "Crime|Drama|Mystery|Thriller",
     "Two detectives hunt a serial killer who targets victims based on deadly sins"),
    (14, "Goodfellas",            "Biography|Crime|Drama",
     "A man rises through the ranks of the New York mob over several decades"),
    (15, "Pulp Fiction",          "Crime|Drama",
     "Interconnected stories of criminals, hitmen, and a boxer in Los Angeles"),
    (16, "Jurassic Park",         "Adventure|Sci-Fi|Thriller",
     "Scientists clone dinosaurs for a theme park that quickly becomes dangerous"),
    (17, "Gravity",               "Drama|Sci-Fi|Thriller",
     "Two astronauts survive a catastrophic accident and struggle to return to Earth"),
    (18, "La La Land",            "Comedy|Drama|Music|Romance",
     "An aspiring actress and a jazz musician fall in love while chasing their dreams"),
    (19, "Whiplash",              "Drama|Music",
     "A music student endures brutal training from a ruthless jazz instructor"),
    (20, "The Godfather",         "Crime|Drama",
     "The aging patriarch of a crime dynasty transfers control to his reluctant son"),
    (21, "Parasite",              "Comedy|Drama|Thriller",
     "A poor family schemes their way into the household of a wealthy family"),
    (22, "Get Out",               "Horror|Mystery|Thriller",
     "A Black man visits his white girlfriend's family and uncovers a sinister secret"),
    (23, "A Quiet Place",         "Drama|Horror|Sci-Fi",
     "A family struggles to survive in a post-apocalyptic world inhabited by blind monsters"),
    (24, "Knives Out",            "Crime|Drama|Mystery",
     "A detective investigates the death of a wealthy crime novelist"),
    (25, "Spider-Man: No Way Home","Action|Adventure|Sci-Fi",
     "Spider-Man asks Doctor Strange to make the world forget his secret identity"),
]


# ─────────────────────────────────────────────
#  Build DataFrame
# ─────────────────────────────────────────────
def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(MOVIES, columns=["id", "title", "genre", "description"])
    # Combine genre + description for richer TF-IDF features
    df["genre_clean"] = df["genre"].str.replace("|", " ", regex=False)
    df["features"]    = df["genre_clean"] + " " + df["description"]
    return df


# ─────────────────────────────────────────────
#  Build TF-IDF + Similarity Matrix
# ─────────────────────────────────────────────
def build_similarity_matrix(df: pd.DataFrame):
    """
    Converts 'features' column to TF-IDF vectors and
    computes cosine similarity between all movie pairs.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["features"])
    similarity   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


# ─────────────────────────────────────────────
#  Get Recommendations
# ─────────────────────────────────────────────
def get_recommendations(movie_title: str,
                         df: pd.DataFrame,
                         sim_matrix: np.ndarray,
                         top_n: int = 5) -> pd.DataFrame:
    """
    Given a movie title, returns the top_n most similar movies.
    Performs case-insensitive partial matching.
    """
    # Find matching titles (case-insensitive)
    mask  = df["title"].str.lower().str.contains(movie_title.lower())
    matches = df[mask]

    if matches.empty:
        return pd.DataFrame()   # Caller will handle "not found"

    # Use the first match
    idx   = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))

    # Sort by similarity score (descending), exclude itself
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top    = [(i, s) for i, s in scores if i != idx][:top_n]

    results = []
    for i, score in top:
        row = df.iloc[i]
        results.append({
            "Title":      row["title"],
            "Genre":      row["genre"],
            "Similarity": f"{score * 100:.1f}%",
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────
def print_all_movies(df: pd.DataFrame) -> None:
    print("\n  Available Movies:")
    print("  " + "-" * 45)
    for _, row in df.iterrows():
        print(f"  [{row['id']:>2}] {row['title']:<35} {row['genre']}")
    print()


def print_recommendations(title: str, recs: pd.DataFrame) -> None:
    print(f"\n{'=' * 58}")
    print(f"  Movies similar to  ➜  \"{title}\"")
    print(f"{'=' * 58}")
    if recs.empty:
        print("  ⚠  No recommendations found. Try a different title.")
    else:
        print(f"  {'#':<4} {'Title':<38} {'Genre':<25} {'Match'}")
        print("  " + "-" * 58)
        for i, row in recs.iterrows():
            print(f"  {i + 1:<4} {row['Title']:<38} {row['Genre']:<25} {row['Similarity']}")
    print(f"{'=' * 58}\n")


# ─────────────────────────────────────────────
#  Demo Run (5 preset queries)
# ─────────────────────────────────────────────
def run_demo(df, sim_matrix):
    demo_queries = [
        "The Dark Knight",
        "Toy Story",
        "Interstellar",
        "The Godfather",
        "Inception",
    ]
    print("\n  ── DEMO MODE: pre-set queries ──")
    for q in demo_queries:
        recs = get_recommendations(q, df, sim_matrix, top_n=5)
        print_recommendations(q, recs)


# ─────────────────────────────────────────────
#  Interactive Mode
# ─────────────────────────────────────────────
def interactive_mode(df, sim_matrix):
    print("  ── INTERACTIVE MODE (type 'list' to see all movies, 'quit' to exit) ──\n")
    while True:
        query = input("  Enter a movie name: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye! 🎬")
            break
        if query.lower() == "list":
            print_all_movies(df)
            continue
        if not query:
            print("  ⚠  Please enter a movie title.")
            continue

        recs = get_recommendations(query, df, sim_matrix, top_n=5)
        print_recommendations(query, recs)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "★" * 58)
    print("   MOVIE RECOMMENDATION SYSTEM  –  SoftGrowTech Task 2")
    print("★" * 58)
    print("   Technique : Content-Based Filtering (TF-IDF + Cosine Similarity)")
    print("   Dataset   : 25 curated movies across multiple genres\n")

    # Build data structures
    df         = build_dataframe()
    sim_matrix = build_similarity_matrix(df)

    print(f"  ✔  Dataset loaded      : {len(df)} movies")
    print(f"  ✔  Similarity matrix   : {sim_matrix.shape[0]} × {sim_matrix.shape[1]}")

    # Show all available movies
    print_all_movies(df)

    # Demo
    run_demo(df, sim_matrix)

    # Interactive
    interactive_mode(df, sim_matrix)


if __name__ == "__main__":
    main()
