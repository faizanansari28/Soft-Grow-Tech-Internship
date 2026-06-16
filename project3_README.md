# 🎬 Project 3 – Movie Recommendation System

> **SoftGrowTech Internship | Task 2**

A **content-based movie recommendation engine** that uses TF-IDF vectorization and Cosine Similarity to suggest movies similar to any title from a curated dataset of 25 films.

---

## ✨ Features

- **Content-based filtering** using genre + plot description
- TF-IDF vectorization for rich text feature extraction
- Cosine Similarity to rank movie-to-movie closeness
- 25 curated movies spanning 10+ genres — **no download needed**
- **Demo mode** with 5 preset queries
- **Interactive mode** — type any movie name to get recommendations
- Partial, case-insensitive title matching
- Displays similarity percentage for each recommendation

---

## 🛠️ Technologies Used

| Library                         | Purpose                              |
|---------------------------------|--------------------------------------|
| `pandas`                        | Dataset management                   |
| `numpy`                         | Matrix operations                    |
| `sklearn` – `TfidfVectorizer`   | Convert text features to TF-IDF vectors |
| `sklearn` – `cosine_similarity` | Compute movie-to-movie similarity    |

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn
```

No external dataset download required — the movie data is built into the script.

---

## ▶️ How to Run

```bash
python project3_recommendation_system.py
```

The script runs a **demo** with 5 preset queries, then enters **interactive mode**.

---

## 📋 How It Works

```
Movie Dataset (25 films)
    │
    ├── Combine: genre + description → "features" column
    │
    ├── TF-IDF Vectorizer
    │       • Converts text to numerical vectors
    │       • Removes common English stop words
    │
    ├── Cosine Similarity Matrix (25 × 25)
    │       • Each cell = similarity score between two movies
    │
    └── Query: input title → find row → sort by score → top N results
```

---

## 🎥 Movie Dataset (25 Films)

Covers a wide range of genres:

| Genre              | Example Titles                                      |
|--------------------|-----------------------------------------------------|
| Action / Sci-Fi    | The Dark Knight, Inception, The Matrix, Avengers    |
| Drama / Romance    | Titanic, Forrest Gump, La La Land, Whiplash         |
| Crime / Thriller   | The Godfather, Pulp Fiction, Se7en, Knives Out      |
| Animation          | Toy Story, The Lion King, Finding Nemo              |
| Horror / Mystery   | Get Out, A Quiet Place, The Silence of the Lambs    |

---

## 🖥️ Sample Output

```
==========================================================
  Movies similar to  ➜  "Inception"
==========================================================
  #    Title                                  Genre                     Match
  ----------------------------------------------------------
  1    The Matrix                             Action|Sci-Fi             72.4%
  2    Interstellar                           Adventure|Drama|Sci-Fi    65.1%
  3    The Dark Knight                        Action|Crime|Thriller     58.3%
  4    Gravity                                Drama|Sci-Fi|Thriller     54.7%
  5    Spider-Man: No Way Home                Action|Adventure|Sci-Fi   48.2%
==========================================================
```

---

## 💬 Interactive Commands

| Input            | Action                          |
|------------------|---------------------------------|
| Movie title      | Get top 5 recommendations       |
| `list`           | Show all 25 available movies    |
| `quit` / `exit`  | Exit the program                |

---

## 📁 Project Structure

```
project3_recommendation_system.py
├── MOVIES                        # Built-in dataset (25 movies)
├── build_dataframe()             # Create pandas DataFrame + features column
├── build_similarity_matrix()     # TF-IDF → Cosine Similarity matrix
├── get_recommendations()         # Core recommendation logic
├── print_all_movies()            # List all available titles
├── print_recommendations()       # Formatted results display
├── run_demo()                    # 5 preset demo queries
└── interactive_mode()            # User input loop
```

---

## 🔧 Customisation

To add more movies, extend the `MOVIES` list in the script:

```python
(26, "Your Movie Title", "Genre1|Genre2", "Brief plot description here"),
```

The similarity matrix rebuilds automatically on next run.

---

## 👤 Author

**Faizan Ansari** — SoftGrowTech Internship
