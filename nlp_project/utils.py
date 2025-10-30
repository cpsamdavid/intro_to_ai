import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess(filepath, sample_n=None):
    """
    Loads the dataset, filters as required, and optionally samples it.
    """
    print(f"Reading data from {filepath}...")
    df = pd.read_csv(filepath)

    print(f"Original dataset shape: {df.shape}")

    # Exclude essays with "PROPER_NAME"
    df_filtered = df[~df['full_text'].str.contains("PROPER_NAME", na=False)]
    print(f"Shape after filtering 'PROPER_NAME': {df_filtered.shape}")

    # Exclude exact duplicates
    df_cleaned = df_filtered.drop_duplicates(subset=['full_text'])
    print(f"Shape after dropping exact duplicates: {df_cleaned.shape}")

    if sample_n and sample_n < len(df_cleaned):
        print(f"Sampling {sample_n} essays for development...")
        df_cleaned = df_cleaned.sample(n=sample_n, random_state=42)

    # Reset index to make sure matrix indices match dataframe indices
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    # Text normalization and preprocessing
    print("Normalizing and cleaning text...")
    df_cleaned['full_text'] = (
        df_cleaned['full_text']
        .str.lower() # convert to lowercase
        .str.replace(r'[^a-z\s]', '', regex=True) # remove punctuation/numbers
        .str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces
        .str.strip() # trim leading/trailing spaces
    )
    
    print(f"Final processed shape: {df_cleaned.shape}")
    df_cleaned.to_csv(filepath.replace('.csv', '_processed.csv'))
    return df_cleaned

# Method 1: Bag of Words Vectorizer
def calculate_similarity_method1_bow(df):
    """
    Vectorizes text using raw word counts (Bag of Words) and
    calculates the pairwise cosine similarity matrix.
    """
    print("Initializing CountVectorizer (Bag of Words)...")

    # We use CountVectorizer instead of TfidfVectorizer
    # This will just count the occurrences of each word.
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)

    print("Fitting and transforming text data")

    # Create the Count matrix (Bag of Words)
    count_matrix = vectorizer.fit_transform(df['full_text'])

    print(f"Count matrix shape: {count_matrix.shape} (essays, unique_words)")

    print("Calculating cosine similarity matrix...")
    # This creates an (N x N) matrix, where N is the number of essays
    cosine_sim_matrix = cosine_similarity(count_matrix)
    print(f"Similarity matrix shape: {cosine_sim_matrix.shape}")

    return count_matrix, cosine_sim_matrix

# Method 2: TF-IDF Vectorizer
def calculate_similarity_tfidf(df):
    """
    Vectorizes text using TF-IDF and calculates the
    pairwise cosine similarity matrix.
    """
    print("Initializing TfidfVectorizer...")

    # This is the key change for Method 2
    # We are using TfidfVectorizer instead of CountVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    print("Fitting and transforming text data (this may take a moment)...")

    # Create the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df['full_text'])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (essays, unique_words)")

    print("Calculating cosine similarity matrix...")
    # This creates an (N x N) matrix, where N is the number of essays
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    print(f"Similarity matrix shape: {cosine_sim_matrix.shape}")

    return tfidf_matrix, cosine_sim_matrix

# Method 3: Hashing Vectorizer
def calculate_similarity_hashing(df):
    """
    Vectorizes text using HashingVectorizer and calculates the
    pairwise cosine similarity matrix.
    """
    print("Initializing HashingVectorizer...")

    # This is the key change for Method 3
    # n_features is the number of "buckets" to hash into.
    # 2**18 (262,144) is a common, large-enough size to reduce
    # the chance of "collisions".
    vectorizer = HashingVectorizer(
        stop_words='english',
        lowercase=True,
        n_features=2 ** 20,  # 1,048,576 buckets instead of 262,144
        alternate_sign=False   # Fix negative similarities
    )

    print("Fitting and transforming text data (this may take a moment)...")

    # Create the Hashing matrix
    # This vectorizer is "stateless" - the 'fit' part doesn't
    # actually learn a vocabulary.
    hash_matrix = vectorizer.fit_transform(df['full_text'])

    print(f"Hashing matrix shape: {hash_matrix.shape} (essays, n_features)")

    print("Calculating cosine similarity matrix...")
    # This creates an (N x N) matrix, where N is the number of essays
    cosine_sim_matrix = cosine_similarity(hash_matrix)
    print(f"Similarity matrix shape: {cosine_sim_matrix.shape}")

    return hash_matrix, cosine_sim_matrix


def find_top_pairs(sim_matrix, df, top_n=5):
    """
    Finds the top_n most similar (non-identical) pairs from the matrix.
    """
    print(f"Finding top {top_n} similar pairs...")

    # Get the upper triangle of the matrix (excluding the diagonal)
    upper_triangle = np.triu(sim_matrix, k=1)

    # Flatten the upper triangle to find the highest values
    flat_indices = np.argsort(upper_triangle.flatten())[-top_n:]

    # Convert the flat indices back to (row, col) matrix indices
    top_pairs_indices = [np.unravel_index(i, upper_triangle.shape) for i in flat_indices]

    print("\n Top 5 Most Similar Pairs (using Count Frequencies)")

    results = []
    for pair in reversed(top_pairs_indices): # Reverse to show highest score first
        row, col = pair
        similarity_score = sim_matrix[row, col]

        # Get the original essay info
        essay_1_id = df.loc[row, 'essay_id']
        essay_1_text = df.loc[row, 'full_text'][:100] + "..." # Get 1-line preview

        essay_2_id = df.loc[col, 'essay_id']
        essay_2_text = df.loc[col, 'full_text'][:100] + "..."

        results.append({
            "score": round(similarity_score * 100, 2),
            "essay_1_id": essay_1_id,
            "essay_2_id": essay_2_id,
            "essay_1_preview": essay_1_text,
            "essay_2_preview": essay_2_text
        })

        print(f"\nSimilarity Score: {similarity_score * 100:.2f}%")
        print(f"  Essay 1 ID: {essay_1_id}")
        print(f"  Preview 1: {essay_1_text}")
        print(f"  Essay 2 ID: {essay_2_id}")
        print(f"  Preview 2: {essay_2_text}")

    return results

def find_bottom_pairs(sim_matrix, df, bottom_n=5):
    """
    Finds the bottom_n least similar pairs, ignoring self-comparisons.
    """
    print(f"\nFinding bottom {bottom_n} least similar pairs...")

    upper_triangle = np.triu(sim_matrix, k=1)

    # Replace zeros with +inf to avoid selecting empty similarities if needed
    # (useful if many essays share no vocabulary)
    masked = np.ma.masked_equal(upper_triangle, 0)

    # Get the smallest values (least similar)
    flat_indices = np.argsort(masked.flatten())[:bottom_n]

    bottom_pairs_indices = [np.unravel_index(i, upper_triangle.shape)
                            for i in flat_indices]

    results = []
    for row, col in bottom_pairs_indices:
        similarity_score = sim_matrix[row, col]

        essay_1_id = df.loc[row, 'essay_id']
        essay_1_text = df.loc[row, 'full_text'][:100] + "..."

        essay_2_id = df.loc[col, 'essay_id']
        essay_2_text = df.loc[col, 'full_text'][:100] + "..."

        results.append({
            "score": round(similarity_score * 100, 2),
            "essay_1_id": essay_1_id,
            "essay_2_id": essay_2_id,
            "essay_1_preview": essay_1_text,
            "essay_2_preview": essay_2_text
        })

        print(f"\nSimilarity Score: {similarity_score * 100:.2f}%")
        print(f"  Essay 1 ID: {essay_1_id}")
        print(f"  Preview 1: {essay_1_text}")
        print(f"  Essay 2 ID: {essay_2_id}")
        print(f"  Preview 2: {essay_2_text}")

    return results

def get_top_pairs(method_function, df, top_n=5):
    count_matrix, similarity_matrix = method_function(df)
    top_pairs = find_top_pairs(similarity_matrix, df, top_n=top_n)
    return count_matrix, similarity_matrix, top_pairs