import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

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
    print(f"Final processed shape: {df_cleaned.shape}")

    df_cleaned.to_csv(filepath.replace('.csv', '_processed.csv'))
    return df_cleaned

def calculate_similarity_bow(df):
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
            "score": similarity_score,
            "essay_1_id": essay_1_id,
            "essay_2_id": essay_2_id,
            "essay_1_preview": essay_1_text,
            "essay_2_preview": essay_2_text
        })

        print(f"\nSimilarity Score: {similarity_score:.4f}")
        print(f"  Essay 1 ID: {essay_1_id}")
        print(f"  Preview 1: {essay_1_text}")
        print(f"  Essay 2 ID: {essay_2_id}")
        print(f"  Preview 2: {essay_2_text}")

    return results

if __name__ == "__main__":
    SAMPLE_SIZE = 500
    FILE_PATH = 'data/train.csv'

    processed_df = load_and_preprocess(FILE_PATH, sample_n=SAMPLE_SIZE)
    processed_df.head()

    if processed_df is not None:
        start_time = time.time()

        count_matrix, similarity_matrix = calculate_similarity_bow(processed_df)
        top_pairs = find_top_pairs(similarity_matrix, processed_df, top_n=5)

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")