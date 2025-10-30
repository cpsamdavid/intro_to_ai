import pandas as pd
import numpy as np
from utils import (load_and_preprocess, calculate_similarity_method1_bow, calculate_similarity_tfidf,
                   calculate_similarity_hashing, find_bottom_pairs, get_top_pairs)
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc

if __name__ == "__main__":
    SAMPLE_SIZE = 5000
    FILE_PATH = 'nlp_project\\data\\train.csv'

    processed_df = load_and_preprocess(FILE_PATH, sample_n=SAMPLE_SIZE)
    processed_df.head()

    if processed_df is not None:
        methods = {
            "Bag of Words": calculate_similarity_method1_bow,
            "TF-IDF": calculate_similarity_tfidf,
            "Hashing": calculate_similarity_hashing
        }

        execution_times = {}

        for name, method in methods.items():
            print(f"\nRunning {name}...")
            start_time = time.time()
            count_matrix, similarity_matrix, top_pairs = get_top_pairs(method, processed_df, top_n=5)
            bottom_pairs = find_bottom_pairs(similarity_matrix, processed_df, bottom_n=5)
            end_time = time.time()
            execution_times[name] = end_time - start_time
            print(f"{name} took {execution_times[name]:.2f} seconds.")
            
            # Visualize Similarity Score Distribution for the last method
            flat_scores = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            plt.figure(figsize=(8, 4))
            sns.histplot(flat_scores * 100, bins=50, kde=True, color="steelblue")
            plt.xlabel("Similarity (%)")
            plt.ylabel("Number of Essay Pairs")
            plt.title(f"Distribution of Essay Similarity Scores - {name}")
            plt.tight_layout()
            plt.savefig(f"similarity_distribution_{name}.png")
            plt.show()
            
            # Visualize Top Pairs Similarity Scores
            top_scores = [pair['score'] for pair in top_pairs]
            pair_labels = [f"{pair['essay_1_id']}–{pair['essay_2_id']}" for pair in top_pairs]

            plt.figure(figsize=(8, 5))
            sns.barplot(x=top_scores, y=pair_labels, palette="viridis")
            plt.xlabel("Similarity (%)")
            plt.ylabel("Essay Pair (ID-ID)")
            plt.title(f"Top 5 Most Similar Essay Pairs - {name}")
            plt.tight_layout()
            plt.savefig(f"similarity_top_pairs_{name}.png")  # saves chart for your report
            plt.show()
            
            # Clean RAM Between Methods
            del similarity_matrix
            del count_matrix
            gc.collect()
        
        # Visualize Execution Times
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(execution_times.keys()), y=list(execution_times.values()), palette="mako")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Runtime Comparison of Similarity Methods")
        plt.tight_layout()
        plt.savefig("similarity_runtime_comparison.png")
        plt.show()
