import PyPDF2
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import camelot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Function to extract text from a PDF
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        
        tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
        for table in tables:
            for index, row in table.df.iterrows():
                text += ' '.join(row.tolist()) + ' '
    return text

def normalize_case(text):
    return text.upper() 

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return ' '.join(lemmatized_tokens)

def process_documents(pdf_paths):
    documents = []
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        text = normalize_case(text)
        text = remove_stop_words(text)
        text = lemmatize_text(text)
        documents.append(text)
    return documents

# Function to calculate Euclidean distance matrix between documents
def calculate_euclidean_similarity(documents):
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)
    return euclidean_distances(doc_term_matrix)

# Function to display Euclidean distance results as bar charts
def display_results(distance_matrix):
    n_documents = len(distance_matrix)
    for i in range(n_documents):
        fig, ax = plt.subplots(figsize=(10, 6))
        distances_without_self = [0 if i == j else dist for j, dist in enumerate(distance_matrix[i])]
        ax.bar(range(1, n_documents + 1), distances_without_self, color='red')
        ax.set_xlabel('Document')
        ax.set_ylabel('Euclidean Distance')
        ax.set_title(f'Euclidean Distance of Document {i+1} to Others')
        ax.set_xticks(np.arange(1, n_documents + 1))
        ax.set_xticklabels([f'Doc {j+1}' for j in range(n_documents)])
        plt.show()

# Main function to run the plagiarism detection with Euclidean Distance
# def main(pdf_paths):
#     documents = process_documents(pdf_paths)
#     distance_matrix = calculate_euclidean_similarity(documents)
#     display_results(distance_matrix)

# # Example usage
# if __name__ == "__main__":
#     # List of PDF paths
#     pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]
#     main(pdf_paths)

# Adjusted function to return results instead of displaying a figure
# def prepare_results(distance_matrix):
#     n_documents = len(distance_matrix)
#     results = {}
    
#     for i in range(n_documents):
#         doc_results = {}
#         for j in range(n_documents):
#             # Skipping self-comparison to avoid displaying zero distances
#             if i != j:
#                 doc_results[f'Document {j+1}'] = distance_matrix[i][j]
#         results[f'Document {i+1}'] = doc_results
    
#     return results
def prepare_results(distance_matrix):
    n_documents = len(distance_matrix)
    results = {}
    
    # Assuming the maximum distance possible is known or can be calculated
    # If not, you could normalize the distances by the largest distance found in your matrix
    max_distance = max([max(row) for row in distance_matrix if max(row) != 0])
    
    for i in range(n_documents):
        doc_results = {}
        for j in range(n_documents):
            if distance_matrix[i][j] == 0:
                # Handle self-comparison or zero distance
                similarity_percentage = 100
            else:
                # Normalize and convert distance to similarity percentage
                # This assumes distance_matrix[i][j] is never negative
                normalized_distance = distance_matrix[i][j] / max_distance
                similarity_percentage = round((1 - normalized_distance) * 100, 2)
            doc_results[f'Document {j+1}'] = f'{similarity_percentage}%'
        results[f'Document {i+1}'] = doc_results
    
    return results



# Main function adjusted to return JSON-like results
def main(pdf_paths):
    documents = process_documents(pdf_paths)
    distance_matrix = calculate_euclidean_similarity(documents)
    results = prepare_results(distance_matrix)
    
    # If you need the result as a JSON string
    json_results = json.dumps(results, indent=4)
    print(json_results)  # For demonstration; in practice, you might return this
    
    # If you're okay with a Python dictionary (which is JSON-like)
    return results

# Example usage remains the same
if __name__ == "__main__":
    pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]
    results = main(pdf_paths)
    # Now `results` holds the Euclidean distance results as a dictionary