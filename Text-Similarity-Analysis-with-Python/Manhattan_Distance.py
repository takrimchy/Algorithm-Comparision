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
from sklearn.metrics.pairwise import manhattan_distances

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

# Function to calculate Manhattan distance matrix between documents
def calculate_manhattan_similarity(documents):
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)
    return manhattan_distances(doc_term_matrix)

# Function to display Manhattan distance results as bar charts
def display_results(distance_matrix):
    n_documents = len(distance_matrix)
    for i in range(n_documents):
        fig, ax = plt.subplots(figsize=(10, 6))
        distances_without_self = [0 if i == j else dist for j, dist in enumerate(distance_matrix[i])]
        ax.bar(range(1, n_documents + 1), distances_without_self, color='orange')
        ax.set_xlabel('Document')
        ax.set_ylabel('Manhattan Distance')
        ax.set_title(f'Manhattan Distance of Document {i+1} to Others')
        ax.set_xticks(np.arange(1, n_documents + 1))
        ax.set_xticklabels([f'Doc {j+1}' for j in range(n_documents)])
        plt.show()


def prepare_results(distance_matrix):
    n_documents = len(distance_matrix)
    results = {}
    
    for i in range(n_documents):
        doc_results = {}
        for j in range(n_documents):
            # Handle self-comparison or zero distance
            if distance_matrix[i][j] == 0:
                similarity_percentage = "100%"
            else:
                # Normalize and convert distance to similarity percentage
                normalized_similarity = round(1 / (1 + distance_matrix[i][j]) * 100, 2)
                similarity_percentage = f"{normalized_similarity}%"
            doc_results[f'Document {j+1}'] = similarity_percentage
        results[f'Document {i+1}'] = doc_results
    
    return results



# Main function adjusted to return JSON-like results
def main(pdf_paths):
    documents = process_documents(pdf_paths)
    distance_matrix = calculate_manhattan_similarity(documents)
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
    # Now `results` holds the Manhattan distance results as a dictionary