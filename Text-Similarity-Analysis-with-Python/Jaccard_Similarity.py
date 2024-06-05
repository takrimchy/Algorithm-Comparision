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
        documents.append({
            'text': text,
        })
    return documents

# Function to compute Jaccard Similarity between two sets of words
def jaccard_similarity(doc1, doc2):
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union

# Function to calculate Jaccard similarity matrix between documents
def calculate_jaccard_similarity(documents):
    n_documents = len(documents)
    similarity_matrix = np.zeros((n_documents, n_documents))
    
    for i in range(n_documents):
        for j in range(n_documents):
            if i != j:
                similarity = jaccard_similarity(documents[i]['text'], documents[j]['text'])
                similarity_matrix[i][j] = similarity
            else:
                similarity_matrix[i][j] = 1  # Document is always identical to itself
    
    return similarity_matrix

# Function to display Jaccard similarity results as bar charts
def display_results(similarity_matrix):
    n_documents = len(similarity_matrix)
    
    for i in range(n_documents):
        fig, ax = plt.subplots(figsize=(10, 6))
        similarity_without_self = [0 if i == j else sim for j, sim in enumerate(similarity_matrix[i])]
        ax.bar(range(1, n_documents + 1), similarity_without_self, color='green')
        
        ax.set_xlabel('Document')
        ax.set_ylabel('Jaccard Similarity')
        ax.set_title(f'Jaccard Similarity of Document {i+1} with Others')
        ax.set_xticks(np.arange(1, n_documents + 1))
        ax.set_xticklabels([f'Doc {j+1}' for j in range(n_documents)])
        ax.set_ylim(0, 1)  # Jaccard similarity ranges from 0 to 1
        
        plt.show()

# Main function to run the plagiarism detection with Jaccard Similarity
# def main(pdf_paths):
#     documents = process_documents(pdf_paths)
#     similarity_matrix = calculate_jaccard_similarity(documents)
#     display_results(similarity_matrix)

# # Example usage
# if __name__ == "__main__":
#     # List of PDF paths
#     pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]
#     main(pdf_paths)


# Adjusted function to return results instead of displaying a figure
# def prepare_results(similarity_matrix):
#     n_documents = len(similarity_matrix)
#     results = {}
    
#     for i in range(n_documents):
#         doc_results = {}
#         for j in range(n_documents):
#             # Adjusted to skip self-comparison in the output
#             if i != j:
#                 doc_results[f'Document {j+1}'] = similarity_matrix[i][j]
#         results[f'Document {i+1}'] = doc_results
    
#     return results
def prepare_results(similarity_matrix):
    n_documents = len(similarity_matrix)
    results = {}
    
    for i in range(n_documents):
        doc_results = {}
        for j in range(n_documents):
            # Convert similarity to percentage
            similarity_percentage = round(similarity_matrix[i][j] * 100, 2)
            doc_results[f'Document {j+1}'] =  f'{similarity_percentage}%'
        results[f'Document {i+1}'] = doc_results
    
    return results


# Main function adjusted to return JSON-like results
def main(pdf_paths):
    documents = process_documents(pdf_paths)
    similarity_matrix = calculate_jaccard_similarity(documents)
    results = prepare_results(similarity_matrix)
    
    # If you need the result as a JSON string
    json_results = json.dumps(results, indent=4)
    print(json_results)  # For demonstration; in practice, you might return this
    
    # If you're okay with a Python dictionary (which is JSON-like)
    return results

# Example usage remains the same
if __name__ == "__main__":
    pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]
    results = main(pdf_paths)
    # Now `results` holds the Jaccard similarity results as a dictionary