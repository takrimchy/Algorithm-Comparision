import PyPDF2
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
import camelot
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
                # Process each row in the table and concatenate the text
                text += ' '.join(row.tolist()) + ' '

    return text

# Function to normalize case
def normalize_case(text):
    return text.upper() 

# Function to remove stop words from text
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    # Modify the stop words removal technique
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
    return ' '.join(filtered_tokens)

# Function for lemmatization
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    # Modify the lemmatization technique
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return ' '.join(lemmatized_tokens)

# Function to process multiple PDF documents
def process_documents(pdf_paths):
    documents = []
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        text = normalize_case(text)
        text = remove_stop_words(text)
        text = lemmatize_text(text)
        # Remove reference to detect_content_representation and validate_references
        documents.append({
            'text': text,
        })
    return documents

# Function to calculate cosine similarity between documents
def calculate_similarity(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    trsfm = vectorizer.fit_transform([doc['text'] for doc in documents])
    return cosine_similarity(trsfm)

# Function to display similarity results as bar charts
def display_results(similarity_matrix):
    n_documents = len(similarity_matrix)
    
    for i in range(n_documents):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, n_documents + 1), similarity_matrix[i], color='blue')
        
        ax.set_xlabel('Document')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Similarity of Document {i+1} with Others')
        ax.set_xticks(np.arange(1, n_documents + 1))
        ax.set_xticklabels([f'Document {j}' for j in range(1, n_documents + 1)])
        
        plt.show()

# # Main function to run the plagiarism detection
# def main(pdf_paths):
#     documents = process_documents(pdf_paths)
#     similarity_matrix = calculate_similarity(documents)
#     display_results(similarity_matrix)

# # Example usage
# if __name__ == "__main__":
#     # List of PDF paths
#     pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf","PDF/Cricket3.pdf"]
#     main(pdf_paths)
# def prepare_results(similarity_matrix):
#     n_documents = len(similarity_matrix)
#     results = {}
    
#     for i in range(n_documents):
#         doc_results = {}
#         for j in range(n_documents):
#             doc_results[f'Document {j+1}'] = similarity_matrix[i][j]
#         results[f'Document {i+1}'] = doc_results
    
#     return results
def prepare_results(similarity_matrix):
    n_documents = len(similarity_matrix)
    results = {}
    
    for i in range(n_documents):
        doc_results = {}
        for j in range(n_documents):
            # Convert similarity to percentage and round to 2 decimal places
            similarity_percentage = round(similarity_matrix[i][j] * 100, 2)
            doc_results[f'Document {j+1}'] = f'{similarity_percentage}%'
        results[f'Document {i+1}'] = doc_results
    
    return results


# Main function adjusted to return JSON-like results
def main(pdf_paths):
    documents = process_documents(pdf_paths)
    similarity_matrix = calculate_similarity(documents)
    results = prepare_results(similarity_matrix)
    
    # If you need the result as a JSON string
    json_results = json.dumps(results, indent=4)
    print(json_results)  # For demonstration; in practice, you might return this
    
    # If you're okay with a Python dictionary (which is JSON-like)
    return results

# Example usage remains the same
if __name__ == "__main__":
    pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf","PDF/Cricket3.pdf"]
    results = main(pdf_paths)
    # Now `results` holds the similarity results as a dictionary