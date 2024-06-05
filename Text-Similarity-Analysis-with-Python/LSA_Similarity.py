import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
import camelot

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to extract text from a PDF
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

def calculate_lsa_similarity(documents, n_components=100):
    vectorizer = TfidfVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(doc_term_matrix)
    
    similarity_matrix = cosine_similarity(lsa_matrix)
    
    return similarity_matrix

def prepare_results(similarity_matrix):
    n_documents = len(similarity_matrix)
    results = {}
    
    for i in range(n_documents):
        doc_results = {}
        for j in range(n_documents):
            similarity_percentage = round(similarity_matrix[i][j] * 100, 2)
            doc_results[f'Document {j+1}'] = f"{similarity_percentage}%"
        results[f'Document {i+1}'] = doc_results
    
    return results

def main(pdf_paths, n_components=100):
    documents = process_documents(pdf_paths)
    similarity_matrix = calculate_lsa_similarity(documents, n_components)
    results = prepare_results(similarity_matrix)
    return results


if __name__ == "__main__":
    pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]
    documents = process_documents(pdf_paths)
    similarity_matrix = calculate_lsa_similarity(documents, n_components=100)
    results = prepare_results(similarity_matrix)
    
    json_results = json.dumps(results, indent=4)
    print(json_results)
