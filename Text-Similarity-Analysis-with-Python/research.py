# research.py
from Cosine_similarity import main as cosine_main
from Euclidean_Distance import main as euclidean_main
from Jaccard_Similarity import main as jaccard_main
from Manhattan_Distance import main as manhattan_main
from LSA_Similarity import main as lsa_main  # Import the LSA module here
import json

# List of PDF paths for testing
pdf_paths = ["PDF/Cricket.pdf", "PDF/Football.pdf", "PDF/Hockey.pdf", "PDF/Cricket2.pdf", "PDF/Cricket3.pdf"]

# List of similarity calculation functions to execute
similarity_functions = [cosine_main, jaccard_main, euclidean_main, manhattan_main, lsa_main]  # Add lsa_main here

# Names for each similarity/distance calculation for better output readability
function_names = ["Cosine Similarity", "Jaccard Similarity", "Euclidean Distance", "Manhattan Distance", "LSA Similarity"]  # Add "LSA Similarity" here

# Execute each similarity calculation and print results
for function, name in zip(similarity_functions, function_names):
    print(f"Results for {name}:")
    results = function(pdf_paths)
    # Assuming you want to print the results as JSON strings
    json_results = json.dumps(results, indent=4)
    print(json_results)
    print("\n" + "-"*50 + "\n")
