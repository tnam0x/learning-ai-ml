import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
import nltk

# -------- Download required NLTK resources (only need to run once) --------
nltk.download('punkt') # Pre-trained tokenizers for sentence splitting and word tokenization.
nltk.download('punkt_tab') # Pre-trained tokenizers for tab-separated text.
nltk.download('stopwords') # List of common stopwords in various languages.

# -------- Step 1: Define the raw documents --------
docA = "The dog barks loud!"
docB = "A cat meows loudly."

# -------- Step 2: Preprocessing Function --------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords (like 'the', 'a', 'was', etc.)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]

    return " ".join(filtered)  # Return cleaned text as a string

# Apply preprocessing to both documents
cleanA = preprocess(docA)
cleanB = preprocess(docB)

print("Preprocessed Doc A:", cleanA)
print("Preprocessed Doc B:", cleanB)

# -------- Step 3: Build Vocabulary & Vectorize --------
# Use CountVectorizer to convert documents into word frequency vectors
vectorizer = CountVectorizer()

# This will also build the vocabulary internally
vectors = vectorizer.fit_transform([cleanA, cleanB]).toarray()

# Vocabulary mapping (word -> index)
print("Vocabulary:", vectorizer.vocabulary_)

# Document vectors
print("Vector A:", vectors[0])
print("Vector B:", vectors[1])

# -------- Step 4: Compute Similarity and Distance --------
# Cosine similarity (1 = identical direction, 0 = orthogonal)
cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Euclidean distance (lower = more similar)
eucl_dist = euclidean_distances([vectors[0]], [vectors[1]])[0][0]

print(f"\nCosine Similarity: {cos_sim:.3f}")
print(f"Euclidean Distance: {eucl_dist:.3f}")
