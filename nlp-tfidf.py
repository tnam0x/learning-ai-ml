import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# -------- Download required NLTK resources (only need to run once) --------
nltk.download('punkt') # Pre-trained tokenizers for sentence splitting and word tokenization.
nltk.download('punkt_tab') # Pre-trained tokenizers for tab-separated text.
nltk.download('stopwords') # List of common stopwords in various languages.
nltk.download('wordnet') # WordNet lexical database, a large lexical resource for English.
nltk.download('omw-1.4') # Open Multilingual WordNet: provides additional multilingual data supporting lemmatization and WordNet lookups across languages.

# -------- Step 1: Define the raw documents --------
documents = [
    "The dog was barking loudly.",
    "A cat meows loud.",
    "Birds are chirping in the morning."
]

# -------- Step 2: Preprocessing Function --------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens (treating words as verbs)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in filtered]

    # Join tokens back into a single string
    return " ".join(lemmatized)

# -------- Step 3: Preprocess all documents --------
cleaned_docs = [preprocess(doc) for doc in documents]

# -------- Step 4: Vectorize documents using TF-IDF --------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_docs).toarray()

# -------- Step 5: Calculate similarity and distance matrices --------
cos_sim_matrix = cosine_similarity(tfidf_matrix)
eucl_dist_matrix = euclidean_distances(tfidf_matrix)

# -------- Step 6: Output results --------
print("TF-IDF Vocabulary:", vectorizer.vocabulary_)
print("\nTF-IDF Matrix:")
for i, vec in enumerate(tfidf_matrix):
    print(f"Doc {i}:", vec)

print("\nCosine Similarity Matrix:")
print(cos_sim_matrix)

print("\nEuclidean Distance Matrix:")
print(eucl_dist_matrix)
