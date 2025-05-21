import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt
import random
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# downloads sample twitter dataset.
nltk.download('twitter_samples')
nltk.download('stopwords')

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))

print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))

# Visualize the data
plt.figure(figsize=(8, 8))
labels = ['Positive', 'Negative']
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Positive and Negative Tweets')
plt.show()

# Preprocess the tweets
# For NLP, the preprocessing steps are comprised of the following tasks:
    # Tokenizing the string
    # Lowercasing
    # Removing stop words and punctuation
    # Stemming
# Select a random tweet
tweet = all_positive_tweets[2277]
print(tweet)

# remove old style retweet text "RT"
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
# remove hyperlinks
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)
# remove hashtags
# only removing the hash # sign from the word
tweet2 = re.sub(r'#', '', tweet2)

# Tokenize the string
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)
print('Tokenized string:\n', tweet_tokens)

# Remove stop words and punctuations
stopwords_english = stopwords.words('english')
print('Stop words:\n', stopwords_english)
print('Punctuation: ', string.punctuation)

tweet_clean = []
for word in tweet_tokens:
    if (word not in stopwords_english) and (word not in string.punctuation):
        tweet_clean.append(word)
print('Removed stop words and punctuation:\n', tweet_clean)

# Stemming: converting a word to its most general form
stemmer = PorterStemmer()
tweet_stem = []
for word in tweet_clean:
    stem_word = stemmer.stem(word)
    tweet_stem.append(stem_word)
print('Stemmed words:\n', tweet_stem)

