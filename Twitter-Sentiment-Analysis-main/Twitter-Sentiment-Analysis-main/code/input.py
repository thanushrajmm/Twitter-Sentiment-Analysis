import re
import sys
import pickle
import random
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist
from scipy.sparse import lil_matrix
from utils import write_status
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score


# Define file paths
use_stemmer=False
FREQ_DIST_FILE = 'train-tweet-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = 'train-tweet-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = 'train-tweet-processed.csv'
TEST_PROCESSED_FILE = 'test-processed.csv'
MODEL_FILE = 'randomforest_model.pkl'

# Load unigrams and bigrams
with open(FREQ_DIST_FILE, 'rb') as freq_dist_file:
    unigrams = pickle.load(freq_dist_file)

with open(BI_FREQ_DIST_FILE, 'rb') as bi_freq_dist_file:
    bigrams = pickle.load(bi_freq_dist_file)

# Load Porter Stemmer
porter_stemmer = PorterStemmer()


def preprocess_word(word):
    # (Same as in the provided preprocess.py)
     # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word



def is_valid_word(word):
    # (Same as in the provided preprocess.py)
     # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)
    


def handle_emojis(tweet):
    # (Same as in the provided preprocess.py)
     # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet



def preprocess_tweet(tweet):
    # (Same as in the provided preprocess.py)
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)




# Function to get feature vector from a tweet
def get_feature_vector(tweet):
    words = tweet.split()
    uni_feature_vector = [word for word in words if unigrams.get(word)]
    
    bi_feature_vector = []
    if USE_BIGRAMS:
        bi_feature_vector = [(words[i], words[i + 1]) for i in range(len(words) - 1)
                             if bigrams.get((words[i], words[i + 1]))]

    return uni_feature_vector, bi_feature_vector

# Function to extract features from a tweet
def extract_features(tweets, batch_size=500, test_file=True, feat_type='presence'):
    features = lil_matrix((len(tweets), VOCAB_SIZE))
    labels = np.zeros(len(tweets)) if not test_file else None
    
    for i, (tweet_id, sentiment, feature_vector) in enumerate(tweets):
        tweet_words, tweet_bigrams = feature_vector
        if feat_type == 'presence':
            tweet_bigrams = set(tweet_bigrams) if tweet_bigrams else set()
        for word in tweet_words:
            idx = unigrams.get(word)
            if idx is not None:  # Check if the word is in unigrams
                features[i, idx] += 1
        if USE_BIGRAMS:
            for bigram in tweet_bigrams:
                idx = bigrams.get(bigram)
                if idx is not None:  # Check if the bigram is in bigrams
                    features[i, UNIGRAM_SIZE + idx] += 1
        if not test_file:
            labels[i] = sentiment
    
    return features, labels








def apply_tf_idf(X):
    transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    transformer.fit(X)
    return transformer
# Function to train the Random Forest model
def train_random_forest(train_tweets, feat_type='presence'):
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    features, labels = extract_features(train_tweets, test_file=False, feat_type=FEAT_TYPE)
    i = 1
    n_train_batches = 1  # There is only one batch
    write_status(i, n_train_batches)
    if feat_type == 'frequency':
        tfidf = apply_tf_idf(features)  # Assuming only one batch
        features = tfidf.transform(features)
    clf.fit(features, labels)

    # Print accuracy on the training data
    train_predictions = clf.predict(features)
    train_accuracy = accuracy_score(labels, train_predictions)
    print(f'Training Accuracy: {train_accuracy}')

    return clf




# Function to preprocess and classify a tweet
# def process_and_classify_tweet(tweet, clf):
#     processed_tweet = preprocess_tweet(tweet)
#     features, labels = extract_features(train_tweets, test_file=False, feat_type=FEAT_TYPE)

#     if FEAT_TYPE == 'frequency':
#         features = apply_tf_idf(features)
#     prediction = clf.predict(features)
#     return processed_tweet, prediction

# def process_and_classify_tweet(tweet, clf):
#     processed_tweet = preprocess_tweet(tweet)
#     uni_feature_vector, bi_feature_vector = get_feature_vector(processed_tweet)
#     features, _ = extract_features([(None, None, (uni_feature_vector, bi_feature_vector))], test_file=True)

#     if FEAT_TYPE == 'frequency':
#         features = apply_tf_idf(features)
#     prediction = clf.predict(features)
#     return processed_tweet, prediction
def process_and_classify_tweet(tweet, clf):
    processed_tweet = preprocess_tweet(tweet)
    uni_feature_vector, bi_feature_vector = get_feature_vector(processed_tweet)
    features, _ = extract_features([(None, None, (uni_feature_vector, bi_feature_vector))], test_file=True)

    if FEAT_TYPE == 'frequency':
        features = apply_tf_idf(features)
    prediction = clf.predict(features)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return processed_tweet, sentiment




def write_status(i, total):
    sys.stdout.write('\r')
    sys.stdout.write('Processing %d/%d' % (i, total))
    sys.stdout.flush()

# # Main script
# if __name__ == '__main__':
#     np.random.seed(1337)
#     USE_BIGRAMS = False  # Set to True if you want to use bigrams
#     UNIGRAM_SIZE = 15000
#     VOCAB_SIZE = UNIGRAM_SIZE
#     FEAT_TYPE = 'presence'

#     # Load training data
#     tweets = preprocess_tweet(TRAIN_PROCESSED_FILE, test_file=False)
#     random.shuffle(tweets)
#     train_tweets = tweets

#     # Train Random Forest model
#     clf = train_random_forest(train_tweets, feat_type=FEAT_TYPE)

#     # Save the trained model to a file
#     with open(MODEL_FILE, 'wb') as model_file:
#         pickle.dump(clf, model_file)

#     # Now, you can use the trained model for making predictions
#     tweet_input = input("Enter a tweet: ")
#     processed_tweet, classification_result = process_and_classify_tweet(tweet_input)

#     print("\nProcessed Tweet: ", processed_tweet)
#     print("Classification Result: ", "Positive" if classification_result == 1 else "Negative")
def process_tweets(csv_file, test_file=True):
    tweets = []
    print('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            write_status(i + 1, total)
    print('\n')
    return tweets

def top_n_words(freq_dist_file, n):
    with open(freq_dist_file, 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    return {word: idx for idx, (word, _) in enumerate(freq_dist.most_common(n))}
# Main script
if __name__ == '__main__':
    np.random.seed(1337)
    USE_BIGRAMS = False  # Set to True if you want to use bigrams
    UNIGRAM_SIZE = 15000
    VOCAB_SIZE = UNIGRAM_SIZE
    FEAT_TYPE = 'presence'

    train_tweets = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    unigrams = top_n_words(FREQ_DIST_FILE, UNIGRAM_SIZE)
    clf = train_random_forest(train_tweets, feat_type='presence')

    # Get input tweet from user
    input_tweet = input("Enter a tweet: ")

    processed_tweet, sentiment = process_and_classify_tweet(input_tweet, clf)

    print("\nProcessed Tweet:", processed_tweet)
    print("Sentiment:", sentiment)

