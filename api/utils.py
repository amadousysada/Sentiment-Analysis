import re
import string

import nltk
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as tf_text  # noqa: F401
from gensim.models import FastText, Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_preprocess = hub.KerasLayer(
    tfhub_handle_preprocess,
    trainable=False, name="preprocess"
)
bert_encoder = hub.KerasLayer(
    tfhub_handle_encoder,
    trainable=True, name="encoder"
)

nltk.download("punkt_tab")
nltk.download('wordnet')
nltk.download("stopwords")

def clean_tokenize(text: str):
    def remove_numerical_chars(text):
        """
            supprime les caractéres numerique du texte
        """
        return re.sub(r'\d+', '', text).strip()

    def remove_URL(text):
        """
            Supprime des URLs du texte
        """
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def remove_html(text):
        """
            Supprime les tags HTML du texte
        """
        html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(html, "", text)

    def remove_non_ascii(text):
        """
            supprime les caractères non ASCII du texte
        """
        return re.sub(r'[^\x00-\x7f]', r'', text)  # or ''.join([x for x in text if x in string.printable])

    def remove_punct(text):
        """
            supprime les ponctuations du texte
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    text = text.lower()
    text = remove_numerical_chars(text)
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_non_ascii(text)
    text = remove_punct(text)

    tokens = word_tokenize(text)
    stop = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def _average_vector(tokens: list[str], model, vector_size: int, generate_missing: bool = False) -> np.ndarray:
    if not tokens:
        return np.zeros(vector_size)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        elif generate_missing:
            vectors.append(np.random.rand(vector_size))
        else:
            vectors.append(np.zeros(vector_size))
    return np.mean(vectors, axis=0)

def sentence_to_sequence(tokens, model_embedding, max_len=16, dim=300):
    vectors = [model_embedding.wv[word] if word in model_embedding.wv else np.zeros(dim) for word in tokens]
    # Padding or trimming
    vectors = vectors[:max_len] + [np.zeros(dim)] * (max_len - len(vectors))
    return np.array(vectors)

def bert_tokenize(tokens):
    text_preprocessed = bert_preprocess([" ".join(tokens)])
    embeddings = bert_encoder(text_preprocessed)['pooled_output']

    return embeddings.numpy()

def w2vec_tokenizer(tokens):
    model = Word2Vec([tokens], min_count=1, vector_size=300, window=5)
    vec = _average_vector(tokens, model, vector_size=300).reshape(1, -1)
    vec = vec.astype(np.float64)
    return  vec

def ftext_tokenizer(tokens):
    model = FastText([tokens], vector_size=300, window=5, min_count=1)
    emb = sentence_to_sequence(tokens, model)
    emb = np.expand_dims(emb, axis=0).astype(np.float64)
    return emb