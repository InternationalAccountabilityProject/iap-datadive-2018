import numpy as np
from sklearn.externals import joblib
import spacy
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
nlp = spacy.load('en')

W2V_EMBEDDINGS = "dependencies/wiki-news-300d-1M.vec"
PROJECT_EMBEDDINGS = "data/EWS_Published Project_Listing_DD.csv"
STOP_WORDS = "dependencies/stop-word-list.txt"

class ArticleProjectMatcher(object):
    """
    A Python object that ingests an article as input, and outputs the
    semantic similarity score between the articles and all projects.
    """
    def __init__(self, project_ids, project_embeddings, w2v, stopwords):
        self.project_ids = project_ids
        self.w2v = KeyedVectors.load_word2vec_format(w2v)
        with open(stopwords, 'r') as f:
            self.stopwords = set(f.read().strip().split("\n"))
        project_df = pd.read_csv(project_embeddings, encoding='ISO-8859-1')
        self.project_embeddings = {x[0]: x[1] for x in project_df[['Project ID', 'fasttext_embedding']].values}

    def create_embedding(self, textbody):
        """
        Creates a document embedding of some text body by averaging the word
        embeddings of all words in the text body.
        """
        embedding = []
        textbody = nlp(textbody)
        for tok in textbody:
            if tok not in self.stopwords and tok.text in self.w2v:
                embedding.append(self.w2v[tok.text])
        return np.mean(embedding, axis=0) if len(embedding) > 0 else None


    def compute_similarity(self, article):
        """
        Given an article body, compute the semantic similarity between the
        article and all project embeddings stored in the class memory.
        """
        match_scores = []
        article_embedding = create_embedding(article)
        if article_embedding is None:
            return match_scores
        for project in self.project_ids:
            if self.project_embeddings[project] is not None:
                sim_score = euclidean_distances(article_embedding.reshape(1, -1)
                                                , self.project_embeddings[project].reshape(1, -1))[0][0]
                match_scores.append((project, sim_score))
            else:
                match_scores.append((project, None))
        return match_scores
