import numpy as np
from sklearn.externals import joblib
import spacy
from gensim.models import KeyedVectors
import pandas as pd

W2V_EMBEDDINGS = "wiki-news-300d-1M.vec"
PROJECT_EMBEDDINGS = "project.csv"
STOP_WORDS = "stopwords.csv"

class ArticleProjectMatcher(object):
    """
    A Python object that ingests an article as input, and outputs the
    semantic similarity score between the articles and all projects.
    """
    def __init__(self, project_ids, project_embeddings, w2v, stopwords):
        self.project_ids = project_ids
        self.project_embeddings = project_embeddings
        self.w2v = KeyedVectors.load_word2vec_format(w2v)
        self.stopwords = stopwords.split(",")

    def add_project_to_list(project, project_idx):
        """
        Append a new project id and embedding to our object's memory.
        """
        self.project_ids.append(project_idx)
        embed_proj = self.create_embedding(project)
        self.project_embeddings.append(embed_proj)

    def create_embedding(textbody):
        """
        Creates a document embedding of some text body by averaging the word
        embeddings of all words in the text body.
        """
        embedding = []
        for tok in textbody:
            if tok not in self.stopwords and tok.text in self.w2v:
                embedding.append(w2v[tok.text])
        return np.mean(embedding, axis = 0) if len(embedding) > 0 else None


    def compute_similarity(article):
        """
        Given an article body, compute the semantic similarity between the
        article and all project embeddings stored in the class memory.
        """
        match_score = []
        article_embedding = create_embedding(article)
        for idx, project in enumerate(self.project_embeddings):
            if project is not None:
                simlarity_score = np.linalg.norm(project - article_embedding)
                match_scores.append((self.project_ids[idx], similarity_score))
            else:
                match_scores.append((self.project_ids[idx], 0.0))
        return match_scores
