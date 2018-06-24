import spacy
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

class Reconciler(object):
    '''
    Get articles relevant to a project
    '''
    def __init__(self, proj_path, article_path):
        self.nlp = spacy.load('en')
        self.projects = pd.read_csv(proj_path, encoding='ISO-8859-1')
        self.articles = pd.read_pickle(article_path)
    def article_to_doc(self):
        '''
        Parse and tag all the articles
        '''
        docs =[]
        for doc in nlp.pipe(self.articles.article_text.astype("unicode"),
                            batch_size = 50,
                            n_threads = multiprocessing.cpu_count()-1):
            if doc.is_parsed:
                docs.append(doc)
        self.articles['doc_objs'] = docs
        self.docs = docs
        return self
    def extract_entities(self, text_blob):
        doc = self.nlp(text_blob)
        entity = [(entity.text, entity.label_)for entity in doc.ents]
        return entity
    def find_entity_similarity(self, project_row, cat_cols=CAT_COLS):
        row_list = [x for x in list(res.projects[cat_cols].iloc[project_row]) if type(x) not in [np.float64, float]]
        score_tuples = {}
        for idx, row in self.articles.iterrows():
            doc = row['doc_objs']
            score_tmp = (fuzz.partial_token_sort_ratio(row_list, doc.ents))
            row_score = sum(score_tmp) / len(score_tmp)
            score_tuples.update({idx:row_score})
        score_series = pd.Series(score_tuples).sort_values(ascending=False)
        good_scores = score_series[score_series>59]
        good_articles = self.articles.loc[good_scores, :]
        return good_articles
