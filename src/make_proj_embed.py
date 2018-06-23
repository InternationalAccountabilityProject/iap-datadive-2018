import pandas as pd
from gensim.models import KeyedVectors
import spacy
import argparse
nlp = spacy.load('en')

def create_embedding(spac_text, w2v, stop_words):
    embed=[]
    for tok in spac_text:
        if tok not in stop_words and tok.text.lower() in w2v:
            embed.append(w2v[tok.text.lower()])
    return np.mean(embed, axis=0) if len(embed) > 0 else None

def pre_process_projects_df(df):
    df['Project Description'] = df.apply(lambda x: x if x != 'None' else None)
    df.fillna('', inplace=True)
    df.fillna('', inplace=True)
    return df

def make_proj_embed(projects, vectors, stopwords):
    w2v = KeyedVectors.load_word2vec_format(vectors)
    with open(stopwords, 'r') as f:
        stop_words = set(f.read().strip().split("\n"))
    projects = pd.read_csv(projects)
    projects = pre_process_projects_df(projects)
    projects['spac_text'] = projects.apply(lambda x: nlp(x['Project Name'] + ' ' + x['Project Description'])
                                                      , axis=1)
    projects['fasttext_embedding'] = projects['spac_text'].apply(lambda x: create_embedding(x, w2v, stop_words))
    projects.drop('spac_text', inplace=True)
    projects.to_csv(projects, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Create embeddings for project')
    parser.add_argument('-p', '--projects', type=str, help='path to projects.csv')
    parser.add_argument('-v', '--vectors', type=str, help='path to vectors')
    parser.add_argument('-s', '--stopwords', type=str, help='path to stopwords')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    make_proj_embed(args.projects, args.vectors, args.stopwords)

if __name__ == '__main__':
    main()

