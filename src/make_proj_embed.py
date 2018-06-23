import pandas as pd
from gensim.models import KeyedVectors
import spacy
import argparse
nlp = spacy.load('en')

def create_embedding(spac_text, w2v, stop_words):
    """

    Parameters
    ----------
    spac_text : spacy document
        document after loading into spacy
    w2v : KeyedVectors
        embeddings loaded into KeyedVectors
    stop_words : set(str)
        set of string stopwords

    Returns
    -------
    np.array of size 300, mean of tokens
    """
    embed=[]
    for tok in spac_text:
        if tok not in stop_words and tok.text.lower() in w2v:
            embed.append(w2v[tok.text.lower()])
    return np.mean(embed, axis=0) if len(embed) > 0 else None

def pre_process_projects_df(df):
    """

    Parameters
    ----------
    df : DataFrame
        dataframe obj

    Returns
    -------
    Dataframe after preprocessing
    """
    df['Project Description'] = df.apply(lambda x: x if x != 'None' else None)
    df.fillna('', inplace=True)
    df.fillna('', inplace=True)
    return df

def make_proj_embed(project_path, vectors, stopwords):
    """

    Parameters
    ----------
    projects : str
        path to projects csv
    vectors : str
        path to vector file
    stopwords : str
        path to stopwords file

    Returns
    -------
    None
    """
    print('Getting Embeddings')
    w2v = KeyedVectors.load_word2vec_format(vectors)
    print('Getting Stopwords')
    with open(stopwords, 'r') as f:
        stop_words = set(f.read().strip().split("\n"))
    print('Getting Projects')
    projects = pd.read_csv(project_path, encoding='ISO-8859-1')
    projects = pre_process_projects_df(projects)
    projects['spac_text'] = projects.apply(lambda x: nlp(x['Project Name'] + ' ' + x['Project Description'])
                                                      , axis=1)
    projects['fasttext_embedding'] = projects['spac_text'].apply(lambda x: create_embedding(x, w2v, stop_words))
    projects.drop('spac_text', inplace=True)
    print('Output to csv')
    projects.to_csv(project_path, index=False)

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

