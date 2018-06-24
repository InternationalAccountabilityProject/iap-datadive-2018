setup:
    . setup.sh
    pip install -r requirements.txt
    python -m spacy download en

proj_embed:
    python src/make_proj_embed.py --p "Data/EWS_Published Project_Listing_DD.csv" --v dependencies/wiki-news-300d-1M.vec --s dependencies/stop-word-list.txt
