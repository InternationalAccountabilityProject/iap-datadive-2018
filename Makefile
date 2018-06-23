setup:
    . setup.sh

proj_embed:
    python src/make_proj_embed.py --p Data/EWS_Published_Project_Listing_DD.csv --v dependencies/wiki-news-300d-1M.vec --s dependencies/stop-words-list.txt