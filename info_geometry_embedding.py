from data.data_accessors import load_seasons
from models.distributions import Gaussian
from models.metrics import Gauss_Fisher
import numpy as np

"""
    This file contains the logic for embedding visualizing an
    information geometric embedding of prominent NBA player based
    on their shot patterns.
    
    written by Eric Nalisnick, enalisnick@gmail.com, Nov 2014
"""

def generate_info_geo_embedding(seasons):

    # get All-NBA
    all_nba_players = set()
    with open('data/AllNBA.tsv', 'r') as f:
        content=f.readlines()
        for line in content:
            line = line.rstrip().split('\t')
            for season in seasons:
                if season in line[1]:
                    all_nba_players.add(line[0])

    print "Calculating distances between %d NBA players"%(len(all_nba_players))
    labels = []
    x_guass = []
    y_gauss = []
    for idx, player in enumerate(all_nba_players):
        print "processing player #%d"%(idx)
        player_shot_data = load_seasons(seasons, split_flag=True, attributes = [player], search_name_flag=True)[0]
        g = Gaussian(2)
        g.batch_fit(player_shot_data)
        labels.append(player)
        x_guass.append((g.mu[0,0], g.sigma[0][0]))
        y_gauss.append((g.mu[0,1], g.sigma[1][1]))

    info_geo_embed = Gauss_Fisher()
    info_geo_embed.calculate_distance_matrix(x_guass)
    print "finshed computing x sim matrix"
    x_sim_mat = info_geo_embed.sim_mat
    info_geo_embed.calculate_distance_matrix(y_gauss)
    print "finshed computing y sim matrix"
    info_geo_embed.sim_mat += x_sim_mat
    info_geo_embed.visualize(labels)


if __name__ == '__main__':

    generate_info_geo_embedding(seasons=['2006-2007', '2007-2008', '2008-2009'])



