from cfuzzyset import cFuzzySet as FuzzySet
from os import listdir
from os.path import isfile, join

# read in player-position map
player_positions = []
with open('../player_positions.tsv','r') as f:
    content = f.readlines()
    for player in content:
        player = player.rstrip().split('\t')
        name = player[0]
        position = player[1]
        player_positions.append(position)

print set(player_positions)

