from cfuzzyset import cFuzzySet as FuzzySet
from os import listdir
from os.path import isfile, join

# read in player-position map
player_to_position_map = {}
with open('./player_positions.tsv','r') as f:
    content = f.readlines()
    for player in content:
        player = player.rstrip().split('\t')
        name = player[0]
        position = player[1]
        player_to_position_map[name.lower()] = position.replace(',','/')

# use fuzzy set to match player names
player_fuzzy_set = FuzzySet(player_to_position_map.keys())

# read through game-play data
seasons = ['2006-2007', '2007-2008', '2008-2009', '2009-2010']
for season in seasons:
    dir_path = './'+season+'.regular_season'
    game_files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
    season_makes_file = open(season+'.made_shots.csv','w')
    season_misses_file = open(season+'.missed_shots.csv','w')
    for game in game_files:
        with open(dir_path+"/"+game, 'r') as in_file:
            content = in_file.readlines()
            for play in content[1:]:
                play = play.rstrip().split(',')
                if len(play)<32:
                    continue
                play_type = play[13]
                if play_type == 'shot' and not (play[30]=='' or play[31]==''):
                    # we don't have to keep points
                    # if result=='made' and play_type=='3pt', then 3 points.
                    player = play[23]
                    result = play[27]
                    shot_type = play[29]
                    x_coord = int(play[30])
                    y_coord = int(play[31])
                    position = player_to_position_map[player_fuzzy_set.get(player.lower())[0][1]]
                    if result == 'made':
                        season_makes_file.write(",".join([player, position, shot_type, str(x_coord), str(y_coord)])+"\n")
                    else:
                        season_misses_file.write(",".join([player, position, shot_type, str(x_coord), str(y_coord)])+"\n")
    season_makes_file.close()
    season_misses_file.close()