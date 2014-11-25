from os import listdir
from os.path import isfile, join

shot_types = []
for season in ['2006-2007', '2007-2008', '2008-2009', '2009-2010']:
    with open('../made_shots_rollup/'+season+'.made_shots.csv','r') as f:
        content = f.readlines()
        for line in content:
            line = line.rstrip().split(',')
            name = line[0]
            position = line[1]
            shot_type = line[2]
            shot_types.append(shot_type)
    with open('../missed_shots_rollup/'+season+'.missed_shots.csv','r') as f:
        content = f.readlines()
        for line in content:
            line = line.rstrip().split(',')
            name = line[0]
            position = line[1]
            shot_type = line[2]
            shot_types.append(shot_type)

print set(shot_types)
