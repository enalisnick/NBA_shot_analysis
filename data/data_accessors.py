import numpy as np

def load_seasons(seasons, split_flag=False, attributes=None, with_attributes_flag=False, search_name_flag=False):
    if attributes==None:
        data_loader = load_location_data
    else:
        data_loader = load_data_by_attributes

    data1, data2 = data_loader(seasons[0], split_flag, attributes, with_attributes_flag, search_name_flag)
    for season in seasons[1:]:
        temp_data1, temp_data2 = data_loader(season, split_flag, attributes, with_attributes_flag, search_name_flag)
        # stack data
        if temp_data1.shape[1] == data1.shape[1]:
            data1 = np.vstack((data1, temp_data1))
            data2 = np.vstack((data2, temp_data2))
    return (data1, data2)

def load_location_data(season, split_flag, attributes, with_attributes_flag, search_name_flag):
    """ Loads just location data """
    # load made shots
    with open('./data/made_shots_rollup/'+season+'.made_shots.csv') as f:
        X_made = np.loadtxt(f, delimiter=',', usecols=(3, 4), unpack=True).T

    # load missed shots
    with open('./data/missed_shots_rollup/'+season+'.missed_shots.csv') as f:
        X_missed = np.loadtxt(f, delimiter=',', usecols=(3, 4), unpack=True).T

    if split_flag:
        return (X_made, X_missed)

    Y_made = np.ones((X_made.shape[0],1))
    Y_missed = np.zeros((X_missed.shape[0],1))

    X = np.vstack((X_made, X_missed))
    Y = np.vstack((Y_made, Y_missed))
    # shuffle the data
    rand_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rand_state)
    np.random.shuffle(Y)
    return (X, Y)

def load_data_by_attributes(season, split_flag, attributes, with_attributes_flag, search_name_flag):
    """ Loads data with specific attributes """
    X_made = load_data_by_attributes_helper('./data/made_shots_rollup/'+season+'.made_shots.csv', attributes, with_attributes_flag, search_name_flag)
    X_missed = load_data_by_attributes_helper('./data/missed_shots_rollup/'+season+'.missed_shots.csv', attributes, with_attributes_flag, search_name_flag)

    if split_flag:
        return (X_made, X_missed)
    
    Y_made = np.ones((X_made.shape[0],1))
    Y_missed = np.zeros((X_missed.shape[0],1))

    X = np.vstack((X_made, X_missed))
    Y = np.vstack((Y_made, Y_missed))
    # shuffle the data
    rand_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rand_state)
    np.random.shuffle(Y)
    return (X, Y)

def load_data_by_attributes_helper(file_name, attributes, with_attributes_flag, search_name_flag):
    #alt_shot_type_map = {'tip':'layup', 'finger':'layup', 'bank':'jump'}
    data = []
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            line = line.rstrip().replace('tip','layup').replace('finger','layup').replace('bank','jump').split(',')
            player_name = line[0]
            line = line[1:] #get rid of player name
            keep_line = False
            for feature_idx in xrange(len(line)):
                for attribute in attributes:
                    if attribute in line[feature_idx]:
                        keep_line = True
                    # HACK: We can't keep player names since we just 'F', 'G', etc for positions
                    # but I want to get players by name for embedding, so I added this flag for the time being
                    if search_name_flag and len(attribute)>1 and attribute in player_name:
                        keep_line = True
            if keep_line:
                if not with_attributes_flag:
                    data.append([int(line[2]),int(line[3])])
                else:
                    feature_vec = [int(line[2]),int(line[3])]
                    for attribute in attributes:
                        in_flag = False
                        for item in line:
                            if attribute in item:
                                in_flag = True
                        if in_flag:
                            feature_vec.append(1)
                        else:
                            feature_vec.append(0)
                    data.append( feature_vec )
    return np.array(data)



