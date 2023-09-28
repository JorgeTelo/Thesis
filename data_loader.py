import glob, os
import numpy as np
import scipy.io as sio

def synergy_data_global(directory):
    owd = os.getcwd()
    os.chdir(directory)
    my_data = {}
    data = sio.loadmat('shadow_data.mat')
    key = sorted(data.keys())[3]
    os.chdir(owd)
    return data[key]

def synergy_data(directory):
    owd = os.getcwd()
    os.chdir(directory)
    my_data = {}
    for file in glob.glob("*.mat"):
        data = sio.loadmat(file)
        key = sorted(data.keys())[3]
        if key in my_data:
            my_data[key] = np.concatenate((my_data[key], data[key]))
        else:
            my_data[key] = data[key]

    # print (my_data.keys())
    for k in my_data.keys():
        print (k, " grasp has ", my_data[k].shape[0], " datapoints")
    
    if 'globaldata' in my_data:
        my_data.pop('globaldata')

    os.chdir(owd)
    return my_data

def synergy_data_cons(directory):
    data = synergy_data(directory)
    data_new = np.empty((0, 21), float)
    labels = np.empty((0, 1), float)
    count = 1
    for grasp_type in data:
        data_new = np.append(data_new, data[grasp_type], axis=0)
        labels = np.append(labels, np.full((data[grasp_type].shape[0], ), count))
        count += 1

    return data_new, labels

def synergy_data_rand(directory):
    data = synergy_data(directory)
    # data.pop('globaldata')

    data_new = np.empty((0, 21), float)
    labels = np.empty((0, 1), float)

    count = 1
    for grasp_type in data:
        print (grasp_type)
        print (data[grasp_type].shape)
        data_new = np.append(data_new, data[grasp_type], axis=0)
        labels = np.append(labels, np.full((data[grasp_type].shape[0], ), count))
        count += 1

    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)

    return data_new[indices], labels[indices], list(data.keys())

def get_grasps_by_label(grasps, labels):
    """

    Parameters
    ----------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    grasps_type : ndarray
        1-D array containing the grasp type of every grasp posture in grasps.
    
    Returns
    -------
    grasp_data : list
        Contains ndarrays of grasp postures per grasp type
    """
    number_of_types = labels.max()
    grasp_data = []
    
    for grasp_type in range(number_of_types):
        idxs = np.isin(labels, [grasp_type + 1])
        grasp_data.append(grasps[idxs])

    return grasp_data

def get_grasps_with_object_sizes(grasps, labels):
    """

    Parameters
    ----------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    labels : ndarray
        2-D array containing the grasp type and object configuration of every grasp posture in grasps.
    
    Returns
    -------
    grasp_data : list
        Contains ndarrays of grasp postures per grasp type
    """
    idxs = np.isin(labels[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18])

    y = labels[idxs].copy()
    grasps = grasps[idxs]

    idxs_big = np.isin(y[:, 0], [1, 4, 5, 16])
    y[idxs_big, 0] = 1
    idxs_medium = np.isin(y[:, 0], [2, 6, 7, 18])
    y[idxs_medium, 0] = 0.5
    idxs_small = np.isin(y[:, 0], [3, 8, 9, 17])
    y[idxs_small, 0] = 0

    object_size = y[:, 0].astype(int)
    grasp_type = y[:, 1].astype(int)
    return grasps, grasp_type, object_size

def get_grasps_with_object_sizes_and_types(grasps, labels):
    """

    Parameters
    ----------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    labels : ndarray
        2-D array containing the grasp type and object configuration of every grasp posture in grasps.
    
    Returns
    -------
    grasp_data : list
        Contains ndarrays of grasp postures per grasp type
    """
    idxs = np.isin(labels[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18])
    # idxs = np.isin(labels[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20])

    y = labels[idxs].copy()
    grasps = grasps[idxs]

    idxs_big = np.isin(y[:, 0], [1, 4, 5, 16, 20])
    y[idxs_big, 0] = 1
    idxs_medium = np.isin(y[:, 0], [2, 6, 7, 18])
    y[idxs_medium, 0] = 0.5
    idxs_small = np.isin(y[:, 0], [3, 8, 9, 17, 19])
    y[idxs_small, 0] = 0

    object_type = labels[idxs, 0].copy()
    idxs_ball = np.isin(object_type, [1, 2, 3])
    idxs_cylinder = np.isin(object_type, [4, 5, 6, 7, 8, 9])
    idxs_box = np.isin(object_type, [16, 17, 18, 19, 20])
    object_type[idxs_ball] = 1
    object_type[idxs_cylinder] = 2
    object_type[idxs_box] = 3

    object_size = y[:, 0].astype(int)
    grasp_type = y[:, 1].astype(int)

    print ("Loaded ", grasps.shape[0], " grasps!")
    # return grasps, grasp_type, object_type, object_size, idxs
    return grasps, grasp_type, object_type, object_size

def get_grasps_with_object_sizes_and_types_idxs(grasps, labels):
    """

    Parameters
    ----------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    labels : ndarray
        2-D array containing the grasp type and object configuration of every grasp posture in grasps.
    
    Returns
    -------
    grasp_data : list
        Contains ndarrays of grasp postures per grasp type
    """
    idxs = np.isin(labels[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18])
    # idxs = np.isin(labels[:, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20])

    y = labels[idxs].copy()
    grasps = grasps[idxs]

    idxs_big = np.isin(y[:, 0], [1, 4])
    y[idxs_big, 0] = 1.0
    idxs_medium = np.isin(y[:, 0], [2, 5, 8, 16])
    y[idxs_medium, 0] = 0.5
    idxs_small = np.isin(y[:, 0], [3, 6, 7, 9, 17, 18])
    y[idxs_small, 0] = 0.0

    object_type = labels[idxs, 0].copy()
    idxs_ball = np.isin(object_type, [1, 2, 3])
    idxs_cylinder = np.isin(object_type, [4, 5, 6, 7, 8, 9])
    idxs_box = np.isin(object_type, [16, 17, 18])
    object_type[idxs_ball] = 1
    object_type[idxs_cylinder] = 2
    object_type[idxs_box] = 3

    object_size = y[:, 0].astype(int)
    grasp_type = y[:, 1].astype(int)

    print ("Loaded ", grasps.shape[0], " grasps!")
    return grasps, grasp_type, object_type, object_size, idxs

def load_data(direc, robot='shadow'):
    """
    Load dataset of grasp postures and corresponding grasp types
    from mat files.

    Parameters
    ----------
    direc : string
        String containing the directory where the data are stored
    
    Returns
    -------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    grasps_type : ndarray
        1-D array containing the grasp type of every grasp posture in grasps.
    """
    grasps = np.load(direc + '/' + robot + '_data.npy')
    labels = np.load(direc + '/' + robot + '_labels.npy')
    return grasps, labels

def select_grasps(grasps, labels, object_configuration):
    """
    Parameters
    ----------
    grasps : ndarray
        2-D array containing the grasp postures in row-major format.
    labels : ndarray
        2-D array containing the grasp type and object configuration of every grasp posture in grasps.
    
    Returns
    -------
    grasp_data : list
        Contains ndarrays of grasp postures per grasp type
    """
    idxs = np.isin(labels[:, 0], [object_configuration])

    y = labels[idxs].copy()
    grasps = grasps[idxs]

    idxs_big = np.isin(y[:, 0], [1, 4, 5, 16, 20])
    y[idxs_big, 0] = 1
    idxs_medium = np.isin(y[:, 0], [2, 6, 7, 18])
    y[idxs_medium, 0] = 0.5
    idxs_small = np.isin(y[:, 0], [3, 8, 9, 17, 19])
    y[idxs_small, 0] = 0

    object_type = labels[idxs, 0].copy()
    idxs_ball = np.isin(object_type, [1, 2, 3])
    idxs_cylinder = np.isin(object_type, [4, 5, 6, 7, 8, 9])
    idxs_box = np.isin(object_type, [16, 17, 18, 19, 20])
    object_type[idxs_ball] = 1
    object_type[idxs_cylinder] = 2
    object_type[idxs_box] = 3

    object_size = y[:, 0].astype(int)
    grasp_type = y[:, 1].astype(int)

    print ("Loaded ", grasps.shape[0], " grasps!")
    return grasps, grasp_type, object_type, object_size, idxs

def annotate_grasps(grasps, labels):
    y = labels.copy()

    # idxs_big = np.isin(y[:, 0], [1, 4, 5, 8, 12, 14, 16, 20])
    # idxs_big = np.isin(y[:, 0], [1, 4, 8, 12, 14, 16, 20])
    idxs_big = np.isin(y[:, 0], [1, 4, 12, 14, 20])
    y[idxs_big, 0] = 1.0
    # idxs_medium = np.isin(y[:, 0], [2, 5, 7, 19])
    # idxs_medium = np.isin(y[:, 0], [2, 6, 13, 19])
    idxs_medium = np.isin(y[:, 0], [2, 5, 8, 16, 13, 19])
    y[idxs_medium, 0] = 0
    # idxs_small = np.isin(y[:, 0], [3, 9, 10, 11, 13, 15, 17, 18])
    # idxs_small = np.isin(y[:, 0], [3, 7, 9, 10, 11, 15, 17, 18])
    idxs_small = np.isin(y[:, 0], [3, 6, 7, 9, 10, 11, 15, 17, 18])
    y[idxs_small, 0] = -1.0

    object_type = labels[:, 0].copy()
    idxs_ball = np.isin(object_type, [1, 2, 3])
    idxs_cylinder = np.isin(object_type, [4, 5, 6, 7, 8, 9, 10])
    idxs_box = np.isin(object_type, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    object_type[idxs_ball] = 1
    object_type[idxs_cylinder] = 2
    object_type[idxs_box] = 3

    object_size = y[:, 0].astype(int)
    grasp_type = y[:, 1].astype(int)

    print ("Loaded ", grasps.shape[0], " grasps!")
    return grasps, grasp_type, object_type, object_size
