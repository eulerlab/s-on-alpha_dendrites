import networkx as nx
import numpy as np
import pandas as pd

from alphaanalysis.morphkit.preprocessing import get_edge_dict, get_path_dict, sort_path_direction, \
    reconnect_dropped_paths, find_connection, write_swc, read_swc


class Morph(object):

    def __init__(self, filepath):

        """
        Initialize Morph object. Load swc as Pandas DataFrame (df_swc). Split all paths on branch point and save as
        df_paths, related information (connection, path length, branch order etc.) are calculated. Other meta data are
        also saved into Morph Object.
        """

        self.unit = 'um'
        self.filename = filepath.split('/')[-1].split('.')[0].lower()
        self.filetype = filepath.split('/')[-1].split('.')[-1].lower()

        df_swc, df_paths = data_preprocessing(filepath)
        G = swc_to_graph(df_swc)

        self.df_swc = df_swc
        self.df_paths = df_paths
        self.G = G
        self.df_persistence_barcode = None

    def summarize(self):
        """
        Further processing df_paths and get statistics info such as path lengths, branching order into DataFrame.
        A summary of all single value morph statistics is calculatd.
        """
        self.df_paths = get_path_statistics(self.df_paths)
        df_density, density_maps = get_density_data(self.df_paths)
        self.density_maps = density_maps


def data_preprocessing(filepath):
    data = read_swc(filepath)

    e = data['e']
    n = data['n']
    t = data['t']
    pos = data['pos']
    radius = data['radius']
    soma_loc = data['soma_loc']

    edge_dict = get_edge_dict(n, e, soma_loc)
    df_paths = get_path_dict(pos, radius, t, edge_dict, soma_loc)
    df_paths, df_paths_drop = sort_path_direction(df_paths)
    df_paths = reconnect_dropped_paths(df_paths, df_paths_drop)
    df_paths = find_connection(df_paths)
    df_swc = write_swc(df_paths)

    if (df_paths.iloc[1:].type == 0).all():
        type_list = np.ones(len(df_paths)).astype(int) * 3
        type_list[0] = 1
        df_paths = df_paths.assign(type=type_list)

    return df_swc, df_paths


def swc_to_graph(df_swc):

    n = df_swc['n'].tolist()
    pos = np.array([df_swc['x'].tolist(), df_swc['y'].tolist(), df_swc['z'].tolist()]).T
    radius = df_swc['radius'].tolist()
    t = df_swc['type'].tolist()
    pid = df_swc['parent'].tolist()

    node_keys = ['pos', 'type', 'radius']
    node_data = list(zip(n,
                        [dict(zip(node_keys, [pos[ix], t[ix], radius[ix]])) for ix in range(pos.shape[0])]))
    parent_idx = np.array([n.index(pid[ix]) for ix in range(1, len(pid))])

    # edge
    ec = np.sqrt(np.sum((pos[parent_idx] - pos[1:]) ** 2, axis=1))
    edge_keys =['euclidean_dist', 'path_length']
    edge_data = list(zip(pid[1:], n[1:],
                        [dict(zip(edge_keys, [ec[ix], ec[ix]])) for ix in range(ec.shape[0])]))


    G = nx.DiGraph()
    G.add_nodes_from(node_data)
    G.add_edges_from(edge_data)

    return G


def get_path_statistics(df_paths):
    """

    Add path statistics (e.g. real/euclidean length of each path, ordering, index of paths back to soma...)

    Parameters
    ==========
    df_paths

    Returns
    =======
    a updated df_paths
    """


    df_paths = df_paths.copy()

    all_keys = df_paths.index

    real_length_dict = {}
    euclidean_length_dict = {}
    back_to_soma_dict = {}
    branch_order_dict = {}

    for path_id in all_keys:

        path = df_paths.loc[path_id].path

        real_length_dict[path_id] = get_path_real_length(path)
        euclidean_length_dict[path_id] = get_path_euclidean_length(path)
        branch_order_dict[path_id] = len(df_paths.loc[path_id].back_to_soma) - 1

    df_paths['real_length'] = pd.Series(real_length_dict)
    df_paths['euclidean_length'] = pd.Series(euclidean_length_dict)
    df_paths['branch_order'] = pd.Series(branch_order_dict)

    return df_paths


def get_density_data(df_paths):

    """
    A helper function to gether all summarized infomation

    Parameters
    ----------
    df_paths: pandas.DataFrame

    Returns
    -------
    df_density: pandas.DataFrame

    density_maps: numpy.array
        a (3, 100, 100) matrix, each layer is a density map of one neurites type
        (0: axon; 1: basal dendrites; 2: apical dendrites)

    """

    df_paths = df_paths.copy()

    soma = df_paths[df_paths.type == 1]
    axon = df_paths[df_paths.type == 2]
    dend_basal = df_paths[df_paths.type == 3]
    dend_apical = df_paths[df_paths.type == 4]

    axon_density_summary, axon_density_map = get_density_data_of_type(axon, soma)
    dend_basal_density_data, dend_basal_density_map = get_density_data_of_type(dend_basal, soma)
    dend_apical_density_data, dend_apical_density_map = get_density_data_of_type(dend_apical, soma)

    density_maps = np.zeros([3, 100, 100])
    density_maps[0] = axon_density_map
    density_maps[1] = dend_basal_density_map
    density_maps[2] = dend_apical_density_map

    labels = ['type', 'asymmetry', 'radius', 'size']
    neurites = [axon_density_summary,dend_basal_density_data,dend_apical_density_data]
    df_density = pd.DataFrame.from_records([n for n in neurites if n is not None], columns=labels)

    return df_density, density_maps


def get_density_data_of_type(neurites, soma):
    """
    A helper function to gether all summarized infomation

    Parameters
    ----------
    neurites: pandas.DataFrame
    soma: pandas.DataFrame

    Returns
    -------
    result tuple:
        (type, asymmetry, neurites_radius, neurites_size)

    Z: numpy.array
        the density map of neurites, a (100, 100) matrix

    """

    if len(neurites) < 2:
        return None, None

    import cv2
    from scipy.stats import gaussian_kde
    from scipy.spatial import ConvexHull
    from scipy.ndimage.measurements import center_of_mass

    soma_coords = soma.path.values[0].flatten()
    xy = (np.vstack(neurites.path)[:, :2] - soma_coords[:2]).T
    kernel = gaussian_kde(xy, bw_method='silverman')

    lim_max = int(np.ceil((xy.T).max() / 20) * 20)
    lim_min = int(np.floor((xy.T).min() / 20) * 20)
    lim = max(abs(lim_max), abs(lim_min))
    X, Y = np.mgrid[-lim:lim:100j, -lim:lim:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.flipud(np.rot90(np.reshape(kernel(positions).T, X.shape)))

    density_center = np.array(center_of_mass(Z))
    density_center = density_center * (2 * lim) / Z.shape[0] - lim

    asymmetry = np.sqrt(np.sum(density_center ** 2))

    hull = ConvexHull(xy.T)
    outer_terminals = xy.T[hull.vertices]
    outer_terminals = np.vstack([outer_terminals, outer_terminals[0]])
    neurites_radius = np.mean(np.sqrt(np.sum((outer_terminals - density_center) ** 2, 1)))
    neurites_size = cv2.contourArea(outer_terminals.astype(np.float32))

    if neurites.iloc[0].type == 2:
        t = 'axon'
    elif neurites.iloc[0].type == 3:
        t = 'basal_dendrites'
    elif neurites.iloc[0].type == 4:
        t = 'apical_dendrites'
    else:
        t = 'undefined'

    return (t, asymmetry, neurites_radius, neurites_size), Z


def get_path_real_length(path):

    """
    Get the dendritic length of a path, which is the sum of the distance between each consecutive points.

    Parameters
    ----------
    path: array
        a coordinate array with dim=(n, 3)

    Returns
    -------
    the dendritic length of this path: float

    """

    return np.sum(np.sqrt(np.sum((path[1:] - path[:-1])**2, 1)))

def get_path_euclidean_length(path):
    """
    get the euclidean length of a path, which is the distance between the first and last points.

    Parameters
    ----------
    path: array
        a coordinate array with dim=(n, 3)

    Returns
    -------
    the euclidean length of this path: float

    """
    return np.sqrt(np.sum((path[0] - path[-1]) ** 2))

def swc_to_df(filepath):
    df_swc =  pd.read_csv(filepath, sep='\s+', comment='#',
                          names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False)
    df_swc.index = df_swc.n.values
    return df_swc