
# --------------------------------------------------------------------------- #

import numpy as np

# --------------------------------------------------------------------------- #

def calculate_volume_element(coords, shape_functions='trilinear'):
    """Calculate the volume of an element from a list of nodal coordinates.

    Args:
        coords (list(list(float))): A length-8 (HEX) list of nodal
            coordinate values.
        shape_functions (str, optional): The name of the shape function used to 
            interpolate the element's volume from a list of discrete nodal 
            coordinate values. Defaults to `trilinear`.

    Returns:
        float: The volume of the element.

    Raises:
        Exception: `trilinear` is currently the only supported shape function.
    """
    coords = np.array(coords)
    if shape_functions == 'trilinear':
        v_12 = coords[0, 2] * coords[1, 1] * coords[2, 0] \
            - coords[0, 1] * coords[1, 2] * coords[2, 0] \
            - coords[0, 2] * coords[1, 0] * coords[2, 1] \
            + coords[0, 0] * coords[1, 2] * coords[2, 1] \
            + coords[0, 1] * coords[1, 0] * coords[2, 2] \
            - coords[0, 0] * coords[1, 1] * coords[2, 2] \
            + coords[0, 2] * coords[1, 1] * coords[3, 0] \
            - coords[0, 1] * coords[1, 2] * coords[3, 0] \
            + coords[0, 2] * coords[2, 1] * coords[3, 0] \
            + coords[1, 2] * coords[2, 1] * coords[3, 0] \
            - coords[0, 1] * coords[2, 2] * coords[3, 0] \
            - coords[1, 1] * coords[2, 2] * coords[3, 0] \
            - coords[0, 2] * coords[1, 0] * coords[3, 1] \
            + coords[0, 0] * coords[1, 2] * coords[3, 1] \
            - coords[0, 2] * coords[2, 0] * coords[3, 1] \
            - coords[1, 2] * coords[2, 0] * coords[3, 1] \
            + coords[0, 0] * coords[2, 2] * coords[3, 1] \
            + coords[1, 0] * coords[2, 2] * coords[3, 1] \
            + coords[0, 1] * coords[1, 0] * coords[3, 2] \
            - coords[0, 0] * coords[1, 1] * coords[3, 2] \
            + coords[0, 1] * coords[2, 0] * coords[3, 2] \
            + coords[1, 1] * coords[2, 0] * coords[3, 2] \
            - coords[0, 0] * coords[2, 1] * coords[3, 2] \
            - coords[1, 0] * coords[2, 1] * coords[3, 2] \
            - coords[0, 2] * coords[1, 1] * coords[4, 0] \
            + coords[0, 1] * coords[1, 2] * coords[4, 0] \
            + coords[0, 2] * coords[3, 1] * coords[4, 0] \
            - coords[0, 1] * coords[3, 2] * coords[4, 0] \
            + coords[0, 2] * coords[1, 0] * coords[4, 1] \
            - coords[0, 0] * coords[1, 2] * coords[4, 1] \
            - coords[0, 2] * coords[3, 0] * coords[4, 1] \
            + coords[0, 0] * coords[3, 2] * coords[4, 1] \
            - coords[0, 1] * coords[1, 0] * coords[4, 2] \
            + coords[0, 0] * coords[1, 1] * coords[4, 2] \
            + coords[0, 1] * coords[3, 0] * coords[4, 2] \
            - coords[0, 0] * coords[3, 1] * coords[4, 2] \
            - coords[0, 2] * coords[1, 1] * coords[5, 0] \
            + coords[0, 1] * coords[1, 2] * coords[5, 0] \
            - coords[1, 2] * coords[2, 1] * coords[5, 0] \
            + coords[1, 1] * coords[2, 2] * coords[5, 0] \
            + coords[0, 2] * coords[4, 1] * coords[5, 0] \
            + coords[1, 2] * coords[4, 1] * coords[5, 0] \
            - coords[0, 1] * coords[4, 2] * coords[5, 0] \
            - coords[1, 1] * coords[4, 2] * coords[5, 0] \
            + coords[0, 2] * coords[1, 0] * coords[5, 1] \
            - coords[0, 0] * coords[1, 2] * coords[5, 1] \
            + coords[1, 2] * coords[2, 0] * coords[5, 1] \
            - coords[1, 0] * coords[2, 2] * coords[5, 1] \
            - coords[0, 2] * coords[4, 0] * coords[5, 1] \
            - coords[1, 2] * coords[4, 0] * coords[5, 1] \
            + coords[0, 0] * coords[4, 2] * coords[5, 1] \
            + coords[1, 0] * coords[4, 2] * coords[5, 1] \
            - coords[0, 1] * coords[1, 0] * coords[5, 2] \
            + coords[0, 0] * coords[1, 1] * coords[5, 2] \
            - coords[1, 1] * coords[2, 0] * coords[5, 2] \
            + coords[1, 0] * coords[2, 1] * coords[5, 2] \
            + coords[0, 1] * coords[4, 0] * coords[5, 2] \
            + coords[1, 1] * coords[4, 0] * coords[5, 2] \
            - coords[0, 0] * coords[4, 1] * coords[5, 2] \
            - coords[1, 0] * coords[4, 1] * coords[5, 2] \
            - coords[1, 2] * coords[2, 1] * coords[6, 0] \
            + coords[1, 1] * coords[2, 2] * coords[6, 0] \
            - coords[2, 2] * coords[3, 1] * coords[6, 0] \
            + coords[2, 1] * coords[3, 2] * coords[6, 0] \
            + coords[1, 2] * coords[5, 1] * coords[6, 0] \
            + coords[2, 2] * coords[5, 1] * coords[6, 0] \
            - coords[4, 2] * coords[5, 1] * coords[6, 0] \
            - coords[1, 1] * coords[5, 2] * coords[6, 0] \
            - coords[2, 1] * coords[5, 2] * coords[6, 0] \
            + coords[4, 1] * coords[5, 2] * coords[6, 0] \
            + coords[1, 2] * coords[2, 0] * coords[6, 1] \
            - coords[1, 0] * coords[2, 2] * coords[6, 1] \
            + coords[2, 2] * coords[3, 0] * coords[6, 1] \
            - coords[2, 0] * coords[3, 2] * coords[6, 1] \
            - coords[1, 2] * coords[5, 0] * coords[6, 1] \
            - coords[2, 2] * coords[5, 0] * coords[6, 1] \
            + coords[4, 2] * coords[5, 0] * coords[6, 1] \
            + coords[1, 0] * coords[5, 2] * coords[6, 1] \
            + coords[2, 0] * coords[5, 2] * coords[6, 1] \
            - coords[4, 0] * coords[5, 2] * coords[6, 1] \
            - coords[1, 1] * coords[2, 0] * coords[6, 2] \
            + coords[1, 0] * coords[2, 1] * coords[6, 2] \
            - coords[2, 1] * coords[3, 0] * coords[6, 2] \
            + coords[2, 0] * coords[3, 1] * coords[6, 2] \
            + coords[1, 1] * coords[5, 0] * coords[6, 2] \
            + coords[2, 1] * coords[5, 0] * coords[6, 2] \
            - coords[4, 1] * coords[5, 0] * coords[6, 2] \
            - coords[1, 0] * coords[5, 1] * coords[6, 2] \
            - coords[2, 0] * coords[5, 1] * coords[6, 2] \
            + coords[4, 0] * coords[5, 1] * coords[6, 2] \
            + coords[0, 2] * coords[3, 1] * coords[7, 0] \
            - coords[2, 2] * coords[3, 1] * coords[7, 0] \
            - coords[0, 1] * coords[3, 2] * coords[7, 0] \
            + coords[2, 1] * coords[3, 2] * coords[7, 0] \
            - coords[0, 2] * coords[4, 1] * coords[7, 0] \
            - coords[3, 2] * coords[4, 1] * coords[7, 0] \
            + coords[0, 1] * coords[4, 2] * coords[7, 0] \
            + coords[3, 1] * coords[4, 2] * coords[7, 0] \
            - coords[4, 2] * coords[5, 1] * coords[7, 0] \
            + coords[4, 1] * coords[5, 2] * coords[7, 0] \
            + coords[2, 2] * coords[6, 1] * coords[7, 0] \
            + coords[3, 2] * coords[6, 1] * coords[7, 0] \
            - coords[4, 2] * coords[6, 1] * coords[7, 0] \
            - coords[5, 2] * coords[6, 1] * coords[7, 0] \
            - coords[2, 1] * coords[6, 2] * coords[7, 0] \
            - coords[3, 1] * coords[6, 2] * coords[7, 0] \
            + coords[4, 1] * coords[6, 2] * coords[7, 0] \
            + coords[5, 1] * coords[6, 2] * coords[7, 0] \
            - coords[0, 2] * coords[3, 0] * coords[7, 1] \
            + coords[2, 2] * coords[3, 0] * coords[7, 1] \
            + coords[0, 0] * coords[3, 2] * coords[7, 1] \
            - coords[2, 0] * coords[3, 2] * coords[7, 1] \
            + coords[0, 2] * coords[4, 0] * coords[7, 1] \
            + coords[3, 2] * coords[4, 0] * coords[7, 1] \
            - coords[0, 0] * coords[4, 2] * coords[7, 1] \
            - coords[3, 0] * coords[4, 2] * coords[7, 1] \
            + coords[4, 2] * coords[5, 0] * coords[7, 1] \
            - coords[4, 0] * coords[5, 2] * coords[7, 1] \
            - coords[2, 2] * coords[6, 0] * coords[7, 1] \
            - coords[3, 2] * coords[6, 0] * coords[7, 1] \
            + coords[4, 2] * coords[6, 0] * coords[7, 1] \
            + coords[5, 2] * coords[6, 0] * coords[7, 1] \
            + coords[2, 0] * coords[6, 2] * coords[7, 1] \
            + coords[3, 0] * coords[6, 2] * coords[7, 1] \
            - coords[4, 0] * coords[6, 2] * coords[7, 1] \
            - coords[5, 0] * coords[6, 2] * coords[7, 1] \
            + coords[0, 1] * coords[3, 0] * coords[7, 2] \
            - coords[2, 1] * coords[3, 0] * coords[7, 2] \
            - coords[0, 0] * coords[3, 1] * coords[7, 2] \
            + coords[2, 0] * coords[3, 1] * coords[7, 2] \
            - coords[0, 1] * coords[4, 0] * coords[7, 2] \
            - coords[3, 1] * coords[4, 0] * coords[7, 2] \
            + coords[0, 0] * coords[4, 1] * coords[7, 2] \
            + coords[3, 0] * coords[4, 1] * coords[7, 2] \
            - coords[4, 1] * coords[5, 0] * coords[7, 2] \
            + coords[4, 0] * coords[5, 1] * coords[7, 2] \
            + coords[2, 1] * coords[6, 0] * coords[7, 2] \
            + coords[3, 1] * coords[6, 0] * coords[7, 2] \
            - coords[4, 1] * coords[6, 0] * coords[7, 2] \
            - coords[5, 1] * coords[6, 0] * coords[7, 2] \
            - coords[2, 0] * coords[6, 1] * coords[7, 2] \
            - coords[3, 0] * coords[6, 1] * coords[7, 2] \
            + coords[4, 0] * coords[6, 1] * coords[7, 2] \
            + coords[5, 0] * coords[6, 1] * coords[7, 2]
    else:
        raise Exception('Only shape_functions=\'trilinear\' is supported')
    return v_12 / 12.

def calculate_volume_hexahedron(coords):
    """Calculate the volume of a hexahedral mesh.

    Args:
        coords (list(list(float))): A length-8 (HEX) list of nodal
            coordinate values.

    Returns:
        numpy.float64: The volume of the hexahedron.

    Raises:
        AssertionError: The coordinates are not compatible with right-hand 
            positive winding.
    """
    centroid = np.mean(coords, axis=0)
    coordinates = np.row_stack((centroid, coords))
    ts = np.array([
        [1, 5, 2], [6, 2, 5], [2, 6, 3], [7, 3, 6], [3, 7, 4], [8, 4, 7],
        [1, 4, 5], [8, 5, 4], [1, 2, 4], [3, 4, 2], [6, 5, 7], [8, 7, 5]],
        dtype=int)

    volume = 0.
    for i in range(12):
        edge_1 = np.array(
            coordinates[ts[i, 1]]) - np.array(coordinates[ts[i, 0]])
        edge_2 = np.array(
            coordinates[ts[i, 2]]) - np.array(coordinates[ts[i, 0]])
        edge_3 = np.array(coordinates[0]) - np.array(coordinates[ts[i, 0]])
        tm = np.column_stack((edge_1, edge_2, edge_3))
        test = np.linalg.det(tm) >= 0.
        assert test, 'Algorithm assumes right-hand positive winding'
        volume += np.linalg.det(tm) / 6

    return volume

def calculate_volumes_element(mesh, id_blk):
    """Calculate the volume of each element in a specified element block within 
    a given mesh.

    Args:
        mesh (Exodus): The `Exodus` instance of a mesh.
        id_blk (int): Element block ID.

    Returns:
        numpy.ndarray(float): An array of element volumes ordered by element ID
    """
    connectivity, num_elems, num_nodes = mesh.get_elem_connectivity(id_blk)
    connectivity = connectivity.reshape((num_elems, num_nodes))
    coords_node = np.column_stack(mesh.get_coords())
    volumes_element = np.array(
        [calculate_volume_element(coords_node[ns - 1]) for ns in connectivity])
    return volumes_element

def calculate_volumes_block(mesh):
    """Calculate the voluemes of each element block in a mesh.

    Args:
        mesh (Exodus): The `Exodus` instance of a mesh.

    Returns:
        numpy.ndarray(float): An array of element volumes ordered by element 
        block ID.
    """
    ids_blk = mesh.get_elem_blk_ids()
    volumes_blk = np.zeros(len(ids_blk))
    for idx, id_blk in enumerate(ids_blk):
        volumes_blk[idx] = np.sum(calculate_volumes_element(mesh, id_blk))
    return volumes_blk

# --------------------------------------------------------------------------- #
