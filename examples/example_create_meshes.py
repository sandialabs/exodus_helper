
# --------------------------------------------------------------------------- #

import numpy as np

from exodus_helper import RectangularPrism


# --------------------------------------------------------------------------- #

def create_topology():

    shape = (1, 1, 1)
    resolution = (1., 1., 1.)

    num_elements = np.prod(shape)  # 1 element
    num_nodes_x = shape[0] + 1
    num_nodes_y = shape[1] + 1
    num_nodes_z = shape[2] + 1
    num_nodes = num_nodes_x * num_nodes_y * num_nodes_z

    idxs_x = np.arange(num_nodes_x)
    idxs_y = np.arange(num_nodes_y)
    idxs_z = np.arange(num_nodes_z)

    idxs = np.meshgrid(idxs_y, idxs_z, idxs_x)

    idxs_yg = idxs[0].flatten()
    idxs_zg = idxs[1].flatten()
    idxs_xg = idxs[2].flatten()

    topology = {}

    topology['coords_x'] = resolution[0] * idxs_xg
    topology['coords_y'] = resolution[1] * idxs_yg
    topology['coords_z'] = resolution[2] * idxs_zg

    # Each element has the following nodal connectivity:
    #
    #        8 ------ 7
    #       /|       /|
    #      / |      / |     z
    #     5 ------ 6  |     |  y
    #     |  4 ----|- 3     | /
    #     | /      | /       --- x
    #     |/       |/
    #     1 ------ 2

    connectivity = np.zeros((num_elements, num_nodes), dtype=int)
    num_xy = num_nodes_x * num_nodes_y
    for idx_z in range(shape[2]):
        for idx_y in range(shape[1]):
            for idx_x in range(shape[0]):
                idx = idx_z * shape[0] * shape[1] + idx_y * shape[0] + idx_x
                base = idx_y * num_nodes_x + idx_x + idx_z * num_xy + 1
                connectivity[idx, :] = [
                    base,
                    base + 1,
                    base + num_nodes_x + 1,
                    base + num_nodes_x,
                    base + num_xy,
                    base + 1 + num_xy,
                    base + num_nodes_x + 1 + num_xy,
                    base + num_nodes_x + num_xy]
    topology['connectivity'] = connectivity

    return topology


def example_create_mesh_minimal():
    """ This example creates a mesh with a single 8-noded hex element

        The single-element mesh has the minimum amount of information
        required to define a topology:

            | coordinates
            | element block
            | connectivity
    """ 

    topology = create_topology()
    coords_x = topology['coords_x']
    coords_y = topology['coords_y']
    coords_z = topology['coords_z']
    connectivity = topology['connectivity']
    num_elems, num_nodes = connectivity.shape

    mesh = RectangularPrism(
        'test_minimal.g',
        mode='w',
        num_info=1,
        num_dim=3,
        num_nodes=num_nodes,
        num_elem=num_elems)

    ids_blk = [1]
    elem_types = ['HEX8']
    define_maps = True
    nums_elems = [num_elems]
    nums_nodes = [num_nodes]
    nums_attrs = [0]

    mesh.put_coords(coords_x, coords_y, coords_z)
    mesh.put_concat_elem_blk(
        ids_blk, elem_types, nums_elems, nums_nodes, nums_attrs, define_maps)
    mesh.dataset.variables['connect1'][:] = connectivity

    return mesh


def example_create_mesh_full():
    """ This example creates a mesh with a single 8-noded hex element

        The single-element mesh has a relatively complete set of attributes:

            | coordinates
            | connectivity
            | element blocks
            | element attributes
            | node sets
            | side sets
            | times
            | global variables
            | element variables
            | nodal variables
    """

    topology = create_topology()
    coords_x = topology['coords_x']
    coords_y = topology['coords_y']
    coords_z = topology['coords_z']
    connectivity = topology['connectivity']
    num_elems, num_nodes = connectivity.shape

    num_node_nss = [2] * 6 + [1] * 8
    num_side_sss = [1] * 6

    mesh = RectangularPrism(
        'test_full.g',
        mode='w',
        num_info=1,
        time_step=1,
        num_dim=3,
        num_nodes=num_nodes,
        num_elem=num_elems,
        num_el_blk=1,
        num_node_sets=14,
        num_side_sets=6,
        num_nod_nss=num_node_nss,
        num_side_sss=num_side_sss,
        num_elem_var=1,
        num_glo_var=1,
        num_nod_var=1)

    ids_blk = [1]
    elem_types = ['HEX8']
    define_maps = True
    nums_elems = [num_elems]
    nums_nodes = [num_nodes]
    nums_attrs = [1]

    mesh.put_coords(coords_x, coords_y, coords_z)
    mesh.put_concat_elem_blk(
        ids_blk, elem_types, nums_elems, nums_nodes, nums_attrs, define_maps)
    mesh.dataset.variables['connect1'][:] = connectivity

    node_set_nodes = []
    side_set_elements = []
    side_set_sides = []
    for id_surface in range(1, 7):
        node_set_nodes.append(mesh.get_nodes_on_surface(id_surface))
        side_set_elements.append(mesh.get_elements_on_surface(id_surface))
        side_set_sides.append(mesh.get_sides_on_surface(id_surface))
    for x in range(2):
        for y in range(2, 4):
            for z in range(4, 6):
                nodes_xy = np.intersect1d(node_set_nodes[x], node_set_nodes[y])
                intersect = np.intersect1d(nodes_xy, node_set_nodes[z])
                node_set_nodes.append(intersect)

    for idx, nodes in enumerate(node_set_nodes):
        mesh.put_node_set_params(idx + 1, len(nodes))
        mesh.put_node_set(idx + 1, nodes)
        mesh.put_node_set_dist_fact(idx + 1, 1.)

    for idx in range(len(side_set_elements)):
        num_sides = len(side_set_sides[idx])
        mesh.put_side_set_params(idx + 1, num_sides)
        mesh.put_side_set(idx + 1, side_set_elements[idx], side_set_sides[idx])

    mesh.put_elem_id_map(range(1, num_elems + 1))
    mesh.put_node_id_map(range(1, num_nodes + 1))
    mesh.put_elem_attr(1, [-1])
    mesh.put_elem_variable_name('var_elem_test', 1)
    mesh.put_time(1, 1)
    mesh.put_elem_variable_values(1, 'var_elem_test', 1, [-2] * num_elems)
    mesh.put_node_variable_name('var_node_test', 1)
    mesh.put_node_variable_values('var_node_test', 1, [-1] * num_nodes)
    mesh.put_global_variable_name('momentum', 1)
    mesh.put_global_variable_value('momentum', 1, 7.5)
    return mesh

# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    mesh_minimal = example_create_mesh_minimal()
    mesh_full = example_create_mesh_full()

# --------------------------------------------------------------------------- #
