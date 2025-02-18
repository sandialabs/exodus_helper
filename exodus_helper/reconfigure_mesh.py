"""This module contains functionality for modifying existing ExodusII meshes.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

from itertools import permutations
import os
import sys

import netCDF4
import numpy as np

from .core import Exodus
from .topology import RectangularPrism


# --------------------------------------------------------------------------- #

IDXS_EDGES_4 = [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]


# --------------------------------------------------------------------------- #

def convert_tet4_tet10(filename_from, filename_to=None, **kwargs):
    """This converts a 4 node tetrahedral element to a 10 node tetrahedral
    element by placing additional nodes at the midpoint of every edge."""

    # Extract input mesh
    mesh_from = Exodus(filename_from)

    # Get the old connectivity and use it to start the new connectivity
    num_elements = mesh_from.get_num_elems()
    connectivity_to = np.zeros((num_elements, 10), dtype=int)
    connectivity_from = mesh_from.get_elem_connectivity_full()[:]
    connectivity_to[:, :4] = connectivity_from
    breakpoint()

    coords_from = np.stack(mesh_from.get_coords()).T
    num_nodes_from = len(coords_from)

    dims = mesh_from.dataset.dimensions
    num_ns = dims['num_ns_global'].size
    num_ss = dims['num_ss_global'].size

    num_side_sss = [dims[f'num_side_ss{n}'].size for n in range(1, num_ss + 1)]

    pairs_elem_all = {}
    coords_to = coords_from.tolist()
    for idx_elem, ids_node in enumerate(connectivity_from):
        pairs_elem = [tuple(ids_node[i]) for i in IDXS_EDGES_4]
        for idx_pair, pair in enumerate(pairs_elem):
            if pair in pairs_elem_all:
                id_node_new = pairs_elem_all[pair]
            else:
                id_node_new = num_nodes_from + len(pairs_elem_all) + 1
                pairs_elem_all[pair] = id_node_new
                idxs_old = [p - 1 for p in pair]
                coords_to.append(np.mean(coords_from[idxs_old, :], axis=0))
            connectivity_to[idx_elem, 4 + idx_pair] = id_node_new

    ids_node_set = mesh_from.get_node_set_ids()
    sets_nodes = []
    for id_node_set in ids_node_set:
        ids_node = mesh_from.get_node_set_nodes(id_node_set).tolist()
        pairs = permutations(ids_node, 2)
        for pair in pairs:
            id_new = pairs_elem_all.get(pair, pairs_elem_all.get(pair[::-1]))
            if id_new not in ids_node and id_new is not None:
                ids_node.append(id_new)
        sets_nodes.append(ids_node)

    num_nod_nss = [len(s) for s in sets_nodes]

    ids_blk = mesh_from.get_elem_blk_ids()
    define_maps = True
    nums_elems = [mesh_from.get_num_elems_in_blk(i) for i in ids_blk]
    nums_nodes = [10 for i in ids_blk]
    nums_attrs = [0 for i in ids_blk]
    elem_types = ['TETRA10' for i in ids_blk]

    dict_attrs, dict_dimensions, dict_variables = mesh_from._get_dicts_netcdf()

    if filename_to is None:
        filename_to = '_tet10'.join(os.path.splitext(filename_from))
    mesh_to = Exodus(
        filename_to,
        mode='w',
        num_info=mesh_from.get_num_info_records() + 1,
        num_dim=mesh_from.numDim,
        num_nodes=len(coords_to),
        num_elem=num_elements,
        num_el_blk=len(ids_blk),
        num_node_sets=num_ns,
        num_side_sets=num_ss,
        num_nod_nss=num_nod_nss,
        num_side_sss=num_side_sss,
        title=kwargs.get('title', mesh_from.get_title()),
        **kwargs)

    mesh_to.dataset.createDimension('time_step', size=None)

    mesh_to.put_coords(*np.array(coords_to).T)
    mesh_to.put_concat_elem_blk(
        ids_blk, elem_types, nums_elems, nums_nodes, nums_attrs, define_maps)
    variables = mesh_to.dataset.variables
    for i in ids_blk:
        idxs_elem = mesh_from.get_idxs_elem_in_blk(i)
        variables[f'connect{i}'][:] = connectivity_to[idxs_elem]

    ids_side_set = mesh_from.get_side_set_ids()
    for id_side_set in ids_side_set:
        ids_elem, ids_side = mesh_from.get_side_set(id_side_set)
        num_sides = mesh_from.get_num_faces_in_side_set(id_side_set)
        mesh_to.put_side_set_params(id_side_set, num_sides)
        mesh_to.put_side_set(id_side_set, ids_elem, ids_side)

    for id_node_set, ids_node in zip(ids_node_set, sets_nodes):
        mesh_to.put_node_set_params(id_node_set, len(ids_node))
        mesh_to.put_node_set(id_node_set, ids_node)

    return mesh_to


def create_sets_canonical(filename):
    """Create a mesh with node/side sets consistent with a canonical
    rectangular prism.

    Args:
        filename (str): Name of a .g file created in association with a mesh.

    Raises:
        AssertionError: The filename must correspond to an existing .g file.
    """

    assert os.path.isfile(filename)
    root, ext = os.path.splitext(filename)

    mesh = RectangularPrism(filename)
    dataset_from = mesh.dataset

    dataset_to = netCDF4.Dataset(
        root + '_canonical' + ext, mode='w', format=dataset_from.data_model)

    for attr in dataset_from.ncattrs():
        dataset_to.setncattr(attr, dataset_from.getncattr(attr))

    dimensions = dataset_from.dimensions
    variables = dataset_from.variables

    for k in list(dimensions.keys()):
        if k.startswith('num_side_ss'):
            dimensions.pop(k)
        elif k.startswith('num_nod_ns'):
            dimensions.pop(k)

    dimensions.pop('num_node_sets', None)
    dimensions.pop('num_ns_global', None)
    dimensions.pop('num_side_sets', None)
    dimensions.pop('num_ss_global', None)

    for k, v in dimensions.items():
        dataset_to.createDimension(k, size=v.size)

    dataset_to.createDimension('num_node_sets', size=14)
    dataset_to.createDimension('num_ns_global', size=14)
    nodesets = {}

    for i in range(1, 7):
        nodesets[i] = mesh.get_nodes_on_surface(i)
        dataset_to.createDimension(f'num_nod_ns{i}', size=len(nodesets[i]))
    for i in range(7, 15):
        dataset_to.createDimension(f'num_nod_ns{i}', size=1)

    dataset_to.createDimension('num_side_sets', size=6)
    dataset_to.createDimension('num_ss_global', size=6)

    for k in list(variables.keys()):
        if k.startswith('side_ss'):
            variables.pop(k)
        elif k.startswith('elem_ss'):
            variables.pop(k)
        elif k.startswith('dist_fact_ns'):
            variables.pop(k)
        elif k.startswith('node_ns'):
            variables.pop(k)

    variables.pop('ns_prop1', None)
    variables.pop('ns_status', None)
    variables.pop('ns_names', None)
    variables.pop('ss_prop1', None)
    variables.pop('ss_status', None)
    variables.pop('ss_names', None)

    for k, v in variables.items():
        if '_FillValue' in v.ncattrs():
            v2 = dataset_to.createVariable(
                k, v.dtype, dimensions=v.dimensions,
                fill_value=v[:].fill_value)
        else:
            v2 = dataset_to.createVariable(
                k, v.dtype, dimensions=v.dimensions)
        v2.setncatts({a: v.getncattr(a) for a in v.ncattrs()})
        v2[:] = v[:]

    if dataset_from.getncattr('int64_status') == 0:
        int_type = np.dtype('int32')
    else:
        int_type = np.dtype('int64')

    for k, nodes in nodesets.items():
        variable = dataset_to.createVariable(
            f'node_ns{k}', int_type, dimensions=f'num_nod_ns{k}')
        variable[:] = nodes

    i = 6
    for x in range(1, 3):
        for y in range(3, 5):
            for z in range(5, 7):
                i += 1
                intersect = np.intersect1d(nodesets[x], nodesets[y])
                nodes = np.intersect1d(intersect, nodesets[z])
                assert len(nodes) == 1
                variable = dataset_to.createVariable(
                    f'node_ns{i}', int_type, dimensions=f'num_nod_ns{i}')
                variable[0] = nodes[0]

    v = dataset_to.createVariable(
        'ns_prop1', int_type, dimensions='num_node_sets')
    v[:] = np.arange(dataset_to.dimensions['num_node_sets'].size) + 1
    v.setncattr('name', 'ID')

    v = dataset_to.createVariable(
        'ns_status', int_type, dimensions='num_node_sets')
    v[:] = [1] * dataset_to.dimensions['num_node_sets'].size

    dataset_to.createVariable(
        'ns_names', np.dtype('S1'), dimensions=('num_node_sets', 'len_name'))

    for i in range(1, 7):
        elems, sides = mesh.get_elements_sides_on_surface(i)
        assert len(elems) == len(sides)
        dataset_to.createDimension(f'num_side_ss{i}', size=len(elems))
        variable = dataset_to.createVariable(
            f'elem_ss{i}', int_type, dimensions=f'num_side_ss{i}')
        variable[:] = elems
        variable = dataset_to.createVariable(
            f'side_ss{i}', int_type, dimensions=f'num_side_ss{i}')
        variable[:] = sides
    mesh.close()

    v = dataset_to.createVariable(
        'ss_prop1', v.datatype, dimensions='num_side_sets')
    v[:] = np.arange(dataset_to.dimensions['num_side_sets'].size) + 1
    v.setncattr('name', 'ID')

    v = dataset_to.createVariable(
        'ss_status', v.datatype, dimensions='num_side_sets')
    v[:] = [1] * dataset_to.dimensions['num_side_sets'].size

    dataset_to.createVariable(
        'ss_names', np.dtype('S1'), dimensions=('num_side_sets', 'len_name'))

    dataset_to.close()


def scale_mesh(filename, scale=(1., 1., 1.)):
    """Scale the dimensions of a mesh by multiplying the coordinate values of
    the mesh by the scaling weights.

    Args:
        filename (str): Name of a .g file created in association with a mesh.
        scale (tuple(float), optional): A list of scaling weights.
            Defaults to (1., 1., 1.).

    Raises:
        AssertionError: The filename must correspond to an existing .g file.
    """
    assert os.path.isfile(filename)
    root, ext = os.path.splitext(filename)

    mesh = Exodus(filename)
    dataset_from = mesh.dataset

    dataset_to = netCDF4.Dataset(
        root + '_scaled' + ext, mode='w', format=dataset_from.data_model)

    for attr in dataset_from.ncattrs():
        dataset_to.setncattr(attr, dataset_from.getncattr(attr))

    dimensions = dataset_from.dimensions
    for k, v in dimensions.items():
        dataset_to.createDimension(k, size=v.size)

    variables = dataset_from.variables
    for k, v in variables.items():
        attrs = v.ncattrs()
        fargs = {}
        if '_FillValue' in attrs:
            fargs['fill_value'] = v.getncattr('_FillValue')
            attrs.remove('_FillValue')
        variable = dataset_to.createVariable(
            k, v.datatype, dimensions=v.dimensions, **fargs)
        variable[:] = v[:]
        for attr in attrs:
            variable.setncattr(attr, v.getncattr(attr))

    mesh.close()
    dataset_to.variables['coordx'][:] *= scale[0]
    dataset_to.variables['coordy'][:] *= scale[1]
    dataset_to.variables['coordz'][:] *= scale[2]
    dataset_to.close()


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    create_sets_canonical(sys.argv[1])

# --------------------------------------------------------------------------- #
