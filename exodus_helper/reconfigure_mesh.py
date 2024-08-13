"""This module contains functionality for modifying existing ExodusII meshes.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import os
import sys

import netCDF4
import numpy as np

from .core import Exodus
from .topology import RectangularPrism


# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    create_sets_canonical(sys.argv[1])

# --------------------------------------------------------------------------- #
