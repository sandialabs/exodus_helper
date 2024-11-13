"""The core module contains the Exodus class and a handful of closely related
external functions.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

from numbers import Number
import os
import time
from warnings import warn

import netCDF4
import numpy as np

from . import __version__

# --------------------------------------------------------------------------- #

SUPPORTED_DATA_MODELS = [
    'NETCDF3_64BIT_OFFSET',
    'NETCDF4_CLASSIC']

# nodes in Exodus II HEX8 element sides
CONNECTIVITY_SIDES = {
    'HEX': {
        (1, 2, 5, 6): 1,
        (2, 3, 6, 7): 2,
        (3, 4, 7, 8): 3,
        (1, 4, 5, 8): 4,
        (1, 2, 3, 4): 5,
        (5, 6, 7, 8): 6},
    'TETRA': {
        (1, 2, 4): 1,
        (2, 3, 4): 2,
        (1, 3, 4): 3,
        (1, 2, 3): 4}}

items_connectivity = CONNECTIVITY_SIDES.items()
SIDES_CONNECTIVITY = {
    k: {vv[1]: vv[0] for vv in v.items()} for k, v in items_connectivity}

NUM_NODES_ELEM = {
    'HEX': 8,
    'HEX8': 8,
    'TETRA': 4,
    'TETRA4': 4,
    'TETRA10': 10}

NUM_NODES_SIDE = {
    'HEX': 4,
    'HEX8': 4,
    'TETRA': 3,
    'TETRA4': 3,
    'TETRA10': 6}

# Global Constants
LEN_LINE_DEFAULT = 81
LEN_STRING_DEFAULT = 33
LEN_NAME_DEFAULT = 33
SET_NULL = netCDF4.Dataset(filename='NULL', mode='w', diskless=True)
GROUP_NULL = netCDF4.Group(SET_NULL, 'NULL')
DIMENSION_ZERO = netCDF4.Dimension(GROUP_NULL, 'NULL', size=0)
ARRAY_MASKED = np.ma.MaskedArray([])
ARRAY_EMPTY = ARRAY_MASKED.compressed()

# For an element with the           The face numbering is as follows:
#    nodal connectivity:                Face #: Node #s
#
#        8 ------ 7                     1: 1,2,6,5
#       /|        |                     2: 2,3,7,6
#      / |       /|     z               3: 3,4,8,7
#     5 ------ 6  |     |  y            4: 4,1,5,8
#     |  4 ----|- 3     | /             5: 4,3,2,1
#     | /      | /       --- x          6: 5,6,7,8
#     |/       |/
#     1 ------ 2


# --------------------------------------------------------------------------- #


class DatabaseError(RuntimeError):
    """`DatabaseError` is a subclass of `RuntimeError` and should be raised
    when an `Exodus` method tries to create a dimensions that already exists
    in the mesh's dataset."""


class Exodus():
    """Pure python implementation of the Exodus II standard.

    Exodus II is a finite element data model built on the netcdf database
    specification. The seacas version of this functionality interfaces directly
    with netcdf while this version uses the netcdf4 python API.

    See SAND92-2137 for details of the standard.

    The C/C++/Fortran implementation can be viewed in the `seacas`_ packages,
    which is a part of the `trilinos`_ library.


    Args:
        filename (str): Name of the binary file (.g or .e) associated with
            a mesh.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        baseName (str): Name of the binary file (.g or .e) associated with
            a mesh, but without the file suffix.
        dataset (netCDF4._netCDF4.Dataset): The netCDF4 database that underlies
            the entire mesh.
        fileName (str): Name of the binary file (.g or .e) associated with
            a mesh.
        int_type (numpy.dtype): The dataset integer type. Either `int32` or
            `int64`.
        modeChar (str): Character indicating if the mesh is in `write` ('w') or
            `read` ('r') mode.
        mode (int): int represetation of modeChar. 1: write, 2: read.

    .. _seacas:
        https://github.com/gsjaardema/seacas

    .. _trilinos:
        https://github.com/gsjaardema/trilinos
    """

    def __init__(self, filename, **kwargs):
        super().__init__()

        self.fileName = filename
        self.basename = filename.split('.')[0]

        # Set metadata to match exodus.py results
        self.comp_ws = 8
        self.io_ws = kwargs.get('io_size', 8)
        self.use_numpy = True
        self.fileId = None

        self.modeChar = kwargs.get('mode', 'r')  # 'r', 'w'
        self.mode = 2 if self.modeChar == 'r' else 1  # 1: write, 2: read

        # alias to reduce line length
        kget = kwargs.get

        if self.modeChar == 'w' and 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
            if kget('int64_status', 0) == 0:
                self.int_type = np.dtype('int32')
            else:
                self.int_type = np.dtype('int64')
            return

        # Safeguard against accidentally overwriting files
        if self.modeChar == 'w' and os.path.exists(filename):
            print(f'Opening {filename} in write mode will overwrite existing')
            decision = []
            while decision not in ['y', 'n']:
                decision = input('Do you wish to continue? [y/n]: ')
                if decision == 'y':
                    print(f'Opening {filename} in write-mode...')
                elif decision == 'n':
                    print(f'Opening {filename} in read-mode...')
                    self.modeChar = 'r'
                    self.mode = 2

        dataset = _create_dataset(filename, **kwargs)
        self.dataset = dataset

        dimensions = dataset.dimensions
        variables = dataset.variables

        self.int_type = np.dtype('int64')
        if self.modeChar == 'r' and 'int64_status' in self.dataset.ncattrs():
            if dataset.getncattr('int64_status') == 0:
                self.int_type = np.dtype('int32')
        elif self.modeChar == 'w':
            if kget('int64_status', 0) == 0:
                self.int_type = np.dtype('int32')

            # Create ncattrs, dimensions, and variables passed in as dicts
            if 'database_copy' in kwargs:
                database_copy = kwargs['database_copy']
                _set_ncattrs_from_dict(dataset, database_copy[0])
                _set_dimensions_from_dict(dataset, database_copy[1])
                _set_variables_from_dict(dataset, database_copy[2])

            else:
                # Create ncattrs
                dataset.setncattr('title', kget('title', ''))
                version_lib = kget('version', 8.2)
                assert isinstance(version_lib, float)
                dataset.setncattr('version', version_lib)
                version_api = kget('api_version', 8.2)
                assert isinstance(version_api, float)
                dataset.setncattr('api_version', version_api)
                dataset.setncattr('floating_point_word_size', 8)
                dataset.setncattr('file_size', 1)
                dataset.setncattr('maximum_name_length', 32)
                dataset.setncattr('int64_status', kget('int64_status', 0))

                # Default dimensions required by every mesh
                num_node_sets = kget('num_node_sets', 0)
                num_side_sets = kget('num_side_sets', 0)
                dict_data = {
                    'num_info': kget('num_info', 1),
                    'num_qa_rec': kget('num_qa_rec', 1),
                    'len_line': kget('len_line', LEN_LINE_DEFAULT),
                    'len_string': kget('len_string', LEN_STRING_DEFAULT),
                    'four': kget('four', 4),
                    'len_name': kget('len_name', LEN_NAME_DEFAULT),
                    'num_dim': kget('numDims', kget('num_dims', 3)),
                    'num_nodes': kget('numNodes', kget('num_nodes', 8)),
                    'num_elem': kget('numElems', kget('num_elem', 1)),
                    'num_el_blk': kget('numBlocks', kget('num_el_blk', 1)),
                    'num_node_sets': kget('numNodeSets', num_node_sets),
                    'num_side_sets': kget('numSideSets', num_side_sets),
                    'num_processors': kget('num_processors', 1),
                    'num_procs_file': kget('num_procs_file', 1),
                    'time_step': kget('time_step', 0)}

                dict_data['num_nodes_global'] = kget(
                    'num_nodes_global', dict_data['num_nodes'])
                dict_data['num_elems_global'] = kget(
                    'num_elems_global', dict_data['num_elem'])
                dict_data['num_el_blk_global'] = kget(
                    'num_el_blk_global', dict_data['num_el_blk'])
                dict_data['num_ns_global'] = kget(
                    'num_ns_global', dict_data['num_node_sets'])
                dict_data['num_ss_global'] = kget(
                    'num_ss_global', dict_data['num_side_sets'])

                # Optional dimensions
                if 'num_glo_var' in kwargs:
                    dict_data['num_glo_var'] = kwargs['num_glo_var']
                if 'num_nod_var' in kwargs:
                    dict_data['num_nod_var'] = kwargs['num_nod_var']
                if 'num_elem_var' in kwargs:
                    dict_data['num_elem_var'] = kwargs['num_elem_var']

                # Creates dataset dimensions from dict_data
                for k, v in dict_data.items():
                    if isinstance(v, Number):
                        if v > 0:
                            # time_step is given an unlimited size
                            if k == 'time_step':
                                dataset.createDimension(k, size=None)
                            else:
                                dataset.createDimension(k, size=v)

                if dict_data['time_step'] > 0:
                    dataset.createVariable(
                        'time_whole', np.dtype('float'),
                        dimensions='time_step')

                # Create variables for storing node variable values and names
                if 'num_nod_var' in dimensions and 'time_step' in dimensions:
                    dataset.createVariable(
                        'name_nod_var', np.dtype('S1'),
                        dimensions=('num_nod_var', 'len_name'))
                    dataset.createVariable(
                        'vals_nod_var', np.dtype('float'),
                        dimensions=('time_step', 'num_nod_var', 'num_nodes'))

                # num_el_in_blks and num_nodes_per_els should be lists of ints
                # Each int in the list corresponds to the size of a dimension
                num_el_in_blks = kwargs.get('num_el_in_blks', None)
                num_nodes_per_els = kwargs.get('num_nodes_per_els', None)
                if num_el_in_blks and num_nodes_per_els:
                    assert isinstance(num_el_in_blks, list)
                    assert isinstance(num_nodes_per_els, list)

                    for idx in range(dimensions['num_el_blk'].size):
                        d1 = dataset.createDimension(
                            f'num_el_in_blk{idx+1}',
                            size=num_el_in_blks[idx])
                        d2 = dataset.createDimension(
                            f'num_nod_per_el{idx+1}',
                            size=num_nodes_per_els[idx])
                        v = dataset.createVariable(
                            f'connect{idx+1}',
                            self.int_type,
                            dimensions=(d1.name, d2.name))

                        if 'elem_type' in kwargs:
                            v.setncattr('elem_type', kwargs['elem_type'])
                        elif num_nodes_per_els[idx] == 4:
                            v.setncattr('elem_type', 'TETRA')
                        elif num_nodes_per_els[idx] == 8:
                            v.setncattr('elem_type', 'HEX')
                        elif num_nodes_per_els[idx] == 10:
                            v.setncattr('elem_type', 'TETRA10')
                        else:
                            raise ValueError

                if 'num_elem_var' in dimensions:
                    dataset.createVariable(
                        'name_elem_var',
                        np.dtype('S1'),
                        dimensions=('num_elem_var', 'len_name'))
                    dataset.createVariable(
                        'elem_var_tab',
                        self.int_type,
                        dimensions=('num_el_blk', 'num_elem_var'))

                if 'num_glo_var' in dimensions:
                    dataset.createVariable(
                        'name_glo_var', np.dtype('S1'),
                        dimensions=('num_glo_var', 'len_name'))
                    if 'time_step' in dimensions:
                        dataset.createVariable(
                            'vals_glo_var', np.dtype('float'),
                            dimensions=('time_step', 'num_glo_var'))

                if 'num_node_sets' in dimensions:
                    dataset.createVariable(
                        'ns_prop1', self.int_type, dimensions='num_node_sets')
                    dataset.createVariable(
                        'ns_status', self.int_type, dimensions='num_node_sets')
                    dataset.createVariable(
                        'ns_names', np.dtype('S1'),
                        dimensions=('num_node_sets', 'len_name'))

                if 'num_side_sets' in dimensions:
                    dataset.createVariable(
                        'ss_prop1', self.int_type, dimensions='num_side_sets')
                    dataset.createVariable(
                        'ss_status', self.int_type, dimensions='num_side_sets')
                    dataset.createVariable(
                        'ss_names', np.dtype('S1'),
                        dimensions=('num_side_sets', 'len_name'))

                # Create default variables - coords
                dataset.createVariable(
                    'coordx', np.dtype('float'), dimensions='num_nodes')
                dataset.createVariable(
                    'coordy', np.dtype('float'), dimensions='num_nodes')
                dataset.createVariable(
                    'coordz', np.dtype('float'), dimensions='num_nodes')
                dataset.createVariable(
                    'coor_names', np.dtype('S1'),
                    dimensions=('num_dim', 'len_name'))
                for i in range(dimensions['num_dim'].size):
                    dataset.variables['coor_names'][i, 0] = ['x', 'y', 'z'][i]

                # element block variables
                dataset.createVariable(
                    'eb_prop1', self.int_type, dimensions='num_el_blk')
                dataset.createVariable(
                    'eb_status', self.int_type, dimensions='num_el_blk')
                dataset.createVariable(
                    'eb_names', np.dtype('S1'),
                    dimensions=('num_el_blk', 'len_name'))

                if self.get_num_blks() > 0:
                    variable = variables['eb_prop1']
                    variable[:] = np.arange(dimensions['num_el_blk'].size) + 1
                    variable.setncattr('name', 'ID')
                    variable = variables['eb_status']
                    variable[:] = [0] * dimensions['num_el_blk'].size

                # record variables
                dataset.createVariable(
                    'qa_records', np.dtype('S1'),
                    dimensions=('num_qa_rec', 'four', 'len_string'))
                dataset.createVariable(
                    'info_records', np.dtype('S1'),
                    dimensions=('num_info', 'len_line'))

                # map variables
                dataset.createVariable(
                    'elem_order_map', self.int_type,
                    dimensions='num_elem', fill_value=1)
                dataset.createVariable(
                    'node_id_map', self.int_type, dimensions='num_nodes')

                # node and side set variables
                if self.get_num_node_sets() > 0:
                    v = variables['ns_prop1']
                    v[:] = np.arange(dimensions['num_node_sets'].size) + 1
                    v.setncattr('name', 'ID')
                    v = variables['ns_status']
                    v[:] = [1] * dimensions['num_node_sets'].size

                if self.get_num_side_sets() > 0:
                    v = variables['ss_prop1']
                    # v[:] = np.arange(dimensions['num_side_sets'].size) + 1
                    v.setncattr('name', 'ID')
                    v = variables['ss_status']
                    v[:] = [0] * dimensions['num_side_sets'].size

            date_stamp = time.strftime('%Y-%m-%d')
            time_stamp = time.strftime('%H:%M:%S')
            qa_record = ('exodus_helper', __version__, date_stamp, time_stamp)
            if 'qa_records' not in variables:
                dataset.createVariable(
                    'qa_records',
                    np.dtype('S1'),
                    dimensions=('num_qa_rec', 'four', 'len_string'))
            self.put_qa_record(-1, qa_record)

    def __eq__(self, other):

        # Define dataset variables
        dataset1 = self.dataset
        dataset2 = other.dataset
        dimensions1 = dataset1.dimensions
        dimensions2 = dataset2.dimensions
        variables1 = dataset1.variables
        variables2 = dataset2.variables

        dict_ncattrs1 = {k: dataset1.getncattr(k) for k in dataset1.ncattrs()}
        dict_ncattrs2 = {k: dataset2.getncattr(k) for k in dataset2.ncattrs()}

        # Check that all ncattrs and dimensions are identical
        if not dict_ncattrs1 == dict_ncattrs2:
            return False

        if not set(dimensions1.keys()) == set(dimensions2.keys()):
            return False

        for k in dimensions1.keys():
            if k in ['num_qa_rec']:
                continue
            d_1 = dimensions1[k]
            d_2 = dimensions2[k]
            if not d_1.isunlimited() == d_2.isunlimited():
                return False
            if not d_1.size == d_2.size:
                return False

        if not set(variables1.keys()) == set(variables2.keys()):
            return False

        # Ensure all variables are identical
        for k, v_1 in variables1.items():
            if k in ['qa_records']:
                continue
            v_2 = variables2[k]
            if not v_1.dtype == v_2.dtype:
                return False
            if not v_1.size == v_2.size:
                return False
            if not v_1.dimensions == v_2.dimensions:
                return False
            if not v_1.ncattrs() == v_2.ncattrs():
                return False

            # Check that all variable values are the same
            try:
                if not np.all(v_1[:].compressed() == v_2[:].compressed()):
                    return False
            except Exception:
                return False

        return True

    # Properties ------------------------------------------------------------ #

    @property
    def idxs_elem_start_blk(self):
        """`numpy.ndarray`: The index of each element start block."""
        return self.get_idxs_elem_start_blk()

    @property
    def numDim(self):
        """int: The number of coordinate dimensions in the mesh."""
        return self.get_num_dimensions()

    @property
    def numElem(self):
        """int: The number of elements in the mesh."""
        return self.get_num_elems()

    @property
    def numElemBlk(self):
        """int: The number of element blocks in the mesh."""
        return self.get_num_blks()

    @property
    def numNodes(self):
        """int: The number of nodes in the mesh."""
        return self.get_num_nodes()

    @property
    def numNodeSets(self):
        """int: The number of node sets in the mesh."""
        return self.get_num_node_sets()

    @property
    def numSideSets(self):
        """int: The number of side sets in the mesh."""
        return self.get_num_side_sets()

    @property
    def numTimes(self):
        """int: The number of times recorded in the mesh."""
        return self.get_num_times()

    @property
    def times(self):
        """`numpy.ndarray`: The time values recorded in the mesh."""
        return self.get_times()

    def title(self):
        """str: The title of the mesh."""
        return self.get_title()

    @property
    def version(self):
        """float: The Exodus version number."""
        return self.dataset.getncattr('version')

    # exodus.py get methods not named as such ------------------------------- #

    def elem_blk_info(self, id_blk):
        """See `Exodus.get_elem_blk_info()`."""
        return self.get_elem_blk_info(id_blk)

    def elem_type(self, id_blk) -> str:
        """See `Exodus.get_elem_type()`."""
        return self.get_elem_type(id_blk)

    def num_attr(self, id_blk) -> int:
        """See `Exodus.get_num_attr()`."""
        return self.get_num_attr(id_blk)

    def num_blks(self) -> int:
        """See `Exodus.get_num_blks()`."""
        return self.get_num_blks()

    def num_dimensions(self) -> int:
        """See `Exodus.get_num_dimensions()`."""
        return self.get_num_dimensions()

    def num_elems(self) -> int:
        """See `Exodus.get_num_elems()`."""
        return self.get_num_elems()

    def num_elems_in_blk(self, id_blk) -> int:
        """See `Exodus.get_num_elems_in_blk()`."""
        return self.get_num_elems_in_blk(id_blk)

    def num_faces_in_side_set(self, id_ss) -> int:
        """See `Exodus.get_num_faces_in_side_set()`."""
        return self.get_num_faces_in_side_set(id_ss)

    def num_info_records(self) -> int:
        """See `Exodus.get_num_info_records()`."""
        return self.get_num_info_records()

    def num_node_sets(self) -> int:
        """See `Exodus.get_num_node_sets()`."""
        return self.get_num_node_sets()

    def num_nodes(self) -> int:
        """See `Exodus.get_num_nodes()`."""
        return self.get_num_nodes()

    def num_nodes_in_node_set(self, id_ns) -> int:
        """See `Exodus.get_num_nodes_in_node_set()`."""
        return self.get_num_nodes_in_node_set(id_ns)

    def num_nodes_per_elem(self, id_blk) -> int:
        """See `Exodus.get_num_nodes_per_elem()`."""
        return self.get_num_nodes_per_elem(id_blk)

    def num_side_sets(self) -> int:
        """See `Exodus.get_num_side_sets()`."""
        return self.numSideSets

    def num_times(self) -> int:
        """See `Exodus.get_num_times()`."""
        return self.get_num_times()

    def version_num(self) -> float:
        """See `Exodus.get_version_num()`."""
        return self.get_version_num()

    # exodus.py get methods named as such ----------------------------------- #

    def get_all_global_variable_values(self, step):
        """Get all global variable values at a specified time step.
        One for each global variable name, and in the order given
        by `Exodus.get_global_variable_names()`.

        Args:
            step (int): 1-based index of time steps.

        Returns:
            A `numpy.ndarray` of global variable values (floats).
        """
        num_times = self.get_num_times()
        if step < 1 or step > num_times:
            warn(f'Step {step} is not in the allowed range: [1, {num_times}]')
            return np.zeros(self.get_global_variable_number())
        try:
            glo_vals = self.dataset.variables['vals_glo_var'][step - 1].data
        except KeyError:
            print('This mesh has no global variables')
            glo_vals = ARRAY_EMPTY
        return glo_vals

    def get_all_node_set_params(self):
        warn('Method not implemented: get_all_node_set_params')

    def get_coord(self, id_node):
        """Get the model coordinates of a single node.

        Args:
            id_node (int): 1-based node index.

        Returns:
            A `tuple` of floats (coordx, coordy, coordz).
        """
        coord_x = self.dataset.variables['coordx'][id_node - 1].data
        coord_y = self.dataset.variables['coordy'][id_node - 1].data
        coord_z = self.dataset.variables['coordz'][id_node - 1].data
        return coord_x, coord_y, coord_z

    def get_coords(self):
        """Get the model coordinates for all nodes.

        Returns:
            A `tuple` of numpy arrays (coords_x, coords_y, coords_z).
        """
        coords_x = self.dataset.variables['coordx'][:].compressed()
        coords_y = self.dataset.variables['coordy'][:].compressed()
        coords_z = self.dataset.variables['coordz'][:].compressed()
        return coords_x, coords_y, coords_z

    def get_coord_names(self) -> list:
        """Get the name of each model coordinate direction.

        Returns:
            A `list` of coordinate names, e.g. ['x', 'y', 'z'].
        """
        variables = self.dataset.variables
        return [decode(variables['coor_names'][i]) for i in range(self.numDim)]

    # get_elem -------------------------------------------------------------- #

    def get_elem_attr(self, id_blk):
        """Get all the attributes of an element block.

        Args:
            id_blk (int): Element block ID.

        Returns:
            A `numpy.ndarray` of attribute values (floats).
        """
        key = f'attrib{self.get_elem_blk_idx(id_blk) + 1}'
        return self.dataset.variables.get(key, ARRAY_MASKED)[:].compressed()

    def get_elem_attr_all(self):
        """Get the attributes of every element block in the mesh.

        Returns:
            A `numpy.ndarray` of length numElemBlk. Each item in the array is
            a numpy array of attribute values for a single block.
        """
        ids_blk = self.get_elem_blk_ids()
        return np.concatenate([self.get_elem_attr(i) for i in ids_blk])

    def get_elem_attr_names(self, id_blk) -> list:
        """Get the names of each attribute in a specified element block.

        Args:
            id_blk (int): Element block ID.

        Returns:
            A `list` of attribute names (strings).
        """
        if f'attrib_name{id_blk}' in self.dataset.variables:
            var = self.dataset.variables[f'attrib_name{id_blk}']
            return [decode(var[i]) for i in range(var.shape[0])]
        return []

    def get_elem_attr_values(self, id_blk, name_elem_attr) -> list:
        """Get the named attribute value from each element in a block.

        Args:
            id_blk (int): Element block ID.
            name_elem_attr (str): Name of the element attribute.
        """
        names = self.get_element_attribute_names(id_blk)
        try:
            idx = names.index(name_elem_attr)
            attrs = self.get_elem_attr(id_blk)[idx::len(names)]
        except (AttributeError, ValueError):
            print(f'There is no attr named {name_elem_attr} in block {id_blk}')
            attrs = ARRAY_EMPTY
        return attrs[:]

    def get_elem_attr_values_all(self, name_elem_attr):
        """Get the values of every element block attribute.

        Each list in the returned array corresponds to attribute values from a
        single element block. The index of each list is equal to the ID of the
        element block it came from.

        Args:
            name_elem_attr (str): Name of the element attribute.

        Returns:
            A `numpy.ndarray` of lists.
        """
        ids_blk = self.get_elem_blk_ids()
        return np.concatenate(
            [self.get_elem_attr_values(i, name_elem_attr) for i in ids_blk])

    def get_elem_blk_ids(self):
        """ Get all the element block IDs.

        Get a mapping of exodus element block index to user- or
        application-defined element block id; elem_blk_ids is ordered
        by the element block *INDEX* ordering, a 1-based system going
        from 1 to numElemBlk(), used by exodus for storage
        and input/output of array data stored on the element blocks; a
        user or application can optionally use a separate element block
        *ID* numbering system, so the elem_blk_ids array points to the
        element block *ID* for each element block *INDEX*

        Returns:
            A `numpy.ndarray` of all 1-based element block IDs (ints)
        """
        return self.dataset.variables['eb_prop1'][:].data

    def get_elem_blk_idx(self, id_blk) -> int:
        """Get the 0-based index of the element block with a given block ID.

        Args:
            id_blk (int): Element block ID.
        """
        return list(self.get_elem_blk_ids()).index(id_blk)

    def get_elem_blk_info(self, id_blk):
        """Get the element block info of a specified element block.

        The element block info is a `tuple` of:

                1. (`str`) - Element type, e.g. 'HEX8'.
                2. (`int`) - Number of elements in the block.
                3. (`int`) - Number of nodes per element.
                4. (`int`) - Number of attributes per element.

        Args:
            id_blk (int): Element block ID.

        Returns:
            A length-4 `tuple` of element block info.
        """
        elem_type = self.elem_type(id_blk)
        num_elems_blk = self.get_num_elems_in_blk(id_blk)
        num_nodes_per_elem = self.get_num_nodes_per_elem(id_blk)
        num_attrs = self.get_num_attr(id_blk)
        return elem_type, num_elems_blk, num_nodes_per_elem, num_attrs

    def get_elem_blk_name(self, id_blk):
        """Get the element block name.

        Args:
            id_blk (int): Element block ID.
        """
        idx = self.get_elem_blk_idx(id_blk)
        return char_to_string(self.dataset.variables['eb_names'][idx])[0]

    def get_elem_blk_names(self) -> list:
        """Get the name of each element block in order of block ID."""
        return char_to_string(self.dataset.variables['eb_names'][:])

    def get_elem_centroids(self):
        """Get a numpy array of element centroids."""
        coordinates = np.column_stack(self.get_coords())
        connectivity = self.get_elem_connectivity_full()[:]
        return np.mean([coordinates[c - 1] for c in connectivity], axis=1)

    def get_elem_connectivity(self, id_blk):
        """Get the nodal connectivity, number of elements, and
        number of nodes per element for a single block."

        Args:
            id_blk (int): Element block ID.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        connect = self.dataset.variables[f'connect{idx}']
        return connect[:].compressed(), connect.shape[0], connect.shape[1]

    def get_elem_connectivity_full(self):
        """Get the nodal connectivity of every element block.

        Returns:
            A 2D `numpy.ndarray` of nodal connectivities (nodes per element).
        """
        connectivity_full = np.empty((0, 0), dtype=int)
        for id_blk in self.get_elem_blk_ids():
            connect, num_elems, num_nodes = self.get_elem_connectivity(id_blk)
            shape_1 = max(connectivity_full.shape[1], num_nodes)
            if shape_1 != connectivity_full.shape[1]:
                pad_width = ((0, 0), (0, shape_1 - connectivity_full.shape[1]))
                connectivity_full = np.pad(
                    connectivity_full, pad_width, constant_values=-1)
            connect = connect.reshape((num_elems, num_nodes))
            if shape_1 != connect.shape[1]:
                pad_width = ((0, 0), (0, shape_1 - connect.shape[1]))
                connect = np.pad(connect, pad_width, constant_values=-1)
            connectivity_full = np.concatenate((connectivity_full, connect))
        return connectivity_full

    def get_elem_id_map(self):
        """Get all the element IDs.

        Get a mapping of exodus element index to user- or application-
        defined element id; elem_id_map is ordered by the element
        *INDEX* ordering, a 1-based system going from 1 to
        numElem, used by exodus for storage and input/output
        of array data stored on the elements; a user or application
        can optionally use a separate element *ID* numbering system,
        so the elem_id_map points to the element *ID* for each
        element *INDEX*

        Returns:
            A `numpy.ndarray` of all 1-based element element IDs (ints)
        """
        if 'elem_id_map' in self.dataset.variables:
            return self.dataset.variables['elem_id_map'][:].data
        return np.arange(1, self.num_elems() + 1)

    def get_elem_num_map(self):
        """**Deprecated** use: `Exodus.get_elem_id_map()`"""
        warn(
            'This method is deprecated. Use get_elem_id_map() instead.',
            DeprecationWarning)

    def get_elem_order_map(self):
        """Get the element order mapping.

        Get a mapping of `Exodus` element indices to application-defined
        optimal ordering; `Exodus.get_elem_order_map()` is ordered by the
        element index ordering used by exodus for storage and input/output
        of array data stored on the elements; a user or application
        can optionally use a separate element ordering, e.g. for
        optimal solver performance, so the elem_order_map points to
        the index used by the application for each exodus element
        index.

        Returns:
            A `numpy.ndarray` of the element order mapping.
        """
        if 'elem_order_map' in self.dataset.variables:
            return self.dataset.variables['elem_order_map'][:].data
        return np.arange(1, self.num_elems() + 1)

    def get_elem_property_names(self):
        items = self.dataset.variables.items()
        props = [v for k, v in items if k.startswith('eb_prop')]
        return [p.getncattr('name') for p in props]

    def get_elem_property_value(self, id_blk, name):
        id_prop = self.get_elem_property_names().index(name) + 1
        props = self.dataset.variables[f'eb_prop{id_prop}']
        idx_blk = self.get_elem_blk_idx(id_blk)
        return props[idx_blk]

    def get_idx_ns(self, id_ns) -> int:
        """Get the index of a node set.

        Returns:
            Returns a 0-based index if the node set exists. If the node set
            doesn't exist, an empty numpy array is returned instead.
        """
        try:
            ns_idx = list(self.get_node_set_ids()).index(id_ns)
        except ValueError:
            print(f'This mesh has no side set whose id_ss={id_ns}')
            ns_idx = ARRAY_EMPTY
        return ns_idx

    def get_idx_ss(self, id_ss) -> int:
        """Get the index of a side set.

        Returns:
            Returns a 0-based index if the side set exists. If the side set
            doesn't exist, an empty numpy array is returned instead.
        """
        try:
            ss_idx = list(self.get_side_set_ids()).index(id_ss)
        except ValueError:
            print(f'This mesh has no side set whose id_ss={id_ss}')
            ss_idx = ARRAY_EMPTY
        return ss_idx

    def get_elem_type(self, id_blk) -> str:
        """Get the element type of an element block, e.g. 'HEX8'.

        Args:
            id_blk (int): Element block ID.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        return self.dataset.variables[f'connect{idx}'].getncattr('elem_type')

    def get_elem_variable_names(self) -> list:
        """Get the name of each element variable."""
        if 'name_elem_var' in self.dataset.variables:
            return char_to_string(self.dataset.variables['name_elem_var'][:])
        return []

    def get_elem_variable_number(self) -> int:
        """Get the number of element variable."""
        return self.dataset.dimensions.get('num_elem_var', DIMENSION_ZERO).size

    def get_elem_variable_truth_table(self):
        return self.dataset.variables.get('elem_var_tab', [])

    def get_elem_variable_values(self, id_blk, name, step):
        """Get an array of element variable values for a specified element
        block, element variable name, and time step.

        Args:
            id_blk (int): Element block ID.
            name_var (str): Element variable name.
            step (int): 1-based index of time steps.

        Returns:
            A `numpy.ndarray` of variable values (floats).
        """
        num_times = self.get_num_times()
        if step < 1 or step > num_times:
            warn(f'Step {step} is not in the allowed range: [1, {num_times}]')
            return np.zeros(self.num_elems_in_blk(id_blk))
        if name in self.dataset.variables:
            v = self.dataset.variables[name]
        else:
            try:
                name_netcdf = self.get_name_elem_variable_netcdf(name, id_blk)
                v = self.dataset.variables[name_netcdf]
            except KeyError as exc:
                raise KeyError(f'No variable named {name}') from exc
        return v[:].data[step - 1, :]

    def get_elem_variable_values_all(self, name):
        """Get an array of element variable values from each element block.

        Args:
            name_var (str): Element variable name.

        Returns:
            A 2D `numpy.ndarray` of variable values (floats).
            The shape of the return array is (numElemBlk x numTimes).
        """
        ids_blk = self.get_elem_blk_ids()
        return np.column_stack(
            [self.get_element_variable_values_block(i, name) for i in ids_blk])

    def get_elem_variable_values_block(self, id_blk, name_var):
        """Get an array of element variable values from a specified block.

        Args:
            id_blk (int): Element block ID.
            name_var (str): Element variable name.

        Returns:
            A `numpy.ndarray` of variable values for each time step (floats).
            The returned array is of length NumTimes.
        """
        name_var_netcdf = self.get_name_elem_variable_netcdf(name_var, id_blk)
        return self.dataset.variables[name_var_netcdf][:].data

    # get_element ----------------------------------------------------------- #

    def get_element_attribute(self, id_blk):
        """See `Exodus.get_elem_attr()`"""
        return self.get_elem_attr(id_blk)

    def get_element_attribute_all(self):
        """See `Exodus.get_elem_attr_all()`"""
        return self.get_elem_attr_all()

    def get_element_attribute_names(self, id_blk):
        """See `Exodus.get_elem_attr_names()`"""
        return self.get_elem_attr_names(id_blk)

    def get_element_attribute_values(self, id_blk, name_elem_attr):
        """See `Exodus.get_elem_attr_values()`"""
        return self.get_elem_attr_values(id_blk, name_elem_attr)

    def get_element_attribute_values_all(self, name_elem_attr):
        """See `Exodus.get_elem_attr_values_all()`"""
        return self.get_elem_attr_values_all(name_elem_attr)

    def get_element_blk_ids(self):
        """See `Exodus.get_elem_blk_ids()`"""
        return self.get_elem_blk_ids()

    def get_element_blk_idx(self, id_blk):
        """See `Exodus.get_elem_blk_idx()`"""
        return self.get_elem_blk_idx(id_blk)

    def get_element_blk_info(self, id_blk):
        """See `Exodus.get_elem_blk_info()`"""
        return self.get_elem_blk_info(id_blk)

    def get_element_blk_name(self, id_blk):
        """ See `Exodus.get_elem_blk_name()`"""
        return self.get_elem_blk_name(id_blk)

    def get_element_blk_names(self):
        """See `Exodus.get_elem_blk_names()`"""
        return self.get_elem_blk_names()

    def get_element_centroids(self):
        """See `Exodus.get_elem_centroids()`"""
        return self.get_elem_centroids()

    def get_element_connectivity(self, id_blk):
        """See `Exodus.get_elem_connectivity()`"""
        return self.get_elem_connectivity(id_blk)

    def get_element_connectivity_full(self):
        """See `Exodus.get_elem_connectivity_full()`"""
        return self.get_elem_connectivity_full()

    def get_element_id_map(self):
        """See `Exodus.get_elem_id_map()`"""
        return self.get_elem_id_map()

    def get_element_order_map(self):
        """See `Exodus.get_elem_order_map()`"""
        return self.get_elem_order_map()

    def get_element_property_names(self):
        return self.get_elem_property_names()

    def get_element_property_value(self, id_blk, name):
        return self.get_elem_property_value(id_blk, name)

    def get_element_type(self, id_blk):
        """See `Exodus.get_elem_type()`"""
        return self.get_elem_type(id_blk)

    def get_element_variable_names(self):
        """See `Exodus.get_elem_variable_names()`"""
        return self.get_elem_variable_names()

    def get_element_variable_number(self):
        """See `Exodus.get_elem_variable_number()`"""
        return self.get_elem_variable_number()

    def get_element_variable_truth_table(self):
        return self.get_elem_variable_truth_table()

    def get_element_variable_values(self, id_blk, name_var, step):
        """See `Exodus.get_elem_variable_values()`"""
        return self.get_elem_variable_values(id_blk, name_var, step)

    def get_element_variable_values_all(self, name_var):
        """See `Exodus.get_elem_variable_values_all()`"""
        return self.get_elem_variable_values_all(name_var)

    def get_element_variable_values_block(self, id_blk, name_var):
        """See `Exodus.get_elem_variable_values_block()`"""
        return self.get_elem_variable_values_block(id_blk, name_var)

    # get_global ------------------------------------------------------------ #

    def get_global_variable_names(self) -> list:
        """Get a list of all global variable names."""
        variables = self.dataset.variables
        if 'name_glo_var' in variables:
            return char_to_string(variables['name_glo_var'][:])
        return []

    def get_global_variable_number(self) -> int:
        """Get the number of global variables."""
        return self.dataset.dimensions.get('num_glo_var', DIMENSION_ZERO).size

    def get_global_variable_value(self, name, step) -> float:
        """Get the value of a global variable at a specific time step.

        Args:
            name (str): Name of a global variable.
            step (int): 1-based index of time steps.

        Returns:
            A `float` if the global variable exists, otherwise, an empty `list`
            is returned.
        """
        try:
            idx = self.get_global_variable_names().index(name)
            return self.dataset.variables['vals_glo_var'][step - 1][idx]
        except ValueError:
            print(f'This mesh has no global variable named {name}')
            return []

    def get_global_variable_values(self, name):
        """Get the values of a global variable at every time step.

        Args:
            name (str): Name of a global variable

        Returns:
            A `numpy.ndarray` of values with length `Exodus.numTimes`. If the
            global variable doesn't exist, an empty numpy array is returned.
        """
        try:
            idx = self.get_global_variable_names().index(name)
            return self.dataset.variables['vals_glo_var'][:, idx].data
        except (KeyError, ValueError):
            print(f'This mesh has no global variable named {name}')
            return ARRAY_EMPTY

    # get_id ---------------------------------------------------------------- #

    def get_id_map(self):
        warn('Method not implemented: get_id_map')

    def get_ids(self):
        warn('Method not implemented: get_ids')

    def get_ids_elem_in_blk(self, id_blk):
        """Get element IDs for each element element in an element block.

        Args:
            id_blk (int): Element block ID.

        Returns:
            A `numpy.ndarray` of element IDs (ints).
        """
        idxs_elem_in_blk = self.get_idxs_elem_in_blk(id_blk)
        return self.get_elem_id_map()[idxs_elem_in_blk]

    def get_idxs_elem_in_blk(self, id_blk):
        """Return the indices of the elements in a given block

        Args:
            mesh (Exodus): Mesh to interrogate
            id_blk (int): ID of element block to interrogate

        Returns:
            `np.array` of element indices.

        """
        idx_blk = self.get_elem_blk_idx(id_blk)
        begin = self.idxs_elem_start_blk[idx_blk]
        end = self.idxs_elem_start_blk[idx_blk + 1]
        return np.arange(begin, end)

    def get_idxs_elem_start_blk(self):
        """`numpy.ndarray`: The index of each element start block"""
        ids_blk = self.get_elem_blk_ids()
        return np.cumsum([0] + [self.num_elems_in_blk(i) for i in ids_blk])

    def get_info_records(self) -> list:
        """Get a list of info records where each entry in the list is one info
        record, e.g. a line of an input deck.

        Returns:
            A `list` of info records (strings).
        """
        info = self.dataset.variables['info_records'][:].data
        return [''.join([rr.decode('utf-8') for rr in r]) for r in info]

    def get_name(self):
        warn('Method not implemented: get_name')

    def get_name_elem_variable_netcdf(self, name, id_blk):
        if name in self.dataset.variables:
            return name
        idx_blk = self.get_element_blk_idx(id_blk) + 1
        try:
            idx_var = self.get_element_variable_names().index(name) + 1
        except ValueError:
            idx_var = None
        name_netcdf = f'vals_elem_var{idx_var}eb{idx_blk}'
        if name_netcdf in self.dataset.variables:
            return name_netcdf
        raise KeyError(f'No variable {name} on block {id_blk}')

    def get_name_node_variable_netcdf(self, name):
        if name in self.dataset.variables:
            return name
        try:
            idx_var = self.get_node_variable_names().index(name) + 1
        except ValueError:
            idx_var = None
        name_netcdf = f'vals_nod_var{idx_var}'
        if name_netcdf in self.dataset.variables:
            return name_netcdf
        raise KeyError(f'Could not find the variable {name}')

    def get_names(self):
        warn('Method not implemented: get_names')

    # get_node -------------------------------------------------------------- #

    def get_node_id_map(self):
        """Get the node ID map.

        Get a mapping of exodus node index to user- or application-
        defined node id; `Exodus.get_node_id_map() is ordered the same as the
        nodal coordinate arrays returned by `Exodus.get_coords()` -- this
        ordering follows the exodus node *INDEX* order, a 1-based system going
        from 1 to `Exodus.numNodes`; a user or application can optionally
        use a separate node *ID* numbering system, so the node_id_map
        points to the node *ID* for each node *INDEX*.

        Returns:
            A `numpy.array` of node IDs (ints).
        """
        if 'node_id_map' in self.dataset.variables:
            return self.dataset.variables['node_id_map'][:].data
        if 'node_num_map' in self.dataset.variables:
            return self.dataset.variables['node_num_map'][:].data
        return np.arange(self.get_num_nodes()) + 1

    def get_node_num_map(self) -> None:
        """**Deprecated** use: `Exodus.get_node_id_map()`"""
        warn(
            'This method is deprecated. Use get_node_id_map() instead.',
            DeprecationWarning)

    def get_node_set_dist_facts(self, id_ns):
        """Get the distribution factors of nodes in a nodes set.

        Args:
            id_ns (int): Node set ID.

        Returns:
            A `numpy.ndarray` of distribution factors (floats).
        """
        idx = self.get_idx_ns(id_ns) + 1
        key = f'dist_fact_ns{idx}'
        if key in self.dataset.variables:
            return self.dataset.variables[key][:].compressed()
        return ARRAY_EMPTY

    def get_node_set_ids(self):
        """Get an array of node set IDs.

        Get a mapping of exodus node set index to user- or application-
        defined node set id; node_set_ids is ordered
        by the *INDEX* ordering, a 1-based system going from
        1 to `Exodus.numNodeSets`, used by exodus for storage
        and input/output of array data stored on the node sets; a
        user or application can optionally use a separate node set
        *ID* numbering system, so the node_set_ids array points to the
        node set *ID* for each node set *INDEX*.

        Returns:
            A `numpy.ndarray` of node set IDs (ints).
        """
        ids = self.dataset.variables.get('ns_prop1', ARRAY_MASKED)
        return ids[:].compressed()

    def get_node_set_name(self, id_ns) -> str:
        """Get the name of a node set.

        Args:
            id_ns (int): Node set ID.

        Returns:
            The name of the node set (`str`), or an empty string if there is no
            node set associated with the `id_ns`.
        """
        try:
            idx = self.get_idx_ns(id_ns)
            name_chars = self.dataset.variables['ns_names'][idx]
            return char_to_string(name_chars)[0]
        except KeyError:
            print('This mesh has no node set names')
            return ''

    def get_node_set_names(self) -> list:
        """Get a list of all node set names."""
        if 'ns_names' in self.dataset.variables:
            return char_to_string(self.dataset.variables['ns_names'][:])
        print('This mesh has no node set names')
        return []

    def get_node_set_nodes(self, id_ns) -> list:
        """Get a `list` of node indicies in a node set.

        Args:
            id_ns (int): Node set ID.
        """
        idx = self.get_idx_ns(id_ns) + 1
        nodes = self.dataset.variables.get(f'node_ns{idx}', ARRAY_MASKED)
        return nodes[:].compressed()

    def get_node_set_params(self, id_ns):
        """Get the parameters of a node set.

        The parameters are a `tuple` of:

            1. (`int`) - The number of nodes.
            2. (`int`) - The number of distribution factors.

        Args:
            id_ns (int): Node set ID.

        Returns:
            A length-2 `tuple` of node set parameters.
        """
        variables = self.dataset.variables
        idx = self.get_idx_ns(id_ns) + 1
        dimensions = self.dataset.dimensions
        try:
            variable = variables[f'node_ns{idx}']
            num_nodes = dimensions[variable.dimensions[0]].size
        except KeyError:
            print(f'This mesh has no node sets at id_ns={id_ns}')
            num_nodes = ARRAY_EMPTY
        key = f'dist_fact_ns{id_ns}'
        if key in variables:
            num_dist_facts = dimensions[variables[key].dimensions[0]].size
        else:
            num_dist_facts = 0
        return num_nodes, num_dist_facts

    def get_node_set_property_names(self):
        warn('Method not implemented: get_node_set_property_names')

    def get_node_set_property_value(self):
        warn('Method not implemented: get_node_set_property_value')

    def get_node_set_variable_names(self):
        return self.dataset.variables.get('name_nset_var', [])

    def get_node_set_variable_number(self):
        return self.dataset.dimensions.get('num_nset_var', 0)

    def get_node_set_variable_truth_table(self):
        return self.dataset.variables.get('nset_var_tab', [])

    def get_node_set_variable_values(self):
        warn('Method not implemented: get_node_set_variable_values')

    def get_node_variable_names(self) -> list:
        """Get the `list` of all nodal variable names."""
        try:
            return char_to_string(self.dataset.variables['name_nod_var'][:])
        except KeyError:
            print('This mesh has no node variable names')
            return []

    def get_node_variable_number(self) -> int:
        """Get the number of nodal variables in the model."""
        return len(self.get_node_variable_names())

    def get_node_variable_values(self, name, step):
        """Get the values of a nodel variable at a given time step.

        Args:
            name (str): Name of the node variable.
            step (int): 1-based index of time steps.

        Returns:
            A `numpy.ndarray` of nodal variable values (floats).
        """
        num_times = self.get_num_times()
        if step < 1 or step > num_times:
            warn(f'Step {step} is not in the allowed range: [1, {num_times}]')
            return np.zeros(self.get_num_nodes())
        try:
            name_var_netcdf = self.get_name_node_variable_netcdf(name)
            vals = self.dataset.variables[name_var_netcdf][step - 1].data
            return np.asarray(vals, dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f'No node variable named {name}') from exc

    def get_node_variable_values_all(self, name):
        """Get the values of a nodal variable across all time steps.

        Args:
            name (str): Name of the node variable.

        Returns:
            A `numpy.ndarray` of value arrays.
        """
        name_var_netcdf = self.get_name_node_variable_netcdf(name)
        return self.dataset.variables[name_var_netcdf][:].data

    # get_num --------------------------------------------------------------- #

    def get_num_attr(self, id_blk) -> int:
        """Get the number of attributes in a specified element block.

        Args:
            id_blk (int): Element block ID.
        """
        key = f'num_att_in_blk{self.get_elem_blk_idx(id_blk) + 1}'
        return self.dataset.dimensions.get(key, DIMENSION_ZERO).size

    def get_num_blks(self) -> int:
        """Get the number of element blocks."""
        return self.dataset.dimensions.get('num_el_blk', DIMENSION_ZERO).size

    def get_num_dimensions(self) -> int:
        """Get the number of dimensions."""
        return self.dataset.dimensions.get('num_dim', DIMENSION_ZERO).size

    def get_num_elements(self) -> int:
        """See `Exodus.get_num_elems()`"""
        return self.get_num_elems()

    def get_num_elems(self) -> int:
        """Get the number of elements."""
        return self.dataset.dimensions.get('num_elem', DIMENSION_ZERO).size

    def get_num_elements_in_blk(self, id_blk) -> int:
        """See `Exodus.get_num_elems_in_blk()`"""
        return self.get_num_elems_in_blk(id_blk)

    def get_num_elems_in_blk(self, id_blk) -> int:
        """Get the number of elements in a specified element block.

        Args:
            id_blk (int): Element block ID.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        return self.dataset.variables[f'connect{idx}'].shape[0]

    def get_num_faces_in_side_set(self, id_ss) -> int:
        """Get the number of faces on a specified side set.

        Args:
            id_ss (int): Side set ID.
        """
        dimensions = self.dataset.dimensions
        return dimensions.get(f'num_side_ss{id_ss}', ARRAY_EMPTY).size

    def get_num_info_records(self) -> int:
        """Get the number of info records."""
        return self.dataset.dimensions.get('num_info', DIMENSION_ZERO).size

    def get_num_node_sets(self) -> int:
        """Get the number of node side sets."""
        dimensions = self.dataset.dimensions
        return dimensions.get('num_node_sets', DIMENSION_ZERO).size

    def get_num_nodes(self) -> int:
        """Get the number of nodes in the mesh."""
        return self.dataset.dimensions.get('num_nodes', DIMENSION_ZERO).size

    def get_num_nodes_in_node_set(self, id_ns) -> int:
        """Get the number of nodes in a specified node set.

        Args:
            id_ns (int): Node set ID.
            """
        dimensions = self.dataset.dimensions
        return dimensions.get(f'num_nod_ns{id_ns}', ARRAY_EMPTY).size

    def get_num_nodes_per_elem(self, id_blk) -> int:
        """Get the number of nodes per element.

        Args:
            id_blk (int): Element block ID.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        connect = self.dataset.variables.get(f'connect{idx}', ARRAY_EMPTY)
        try:
            num_nodes = connect.shape[1]
        except IndexError:
            print(f'This mesh has no nodes for id_blk={id_blk}')
            num_nodes = ARRAY_EMPTY
        return num_nodes

    def get_num_side_sets(self) -> int:
        """Get the number of side sets."""
        return self.dataset.dimensions.get(
            'num_side_sets', DIMENSION_ZERO).size

    def get_num_times(self) -> int:
        """Get the number of time steps."""
        return self.dataset.dimensions.get('time_step', DIMENSION_ZERO).size

    def get_num_qa_records(self) -> int:
        """Get the number of qa records."""
        return self.dataset.dimensions.get('num_qa_rec', DIMENSION_ZERO).size

    # ----------------------------------------------------------------------- #

    def get_qa_records(self) -> list:
        """Get the `list` of QA records.

        Each QA record is a length-4 `tuple` of strings:

            1. The software name that accessed/modified the database.
            2. The software descriptor, e.g. version.
            3. Additional software data.
            4. Time stamp of when the database was created.

        Returns:
            A list of length-4 tuples.
        """
        qa_records = []
        try:
            for record in self.dataset.variables['qa_records'][:]:
                soft_name = char_to_string(record[0])[0]
                version = char_to_string(record[1])[0]
                other = char_to_string(record[2])[0]
                time_stamp = char_to_string(record[3])[0]
                qa_records.append((soft_name, version, other, time_stamp))
        except KeyError:
            warn('This mesh has no qa records.')
        return qa_records

    def get_side_set(self, id_ss):
        """Get the elements and side indices in a side set.

        The i'th element of each array defines the face of the element.

        Args:
            id_ss (int): Side set ID.

        Returns:
            Two seperate numpy arrays. The first contains element IDs and the
            second contains side IDs.
        """
        idx = self.get_idx_ss(id_ss) + 1

        ids_elem_chars = self.dataset.variables.get(
            f'elem_ss{idx}', ARRAY_MASKED)
        ids_side_chars = self.dataset.variables.get(
            f'side_ss{idx}', ARRAY_MASKED)

        ids_elem = ids_elem_chars[:].compressed()
        ids_side = ids_side_chars[:].compressed()

        return ids_elem, ids_side

    def get_side_set_dist_fact(self, id_ss):
        """Get the distribution factors for nodes in a side set.

        The number of nodes (and distribution factors) in a side set is
        the sum of all face nodes. A single node can be counted more
        than once, i.e. once for each face it belongs to in the side set.

        Args:
            id_ss (int): Side set ID.

        Returns:
            A `numpy.ndarray` of distribution factors (floats).
        """
        key = f'dist_fact_ss{id_ss}'
        return self.dataset.variables.get(key, ARRAY_MASKED)[:].compressed()

    def get_side_set_ids(self):
        """ Get a numpy array of side set IDs.

        Get a mapping of exodus side set index to user- or application-
        defined side set id; side_set_ids is ordered
        by the *INDEX* ordering, a 1-based system going from
        1 to `Exodus.numSideSets`, used by exodus for storage
        and input/output of array data stored on the side sets; a
        user or application can optionally use a separate side set
        *ID* numbering system, so the side_set_ids array points to the
        side set *ID* for each side set *INDEX*.
        """
        side_set_ids = self.dataset.variables.get('ss_prop1', ARRAY_MASKED)
        return side_set_ids[:].compressed()

    def get_side_set_name(self, id_ss) -> str:
        """Get the name of a side set.

        Args:
            id_ss (int): Side set ID.
        """
        ss_names = self.dataset.variables.get('ss_names', ARRAY_EMPTY)
        idx = self.get_idx_ss(id_ss)
        try:
            name = char_to_string(ss_names[idx])[0]
        except IndexError:
            print('This mesh has no side sets')
            name = ARRAY_EMPTY
        return name

    def get_side_set_names(self) -> list:
        """Get a list of all side set names."""
        ss_names = self.dataset.variables.get('ss_names', ARRAY_EMPTY)
        return [char_to_string(name)[0] for name in ss_names]

    def get_side_set_node_list(self):
        warn('Method not implemented: get_side_set_node_list')

    def get_side_set_params(self, id_ss):
        """Get the number of sides and nodal distribution factors (e.g. nodal
        'weights') in a side set.

        The number of nodes (and distribution factors) in a side set is
        the sum of all face nodes. A single node can be counted more
        than once, i.e. once for each face it belongs to in the side set.

        Args:
            id_ss (int): Side set ID.

        Returns:
            A length-2 `tuple` containing the number of faces and number of
            dist. facts. Both values are 0 if the `id_ss` is invalid.
        """
        dimensions = self.dataset.dimensions
        variables = self.dataset.variables
        idx_ss = self.get_idx_ss(id_ss) + 1
        try:
            key = variables[f'side_ss{idx_ss}'].dimensions[0]
            num_faces = dimensions[key].size
        except KeyError:
            print(f'This mesh has no side set whose id_ss={id_ss}')
            num_faces = 0
            num_dist_facts = 0
        key = f'dist_fact_ss{idx_ss}'
        if key in variables:
            num_dist_facts = dimensions[variables[key].dimensions[0]].size
        else:
            num_dist_facts = 0
        return num_faces, num_dist_facts

    def get_side_set_property_names(self):
        items = self.dataset.variables.items()
        props = [v for k, v in items if k.startswith('ss_prop')]
        return [p.getncattr('name') for p in props]

    def get_side_set_property_value(self, id_ss, name):
        id_prop = self.get_side_set_property_names().index(name) + 1
        props = self.dataset.variables[f'ss_prop{id_prop}']
        return props[self.get_idx_ss(id_ss)]

    def get_side_set_variable_names(self):
        return self.dataset.variables.get('name_sset_var', [])

    def get_side_set_variable_number(self):
        return self.dataset.dimensions.get('num_sset_var', DIMENSION_ZERO).size

    def get_side_set_variable_truth_table(self):
        return self.dataset.variables.get('sset_var_tab', [])

    def get_side_set_variable_values(self):
        warn('Method not implemented: get_side_set_variable_values')

    def get_times(self):
        """Get a numpy array of all time values."""
        return self.dataset.variables.get(
            'time_whole', ARRAY_MASKED)[:].filled()

    def get_title(self) -> str:
        """Get the title of the mesh."""
        return self.dataset.getncattr('title')

    def get_version_num(self) -> float:
        """Get the Exodus version number."""
        return str(self.version)

    # exodus.py set methods ------------------------------------------------- #

    def set_elem_variable_number(self, num_vars):
        if 'num_elem_var' in self.dataset.dimensions:
            return self.get_elem_variable_number() == num_vars
        self.dataset.createDimension('num_elem_var', num_vars)
        dimensions = ('num_elem_var', 'len_name')
        self.dataset.createVariable(
            'name_elem_var', np.dtype('S1'), dimensions=dimensions)
        return True

    def set_element_variable_number(self, num_vars):
        return self.set_elem_variable_number(num_vars)

    def set_elem_variable_truth_table(self, table):
        self.dataset.variables['elem_var_tab'] = table
        return True

    def set_element_variable_truth_table(self, table):
        return self.set_elem_variable_truth_table(table)

    def set_global_variable_number(self, num_vars):
        if num_vars > 0:
            if 'num_glo_var' in self.dataset.dimensions:
                return self.get_global_variable_number() == num_vars
            self.dataset.createDimension('num_glo_var', num_vars)
            dimensions = ('num_glo_var', 'len_name')
            self.dataset.createVariable(
                'name_glo_var', np.dtype('S1'), dimensions=dimensions)
        return True

    def set_node_set_variable_number(self, num_vars):
        if 'num_nset_var' in self.dataset.dimensions:
            return self.get_node_set_variable_number() == num_vars
        if num_vars > 0:
            self.dataset.createDimension('num_nset_var', num_vars)
            dimensions = ('num_nset_var', 'len_name')
            self.dataset.createVariable(
                'name_nset_var', np.dtype('S1'), dimensions=dimensions)
        return True

    def set_node_set_variable_truth_table(self, table):
        if 'nset_var_tab' in self.dataset.variables:
            self.dataset.variables['nset_var_tab'][:] = table
        return True

    def set_node_variable_number(self):
        warn('Method not implemented: set_node_variable_number')

    def set_side_set_variable_number(self, num_vars):
        if num_vars > 0:
            if 'num_sset_var' in self.dataset.dimensions:
                return self.get_side_set_variable_number() == num_vars
            self.dataset.createDimension('num_sset_var', num_vars)
            dimensions = ('num_sset_var', 'len_name')
            self.dataset.createVariable(
                'name_sset_var', np.dtype('S1'), dimensions=dimensions)
        return True

    def set_side_set_variable_truth_table(self, table):
        if 'sset_var_tab' in self.dataset.variables:
            self.dataset.variables['sset_var_tab'][:] = table
        return True

    # exodus.py put methods ------------------------------------------------- #

    def put_all_global_variable_values(self, step, values) -> bool:
        """Store all global variable values (one for each global variable
        name, and in the order given by `Exodus.get_global_variable_names()`)
        at a specified time step.

        Args:
            step (int): 1-based index of time steps
            values (list(float)): List of global values.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: `values` must be a list of numbers whose length is
                equal to the number of global variables.
        """
        try:
            num_vals = self.dataset.dimensions['num_glo_var'].size

            # Ensure valid values
            assert num_vals == len(values)
            for value in values:
                assert isinstance(value, Number)

            v = self.dataset.variables['vals_glo_var']
            v[step - 1] = values
            return True
        except KeyError:
            print('This mesh contains no global variables')
            return False
        except AssertionError:
            print('len(values) must equal the number of global variables')
            raise

    def put_concat_elem_blk(
            self, ids_blk, block_elem_types, block_num_elem,
            block_num_nodes_per_elem, block_num_attributes,
            define_maps) -> bool:
        """Store the element block ID and info for all blocks at once.

        The element block info is:

                1. (`str`) - Element type, e.g. 'HEX8'.
                2. (`int`) - Number of elements in the block.
                3. (`int`) - Number of nodes per element.
                4. (`int`) - Number of attributes per element.

        Information is stored by calling `Exodus.put_elem_blk_info()` on
        each given block ID.

        This method cannot be used to create new element blocks. The number
        of element blocks is set at the time of the `Exodus` object's
        creation by passing the `num_el_blk` kwarrg to the constructor.
        The number of element blocks in the mesh defaults to 1 if no
        `num_el_blk` is given.

        Args:
            ids_blk (list(int)): List of element block IDs.
            block_elem_types (list(str)): List of element types for each block,
                e.g. 'HEX8'.
            block_num_elem (list(int)): The number of elements in each block.
            block_num_nodes_per_elem (list(int)): The number of nodes per
                element for each block.
            block_num_attributes (list(int)): The number of attributes per
                element for each block.
            define_maps (Boolean): Set the element and node ID mappings to be
                consistent with the rest of the mesh.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: All input lists must have the same length.
        """
        if define_maps:
            self.put_elem_id_map(self.get_elem_id_map())
            self.put_node_id_map(self.get_node_id_map())

        # Check if the length of each input is the same
        idx_len = len(ids_blk)
        try:
            assert len(block_elem_types) == idx_len
            assert len(block_num_elem) == idx_len
            assert len(block_num_nodes_per_elem) == idx_len
            assert len(block_num_attributes) == idx_len
        except AssertionError:
            print('All arg array lengths must be equal')
            raise

        # Loop through all the block ids and run put_elem_blk_info()
        # Element block info needs to be set sequentially
        self.dataset.variables['eb_status'][:] = 0
        for i in range(idx_len):
            self.put_elem_blk_info(
                ids_blk[i], block_elem_types[i], block_num_elem[i],
                block_num_nodes_per_elem[i], block_num_attributes[i])

            # The value to modify is the first with status equal to 0
            idx = np.argwhere(
                self.dataset.variables['eb_status'][:] == 0)[0][0]
            self.dataset.variables['eb_status'][idx] = 1
            self.dataset.variables['eb_prop1'][idx] = ids_blk[idx]
        return True

    def put_coord_names(self, coord_names) -> bool:
        """Store the names of each coordinate direction.

        Args:
            coord_names (list(str)): A list of coordinate names.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: The length of `coord_names` must be equal
                to `Exodus.numDim`.
        """
        assert len(coord_names) == self.numDim
        for idx, name in enumerate(coord_names):
            put_string(self.dataset.variables['coor_names'], name, idx=idx)
        return True

    def put_coords(self, coordx, coordy, coordz) -> bool:
        """Store the coordinates of all nodes.

        Each input list must be of length `Exodus.numNodes`.

        Args:
            coordx (list(float)): Global x-direction coordinates.
            coordy (list(float)): Global y-direction coordinates.
            coordz (list(float)): Global z-direction coordinates.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        self.dataset.variables['coordx'][:] = coordx
        self.dataset.variables['coordy'][:] = coordy
        self.dataset.variables['coordz'][:] = coordz
        return True

    # put_elem -------------------------------------------------------------- #

    def put_elem_attr(self, id_blk, attributes) -> bool:
        """Store all attributes for each element in a block.

        This method cannot be used to create a new element attribute.
        To create new attributes, call `Exodus.put_elem_blk_info()`.

        Args:
            id_blk (int): Element block ID
            attributes (list(float)): List of all attribute values for
                all elements in the block.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        attr_name = f'attrib{idx}'

        # Ensure that the attrib# variable exists
        if attr_name not in self.dataset.variables:
            print('Call put_elem_blk_info before putting an element attribute')
            return False
        v = self.dataset.variables[attr_name]
        v[:] = np.array(attributes).reshape(v[:].shape)
        return True

    def put_elem_attr_names(self, id_blk, names) -> bool:
        """Store a list of element attribute names for a block.

        Args:
            id_blk (int): Element block ID.
            names (list(str)): List of attribute names.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: The length of `names` must be equal to the number
                of attributes in the element block.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        name_d0 = f'num_att_in_blk{idx}'

        # Ensure the proper dimensions exist
        if name_d0 not in self.dataset.dimensions:
            print('Call put_elem_blk_info before putting elem attribute names')
            return False

        # Assert that there is a name for each attribute
        assert len(names) == self.dataset.dimensions[name_d0].size

        # Create a variable to store the elem blk's attribute names
        name_v = f'attrib_name{idx}'
        if name_v not in self.dataset.variables:
            self.dataset.createVariable(
                name_v, np.dtype('S1'), dimensions=(name_d0, 'len_name'))

        for idx, name in enumerate(names):
            put_string(self.dataset.variables[name_v], name, idx=idx)

        return True

    def put_elem_attr_values(self, id_blk, name_attr, values) -> bool:
        """Store an element attribute value for each element in a block.

        Args:
            id_blk (int): Element block ID.
            name_attr (str): Element attribute name.
            values (list(float)): List of attribute values.
                The length of the `values` list must be equal to the number
                of elements in the block.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            TypeError: If any of the inputs are of invalid size or type.
        """
        names = self.get_elem_attr_names(id_blk)
        try:
            # Assert the values list is the correct length
            assert len(values) == self.get_num_elems_in_blk(id_blk)
            idx = names.index(name_attr)
            self.dataset.variables[f'attrib{id_blk}'][:, idx] = values
            return True
        except ValueError as exc:
            raise KeyError(
                f'No attr named {name_attr} found in block {id_blk}') from exc
        except AssertionError as exc:
            raise TypeError('len(values) must equal number of elems') from exc

    def put_elem_blk_info(
            self, id_blk, elem_type, num_elems,
            num_nodes_per_elem, num_attrs_per_elem) -> bool:
        """Store the element block ID and element block info.

        This method cannot be used to create new element blocks. The number
        of element blocks is set at the time of the `Exodus` object's
        creation by passing the `num_el_blk` kwarg to the constructor.
        The number of element blocks in the mesh defaults to 1 if no
        `num_el_blk` is given.

        Args:
            id_blk (int): Element block ID.
            elem_type (str): Element type, e.g. 'HEX8'.
            num_elems (int): Number of elements in the block.
            num_nodes_per_elem (int): Number of nodes per element.
            num_attrs_per_elem (int): Number of attributes per element.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        dataset = self.dataset

        try:
            idx = self.get_elem_blk_idx(id_blk) + 1
        except ValueError:
            idx = list(self.dataset.variables['eb_status'][:]).index(0) + 1
            dataset.variables['eb_status'][idx - 1] = 1
            dataset.variables['eb_prop1'][idx - 1] = id_blk

        key_d1 = f'num_el_in_blk{idx}'
        if key_d1 not in dataset.dimensions:
            dataset.createDimension(key_d1, size=num_elems)
        key_d2 = f'num_nod_per_el{idx}'
        if key_d2 not in dataset.dimensions:
            dataset.createDimension(key_d2, size=num_nodes_per_elem)
        key_v1 = f'connect{idx}'
        if key_v1 not in dataset.variables:
            dataset.createVariable(
                key_v1, self.int_type, dimensions=(key_d1, key_d2))
        dataset.variables[key_v1].setncattr('elem_type', elem_type)
        if num_attrs_per_elem > 0:
            key_d3 = f'num_att_in_blk{idx}'
            if key_d3 not in dataset.dimensions:
                dataset.createDimension(key_d3, size=num_attrs_per_elem)
            key_v2 = f'attrib{idx}'
            if key_v2 not in dataset.variables:
                dataset.createVariable(
                    key_v2, np.dtype('float'), dimensions=(key_d1, key_d3))
        return True

    def put_elem_blk_name(self, id_blk, name) -> bool:
        """Store an element block name.

        Args:
            id_blk (int): Element block ID.
            name (str): Element block name.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        idx = self.get_elem_blk_idx(id_blk)
        put_string(self.dataset.variables['eb_names'], name, idx=idx)
        return True

    def put_elem_blk_names(self, names) -> bool:
        """Store a list of element block names ordered by block `index`.

        Args:
            names (list(str)): List of element block names.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        num_el_blk = self.dataset.dimensions['num_el_blk'].size
        num_names = len(names)
        try:
            assert num_names == num_el_blk
            for idx, name in enumerate(names):
                put_string(self.dataset.variables['eb_names'], name, idx=idx)
            return True
        except AssertionError:
            print(
                f'This mesh has {num_el_blk} elements but {num_names}',
                'element names were given.')
            return False

    def put_elem_connectivity(self, id_blk, connectivity) -> bool:
        """Store the nodal connectivity for an element block.

        Args:
            connectivity (numpy.ndarray(float)): A 2D array of node indices.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        idx = self.get_elem_blk_idx(id_blk) + 1
        num_elems = self.get_num_elems_in_blk(id_blk)
        num_nodes = self.get_num_nodes_per_elem(id_blk)
        connect = np.reshape(connectivity, (num_elems, num_nodes))
        self.dataset.variables[f'connect{idx}'][:] = connect
        return True

    def put_elem_id_map(self, elem_id_map) -> bool:
        """Store an element ID mapping.

        Store a mapping of exodus element index to user- or application-
        defined element id; elem_id_map is ordered by the element
        *INDEX* ordering, a 1-based system going from 1 to
        `Exodus.numElems`, used by exodus for storage and input/output
        of array data stored on the elements; a user or application
        can optionally use a separate element *ID* numbering system,
        so the elem_id_map points to the element *ID* for each
        element *INDEX*.

        Args:
            elem_id_map (list(int)): User defined element ID mapping.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        if 'elem_id_map' not in self.dataset.variables:
            self.dataset.createVariable(
                'elem_id_map', self.int_type, dimensions='num_elem')
        self.dataset.variables['elem_id_map'][:] = elem_id_map[:]
        return True

    def put_elem_property_value(self, id_blk, name, value) -> bool:
        """Store a value for an element property

        Args:
            id_blk (int): Element block ID.
            name (str): Name of the element property.
            value(scalar): Value to be set for the element property.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        id_prop = self.get_elem_property_names().index(name) + 1
        props = self.dataset.variables[f'eb_prop{id_prop}']
        idx_blk = self.get_element_blk_idx(id_blk)
        props[idx_blk] = value
        return True

    def put_elem_variable_name(self, name, idx) -> bool:
        """Store the name and index of a new element variable.

        This method creates a new element variable and must be called
        before attempting to store variable values via
        `Exodus.put_elem_variable_values()`.

        Args:
            name (str): New element variable name.
            idx (int): 1-based index of the new element variable. `idx` must be
                within 1 and the number of element variables.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            IndexError: `idx` exceeds the number of element variables.
        """

        if idx > self.get_element_variable_number():
            raise IndexError('idx exceeds the number of element variables')
        put_string(self.dataset.variables['name_elem_var'], name, idx=idx - 1)
        return True

    def put_elem_variable_values(self, id_blk, name, step, values) -> bool:
        """Store a list of element variable values for a specified block,
        element variable name, and time step.

        To store values at a non-existing time step, simply call
        `Exodus.put_time()` or `Exodus.put_times()` to store new time
        values. Then call `Exodus.put_elem_variable_values()`.

        Args:
            id_blk (int): Element block ID.
            name (str): Element variable name.
            step (int): 1-based index of time steps.
            values (list(float)): List of element variable values.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            IndexError: `step` exceeds the current number of time steps.
        """
        if name in self.dataset.variables:
            v = self.dataset.variables[name]
        else:
            try:
                name_in_dataset = self.get_name_elem_variable_netcdf(
                    name, id_blk)
                v = self.dataset.variables[name_in_dataset]
            except KeyError:
                try:
                    assert name in self.get_elem_variable_names()
                    idx_blk = self.get_elem_blk_idx(id_blk) + 1
                    idx_name = self.get_elem_variable_names().index(name) + 1
                    v = self.dataset.createVariable(
                        f'vals_elem_var{idx_name}eb{idx_blk}',
                        np.dtype('float'),
                        dimensions=('time_step', f'num_el_in_blk{idx_blk}'))
                except Exception as exc:
                    raise KeyError(f'No variable named {name}') from exc
        v[step - 1, :] = values
        return True

    # put_element ----------------------------------------------------------- #

    def put_element_attribute(self, id_blk, attributes):
        """See `Exodus.put_elem_attr()`."""
        return self.put_elem_attr(id_blk, attributes)

    def put_element_attribute_names(self, id_blk, names):
        """See `Exodus.put_elem_attr_names()`."""
        return self.put_elem_attr_names(id_blk, names)

    def put_element_attribute_values(self, id_blk, name_attr, values):
        """See `Exodus.put_elem_attr_values()`."""
        return self.put_elem_attr_values(id_blk, name_attr, values)

    def put_element_blk_info(
            self, id_blk, elem_type, num_elems, num_nodes_per_elem,
            num_attrs_per_elem):
        """See `Exodus.put_elem_blk_info()`."""
        return self.put_elem_blk_info(
            id_blk, elem_type, num_elems, num_nodes_per_elem,
            num_attrs_per_elem)

    def put_element_blk_name(self, id_blk, name):
        """See `Exodus.put_elem_blk_name()`."""
        return self.put_elem_blk_name(id_blk, name)

    def put_element_blk_names(self, names):
        """See `Exodus.put_elem_blk_names()`."""
        return self.put_elem_blk_names(names)

    def put_element_connectivity(self, id_blk, connectivity):
        """See `Exodus.put_elem_connectivity()`."""
        return self.put_elem_connectivity(id_blk, connectivity)

    def put_element_id_map(self, elem_id_map):
        """See `Exodus.put_elem_id_map()`."""
        return self.put_elem_id_map(elem_id_map)

    def put_element_property_value(self, id_blk, name, value):
        return self.put_elem_property_value(id_blk, name, value)

    def put_element_variable_name(self, name, idx):
        """See `Exodus.put_elem_variable_name()`."""
        return self.put_elem_variable_name(name, idx)

    def put_element_variable_values(self, id_blk, name, step, values):
        """See `Exodus.put_elem_variable_values()`."""
        return self.put_elem_variable_values(id_blk, name, step, values)

    # put_global ------------------------------------------------------------ #

    def put_global_variable_name(self, name, idx) -> bool:
        """Add the name and index of a new global variable.

        This method creates a new global variable and must be called
        before attempting to store global variable values via
        `Exodus.put_global_variable_value()`.

        Args:
            name (str): Name of the new global variable.
            idx (int): 1-based index of the new global variable.
                Global variable indexing goes from 1 to
                `Exodus.get_global_variable_number()`.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            IndexError: `idx` exceeds the number of global variables.
        """
        # Ensure a correct index
        if idx > self.get_global_variable_number():
            raise IndexError('index exceeds the number of global variables')

        if 'num_glo_var' not in self.dataset.dimensions:
            print('This mesh has no global variables')
            return False
        put_string(self.dataset.variables['name_glo_var'], name, idx=idx - 1)
        return True

    def put_global_variable_value(self, name, step, value) -> bool:
        """Store a global variable value for a specified global variable
        name and time step.

        Args:
            name (str): Name of the new global variable.
            step (int): 1-based index of time steps.
            value (float): Global variable value at time `step`.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        if 'num_glo_var' not in self.dataset.dimensions:
            print('This mesh has no global variables')
            return False
        name_idx = self.get_global_variable_names().index(name)
        self.dataset.variables['vals_glo_var'][step - 1, name_idx] = value
        return True

    # put_info -------------------------------------------------------------- #

    def put_info(self) -> bool:
        warn('Method not implemented: put_info')

    def put_info_records(self, info) -> bool:
        """Store static metadata for the database.

        Args:
            info (str): A string containing metadata. Info can contain any
                information as long as it is below 81 characters.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        for i in info:
            assert isinstance(i, str), 'info must be a list of strings'
            len_line = self.dataset.dimensions['len_line'].size
            assert len(i) <= len_line, f'max string length is {len_line}'

        # Cycle through info records and find empty slots to store info
        info_records = self.get_info_records()
        empty_records = 0
        empty_record_idxs = []
        for record in info_records:
            if record == '':
                empty_records = empty_records + 1
                empty_record_idxs.append(info_records.index(record))
        assert len(info) <= empty_records

        # Put info
        for i in range(len(empty_record_idxs)):
            put_string(self.dataset.variables['info_records'], info[i], idx=i)
        return True

    # put_node -------------------------------------------------------------- #

    def put_node_id_map(self, node_id_map) -> bool:
        """Store a nodal ID mapping.

        Store a mapping of exodus node index to user- or application-
        defined node id; node_id_map is ordered the same as the nodal
        coordinate arrays returned by `Exodus.get_coords()` -- this ordering
        follows the exodus node *INDEX* order, a 1-based system going
        from 1 to `Exodus.numNodes`; a user or application can optionally
        use a separate node *ID* numbering system, so the node_id_map
        points to the node *ID* for each node *INDEX*.

        Args:
            node_id_map (list(int)): A list of nodal IDs.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        self.dataset.variables['node_id_map'][:] = node_id_map
        return True

    def put_node_set(self, id_ns, ns_nodes) -> bool:
        """Store a node set with an node set ID and a list of node IDs.

        Args:
            id_ns (int): Node set ID.
            ns_nodes (list(int)): A list of node IDs comprising the node set.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        try:
            self.dataset.variables[f'node_ns{id_ns}'][:] = ns_nodes
            success = True
        except KeyError:
            print(f'There is no node set at id_ns={id_ns}')
            success = False
        return success

    def put_node_set_dist_fact(self, id_ns, node_set_dist_facts) -> bool:
        """Store the list of distribution factors for nodes in a node set.

        Args:
            id_ns (int): Node set ID.
            node_set_dist_facts (list(float)): A list of distribution factors,
                e.g. nodal 'weights'.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        name_dist_fact = f'dist_fact_ns{id_ns}'
        if name_dist_fact in self.dataset.variables:
            v = self.dataset.variables[name_dist_fact]
        else:
            name_dim = f'num_nod_ns{id_ns}'
            if name_dim not in self.dataset.dimensions:
                raise KeyError(f'No node set found with id = {id_ns}')
            v = self.dataset.createVariable(
                name_dist_fact, np.dtype('float'), dimensions=name_dim)
            print(f'This mesh has no dist_fact_ns variable at id_ns={id_ns}')
        v[:] = node_set_dist_facts

    def put_node_set_name(self, id_ns, name_ns) -> bool:
        """Store the name of a node set.

        Args:
            id_ns (int): Node set ID.
            name_ns (str): Name of the node set.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        # Assigning names must be done character by character
        idx_ns = list(self.get_node_set_ids()).index(id_ns)
        put_string(self.dataset.variables['ns_names'], name_ns, idx=idx_ns)
        return True

    def put_node_set_names(self, names_ns) -> bool:
        """Store a list of all node set names ordered by node set index.

        Args:
            names_ns (list(str)): List of node set names. The list must be
                of length `Exodus.numNodeSets`.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: `names_ns` must contain a name for each node set.
                `names_ns` must be of length `Exodus.numNodeSets`.
        """
        # Ensure names_ns has the correct length
        assert len(names_ns) == self.numNodeSets, 'Req name for each node set'

        for idx, name_ns in enumerate(names_ns):
            put_string(self.dataset.variables['ns_names'], name_ns, idx=idx)
        return True

    def put_node_set_params(self, id_ns, num_nodes, numSetDistFacts=0) -> bool:
        """Initialize a new node set.

        This function is used to create a new node set defined by the number
        of nodes. The node set nodes are left blank and the associated
        distribution factors (e.g. nodal 'weights') are set to 1.

        To store new nodes and distribution factors call
        `Exodus.put_node_set()` and `Exodus.put_node_set_dist_fact()`.

        Args:
            id_ns (int): New node set ID.
            num_nod_ns (int): Number of nodes in the new node set.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            DatabaseError: Tried to create a new node set using the `id_ns` of
                an existing node set.
        """
        try:
            dataset = self.dataset
            d = dataset.createDimension(f'num_nod_ns{id_ns}', size=num_nodes)
            dataset.createVariable(
                f'node_ns{id_ns}', self.int_type, dimensions=d.name)
            if numSetDistFacts > 0:
                d = dataset.createDimension(
                    f'num_df_ns{id_ns}', size=num_nodes)
                dataset.createVariable(
                    f'dist_fact_ns{id_ns}', np.dtype('float'),
                    dimensions=d.name)
                dataset.variables[f'dist_fact_ns{id_ns}'][:] = 1.
        except RuntimeError as exc:
            raise DatabaseError from exc
        return True

    def put_node_set_property_value(self):
        warn('Method not implemented: put_node_set_property_value')

    def put_node_set_variable_name(self):
        warn('Method not implemented: put_node_set_variable_name')

    def put_node_set_variable_values(self):
        warn('Method not implemented: put_node_set_variable_values')

    def put_node_variable_name(self, name, idx) -> bool:
        """Store the name and index of a nodal variable.

        A nodal variable must have it's name and index defined before assigning
        a value via `Exodus.put_node_variable_values()`.

        Nodal variable indexing goes from 1 to
        `Exodus.get_node_variable_number()`.

        Args:
            name (str): Name of the nodal variable.
            idx (int): 1-based index of the nodal variable.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        try:
            # strings need to be entered into the variable as characters
            v = self.dataset.variables['name_nod_var']
            put_string(v, name, idx=idx - 1)
            return True
        except KeyError as exc:
            raise KeyError('This mesh has no node variables') from exc

    def put_node_variable_values(self, name, step, values) -> bool:
        """Store a list of nodal variable values for a nodal variable
        name and time step.

        A nodal variable must have it's name and index defined via
        `Exodus.put_node_variable_name()` before any values can be assigned.

        Args:
            name (str): Name of the nodal variable.
            step (int): 1-based index of time steps.
            values (list(float)): Nodal variable values. Must be of length
                `Exodus.numNodes`.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: The `values` list must contain a value for each
                node. length of `values` must equal `Exodus.numNodes`.
        """
        # Ensure correct length of values
        assert len(values) == self.numNodes, 'Each node must be given a value'
        if name in self.dataset.variables:
            v = self.dataset.variables[name]
        else:
            try:
                name_in_dataset = self.get_name_node_variable_netcdf(name)
                v = self.dataset.variables[name_in_dataset]
            except KeyError:
                try:
                    assert name in self.get_node_variable_names()
                    v = self.dataset.createVariable(
                        name,
                        np.dtype('float'),
                        dimensions=('time_step', 'num_nodes'))
                except Exception as exc:
                    raise KeyError(f'No variable named {name}') from exc
        v[step - 1, :] = values
        return True

    def put_qa_record(self, idx, qa_record) -> bool:
        """Store a QA record consisting of a length-4 tuple of strings:

            1. The software name that accessed/modified the database
            2. The software descriptor, e.g. version
            3. Additional software data
            4. Time stamp

        Args:
            qa_record (tuple(str)): A length-4 tuple of QA record entries.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: `qa_record` must be a length-4 tuple.
            TypeError: `qa_record` must be a tuple of strings.
        """
        assert len(qa_record) == 4, 'qa_records must be a length-4 tuple'
        for entry in qa_record:
            if not isinstance(entry, str):
                raise TypeError('qa_records must be a list of strings')

        v = self.dataset.variables['qa_records']
        for idx_char, entry in enumerate(qa_record):
            put_string(v, entry, idx=(idx, idx_char))
        return True

    def put_qa_records(self, qa_records) -> bool:
        """Store a list of QA records where each QA record is a
        length-4 tuple of strings:

            1. The software name that accessed/modified the database
            2. The software descriptor, e.g. version
            3. Additional software data
            4. Time stamp

        Args:
            qa_records (list(str)): A length-4 list of QA records.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: `qa_records` must be a length-4 lists.
            TypeError: `qa_records` must be a list of strings.
        """
        num_qa_records = self.get_num_qa_records()
        assert len(qa_records) == num_qa_records, \
            f'Expected {num_qa_records} records, got {len(qa_records)}'
        for idx, qa_record in enumerate(qa_records):
            self.put_qa_record(idx, qa_record)
        return True

    # put_side -------------------------------------------------------------- #

    def put_side_set(self, id_ss, side_set_elems, side_set_sides) -> bool:
        """Store a side set by it's ID and the lists of element and side
        indices in the side set.

        Together, side_set_elems[i] and side_set_sides[i] define
        the face of an element.

        Args:
            id_ss (int): 1-based side set ID.
            side_set_elems (list(int)): List of side set element IDs.
            side_set_sides (list(int)): List of side IDs.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        idx_ss = self.get_idx_ss(id_ss) + 1
        self.dataset.variables[f'elem_ss{idx_ss}'][:] = side_set_elems
        self.dataset.variables[f'side_ss{idx_ss}'][:] = side_set_sides
        return True

    def put_side_set_dist_fact(self, id_ss, side_set_dist_facts) -> bool:
        """Store the list of distribution factors for nodes in a side set.

        The number of nodes (and distribution factors) in a side set is
        the sum of all face nodes.  A single node can be counted more
        than once, i.e. once for each face it belongs to in the side set.

        Args:
            id_ss (int): Side set ID.
            side_set_dist_facts (list(float)): List of distribution factors,
                e.g. nodal `weights`.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        if len(side_set_dist_facts) == 0:
            return True
        idx_ss = self.get_idx_ss(id_ss) + 1
        name_dist_fact = f'dist_fact_ss{idx_ss}'
        if name_dist_fact in self.dataset.variables:
            v = self.dataset.variables[name_dist_fact]
        else:
            name_dim = f'num_side_ss{idx_ss}'
            if name_dim not in self.dataset.dimensions:
                raise KeyError(f'No side set found with id = {id_ss}')
            v = self.dataset.createVariable(
                name_dist_fact,
                np.dtype('float'),
                dimensions=name_dim)
        v[:] = side_set_dist_facts
        return True

    def put_side_set_name(self, id_ss, name_ss) -> bool:
        """Store the name of a side set.

        Args:
            id_ss (int): Side set ID.
            name_ss (str): New side set name.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        idx_ss = np.where(self.get_side_set_ids() == id_ss)[0][0]
        put_string(self.dataset.variables['ss_names'], name_ss, idx=idx_ss)
        return True

    def put_side_set_names(self, names_ss) -> bool:
        """Store a list of all side set names ordered by side set index.

        Args:
            id_ss (int): Side set ID.
            names_ss (list(str)): List of side set names. The length of
                `names_ss` must be equal to `Exodus.numSideSets`.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            AssertionError: `names_ss` is not of length `Exodus.numSideSets`.
        """
        # Ensure names_ss has a name for each side set
        assert len(names_ss) == self.numSideSets, 'Name reqd for each nodeset'

        for idx, name_ss in enumerate(names_ss):
            self.dataset.variables['ss_names'][idx][:len(name_ss)] = name_ss
        return True

    def put_side_set_params(self, id_ss, num_sides, numSetDistFacts=0) -> bool:
        """Initialize a new side set with a new ID and number of sides.

        The number of nodes (and distribution factors) in a side set is
        the sum of all face nodes. A single node can be counted more
        than once, i.e. once for each face it belongs to in the side set.

        Args:
            id_ss (int): New side set ID.
            num_side_ss (int): Number of sides in the new side set.

        Returns:
            `True` if successful, otherwise returns `False`.

        Raises:
            DatabaseError: Tried to create a new side set using the `id_ss` of
                an existing side set.
        """
        dataset = self.dataset
        try:
            if id_ss in self.get_side_set_ids():
                idx_ss = self.get_idx_ss(id_ss) + 1
            else:
                idx_ss = list(dataset.variables['ss_status'][:]).index(0) + 1
                dataset.variables['ss_status'][idx_ss - 1] = 1
                dataset.variables['ss_prop1'][idx_ss - 1] = id_ss
            key_1 = f'num_side_ss{idx_ss}'
            if key_1 not in dataset.dimensions:
                dataset.createDimension(key_1, size=num_sides)
            key_2 = f'side_ss{idx_ss}'
            if key_2 not in dataset.variables:
                dataset.createVariable(key_2, self.int_type, dimensions=key_1)
            key_3 = f'elem_ss{idx_ss}'
            if key_3 not in dataset.variables:
                dataset.createVariable(key_3, self.int_type, dimensions=key_1)
            if numSetDistFacts > 0:
                key_4 = f'num_df_ss{idx_ss}'
                if key_4 not in dataset.dimensions:
                    dataset.createDimension(key_4, size=numSetDistFacts)
                dataset.createVariable(
                    f'dist_fact_ss{idx_ss}', np.dtype('float'),
                    dimensions=key_4)
                dataset.variables[f'dist_fact_ss{idx_ss}'][:] = np.ones(
                    numSetDistFacts)
        except RuntimeError as exc:
            raise DatabaseError from exc
        return True

    def put_side_set_property_value(self, id_ss, name, value):
        id_prop = self.get_side_set_property_names().index(name) + 1
        props = self.dataset.variables[f'ss_prop{id_prop}']
        props[self.get_idx_ss(id_ss)] = value
        return True

    def put_side_set_variable_name(self):
        warn('Method not implemented: put_side_set_variable_name')

    def put_side_set_variable_values(self):
        warn('Method not implemented: put_side_set_variable_values')

    def put_time(self, step, value) -> bool:
        """Store a new time.

        Args:
            step (int): 1-based index of time steps.
            value (float): Time value at the specified time step.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        # Create the time dimension and variable if it doesn't exist already
        if 'time_step' not in self.dataset.dimensions:
            self.dataset.createDimension('time_step')
            self.dataset.createVariable(
                'time_whole', np.dtype('float'), dimensions='time_step',
                fill_value=0.)
        self.dataset.variables['time_whole'][step - 1] = value
        return True

    def put_times(self, values) -> bool:
        """Store a set of time values for all time steps.

        Args:
            values (list(float)): List of time values.

        Returns:
            `True` if successful, otherwise returns `False`.
        """
        # Create the time dimension and variable if it doesn't exist already
        if 'time_step' not in self.dataset.dimensions:
            self.dataset.createDimension('time_step')
            self.dataset.createVariable(
                'time_whole', np.dtype('float'), dimensions='time_step',
                fill_value=0.)
        self.dataset.variables['time_whole'][:] = values
        return True

    # exodus.py special methods --------------------------------------------- #

    def close(self):
        """Close the mesh database."""
        self.dataset.close()
        return True

    def copy(self, filename):
        """Create a copy of the mesh.

        The .g file associated with the mesh will be created in the
        current working directory.

        Args:
            filename (str): Name of the .g file for the new mesh.
        """
        dict_attrs, dict_dimensions, dict_variables = self._get_dicts_netcdf()

        # time_step set to None so that it is created as an unlimited dimension
        dict_dimensions['time_step'] = None

        # Increase number of qa records by 1 for each new instance of the mesh
        num_qa_rec = self.get_num_qa_records() + 1
        dict_dimensions['num_qa_rec'] = num_qa_rec
        date_stamp = time.strftime('%Y-%m-%d')
        time_stamp = time.strftime('%H:%M:%S')
        if 'qa_records' in dict_variables:
            _ = dict_variables.pop('qa_records')
        qa_records = self.get_qa_records()

        database_copy = [dict_attrs, dict_dimensions, dict_variables]
        mesh_copy = Exodus(
            filename,
            int_type=self.int_type,
            mode='w',
            database_copy=database_copy)

        qa_new = ('exodus_helper', __version__, date_stamp, time_stamp)
        mesh_copy.put_qa_records(qa_records + [qa_new])

        return mesh_copy

    # Unimplemented methods ------------------------------------------------- #

    def get_all_side_set_params(self):
        warn('Method not implemented: get_all_side_set_params')

    def get_assemblies(self):
        warn('Method not implemented: get_assemblies')

    def get_assembly(self):
        warn('Method not implemented: get_assembly')

    def get_attribute_count(self):
        warn('Method not implemented: get_attribute_count')

    def get_attributes(self):
        warn('Method not implemented: get_attributes')

    def get_blob(self):
        warn('Method not implemented: get_blob')

    def get_partial_element_variable_values(self):
        warn('Method not implemented: get_partial_element_variable_values')

    def get_partial_node_set_variable_values(self):
        warn('Method not implemented: get_partial_node_set_variable_values')

    def get_partial_node_variable_values(self):
        warn('Method not implemented: get_partial_node_variable_values')

    def get_partial_side_set_variable_values(self):
        warn('Method not implemented: get_partial_side_set_variable_values')

    def get_reduction_variable_names(self):
        warn('Method not implemented: get_reduction_variable_names')

    def get_reduction_variable_number(self):
        warn('Method not implemented: get_reduction_variable_number')

    def get_reduction_variable_values(self):
        warn('Method not implemented: get_reduction_variable_values')

    def get_set_params(self):
        warn('Method not implemented: get_set_params')

    def get_variable_names(self):
        warn('Method not implemented: get_variable_names')

    def get_variable_number(self):
        warn('Method not implemented: get_variable_number')

    def get_variable_truth_table(self):
        warn('Method not implemented: get_variable_truth_table')

    def inquire(self):
        warn('Method not implemented: inquire')

    def num_assembly(self):
        warn('Method not implemented: num_assembly')

    def num_blob(self):
        warn('Method not implemented: num_blob')

    def num_qa_records(self):
        return self.get_num_qa_records()

    def put_assemblies(self):
        warn('Method not implemented: put_assemblies')

    def put_assembly(self):
        warn('Method not implemented: put_assembly')

    def put_attribute(self):
        warn('Method not implemented: put_attribute')

    def put_elem_face_conn(self):
        warn('Method not implemented: put_elem_face_conn')

    def put_face_count_per_polyhedra(self):
        warn('Method not implemented: put_face_count_per_polyhedra')

    def put_face_node_conn(self):
        warn('Method not implemented: put_face_node_conn')

    def put_id_map(self):
        warn('Method not implemented: put_id_map')

    def put_info_ext(self):
        warn('Method not implemented: put_info_ext')

    def put_name(self):
        warn('Method not implemented: put_name')

    def put_names(self):
        warn('Method not implemented: put_names')

    def put_node_count_per_face(self):
        warn('Method not implemented: put_node_count_per_face')

    def put_polyhedra_elem_blk(self):
        warn('Method not implemented: put_polyhedra_elem_blk')

    def put_polyhedra_face_blk(self):
        warn('Method not implemented: put_polyhedra_face_blk')

    def put_reduction_variable_name(self):
        warn('Method not implemented: put_reduction_variable_name')

    def put_reduction_variable_values(self):
        warn('Method not implemented: put_reduction_variable_values')

    def put_set_params(self):
        warn('Method not implemented: put_set_params')

    def put_variable_name(self):
        warn('Method not implemented: put_variable_name')

    def set_reduction_variable_number(self):
        warn('Method not implemented: set_reduction_variable_number')

    def set_variable_number(self):
        warn('Method not implemented: set_variable_number')

    def set_variable_truth_table(self):
        warn('Method not implemented: set_variable_truth_table')

    def summarize(self):
        warn('Method not implemented: summarize')

    def _get_dicts_netcdf(self):
        """Get dictionaries of all the NetCDF dataset entities.

        Args:
            mesh (Exodus): Exodus object to read from.

        Returns:
            dict_ncattrs (dict): A dictionary of the NetCDF attributes.
            dict_dimensions (dict): A dictionary of the NetCDF dimensions.
            dict_variables (dict): A dictionary of the NetCDF variables.
        """
        dataset = self.dataset
        dimensions = dataset.dimensions
        variables = dataset.variables
        dict_ncattrs = {k: dataset.getncattr(k) for k in dataset.ncattrs()}
        dict_dimensions = {k: d.size for k, d in dimensions.items()}
        dict_variables = dict(variables.items())
        return dict_ncattrs, dict_dimensions, dict_variables

    def _put_ncall(self, dict_ncattrs, dict_dimensions, dict_variables):
        self._put_ncattrs(dict_ncattrs)
        self._put_ncdims(dict_dimensions)
        self._put_ncvars(dict_variables)

    def _put_ncattrs(self, dict_ncattrs):
        for k, v in dict_ncattrs.items():
            self.dataset.setncattr(k, v)

    def _put_ncdims(self, dict_dimensions):
        for k, v in dict_dimensions.items():
            self.dataset.createDimension(k, size=v)

    def _put_ncvars(self, dict_variables):
        for k, v in dict_variables.items():
            if '_FillValue' in v.ncattrs():
                v2 = self.dataset.createVariable(
                    k, v.dtype, dimensions=v.dimensions,
                    fill_value=v[:].fill_value)
            else:
                v2 = self.dataset.createVariable(
                    k, v.dtype, dimensions=v.dimensions)
            v2.setncatts({a: v.getncattr(a) for a in v.ncattrs()})
            v2[:] = v[:]


# External Functions -------------------------------------------------------- #

def _create_dataset(filename, **kwargs):
    data_model = kwargs.get('data_model', 'NETCDF3_64BIT_OFFSET')
    assert data_model in SUPPORTED_DATA_MODELS
    modeChar = kwargs.get('mode', 'r')
    return netCDF4.Dataset(filename, format=data_model, mode=modeChar)


def _set_ncattrs_from_dict(dataset, dict_ncattrs):
    for k, v in dict_ncattrs.items():
        dataset.setncattr(k, v)


def _set_dimensions_from_dict(dataset, dict_dimensions):
    for k, v in dict_dimensions.items():
        dataset.createDimension(k, size=v)


def _set_variables_from_dict(dataset, dict_variables):
    for k, v in dict_variables.items():
        if '_FillValue' in v.ncattrs():
            v2 = dataset.createVariable(
                k, v.dtype, dimensions=v.dimensions,
                fill_value=v[:].fill_value)
        else:
            v2 = dataset.createVariable(
                k, v.dtype, dimensions=v.dimensions)
        v2.setncatts({a: v.getncattr(a) for a in v.ncattrs()})
        v2[:] = v[:]


def add_variable(mesh, filename, name, step, values, ids_blk=None):
    dataset = mesh.dataset
    data_model = dataset.data_model
    dataset_copy = _create_dataset(filename, data_model=data_model, mode='w')
    dict_ncattrs, dict_dimensions, dict_variables = mesh._get_dicts_netcdf()
    _set_ncattrs_from_dict(dataset_copy, dict_ncattrs)
    dict_dimensions['num_elem_var'] += 1
    # time_step is set to none so that it is created as an unlimited dimension
    dict_dimensions['time_step'] = None
    _set_dimensions_from_dict(dataset_copy, dict_dimensions)
    keys_to_pop = []
    for k, v in dict_variables.items():
        if 'num_elem_var' in v.dimensions:
            if '_FillValue' in v.ncattrs():
                v2 = dataset_copy.createVariable(
                    k, v.dtype, dimensions=v.dimensions,
                    fill_value=v[:].fill_value)
            else:
                v2 = dataset_copy.createVariable(
                    k, v.dtype, dimensions=v.dimensions)
            v2.setncatts({a: v.getncattr(a) for a in v.ncattrs()})
            v2[[slice(None, s, None) for s in v.shape]] = v[:]
            keys_to_pop.append(k)
    for k in keys_to_pop:
        dict_variables.pop(k)
    _set_variables_from_dict(dataset_copy, dict_variables)
    mesh_add = Exodus(filename, mode='w', dataset=dataset_copy)
    num_vars_elem = mesh_add.get_elem_variable_number()
    mesh_add.put_elem_variable_name(name, num_vars_elem)
    if ids_blk is None:
        ids_blk = mesh_add.get_elem_blk_ids()
    values = np.array(values)
    for id_blk in ids_blk:
        idxs = mesh_add.get_idxs_elem_in_blk(id_blk)
        values_blk = values[idxs]
        mesh_add.put_elem_variable_values(id_blk, name, step, values_blk)
        idx_blk = mesh.get_elem_blk_idx(id_blk)
        mesh_add.dataset.variables['elem_var_tab'][idx_blk, -1] = 1
    return mesh_add


def add_qa_record(mesh):
    _, dict_dimensions, dict_variables = mesh._get_dicts_netcdf()
    num_qa_rec = mesh.get_num_qa_records() + 1
    dict_dimensions['num_qa_rec'] = num_qa_rec
    date_stamp = time.strftime('%Y-%m-%d')
    time_stamp = time.strftime('%H:%M:%S')
    if 'qa_records' in dict_variables:
        _ = dict_variables.pop('qa_records')
    qa_records = mesh.get_qa_records()
    qa_new = ('exodus_helper', __version__, date_stamp, time_stamp)
    mesh.put_qa_records(qa_records + [qa_new])


def char_to_string(variable):
    """Converts an array of chars from a `MaskedArray` to a string.

    Args:
        variable: A database variable of type S1.

    Returns:
        str: The contents of the database variable.
    """
    variable_array = np.ma.MaskedArray(variable, ndmin=2)
    return [bytearray(v.compressed()).decode() for v in variable_array]


def get_data_exodus(mesh_name_or_object):
    if isinstance(mesh_name_or_object, str):
        mesh = Exodus(mesh_name_or_object, mode='r', array_type='numpy')
    elif isinstance(mesh_name_or_object, Exodus):
        mesh = mesh_name_or_object
    else:
        raise ValueError('Input must be filename or Exodus object')
    return mesh


def put_string(variable, string, idx=...):
    variable[idx] = netCDF4.stringtoarr(
        string, variable.shape[-1], dtype=variable.dtype.char)


def decode(string):
    """Decode strings stored in the database to a human readable string.

    Args:
        strings: Strings from the database.
    """
    return bytearray(string.compressed()).decode()

# --------------------------------------------------------------------------- #
