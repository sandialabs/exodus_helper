"""This module contains the unit tests for the exodus_helper package.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import os
import time

import numpy as np
import pytest

try:  # Get function names for testing
    from exodus import exodus
    _dir_exodus_full = list(exodus.__dict__)
except ImportError:
    from exodus_helper.dir_exodus import _dir_exodus_full
import exodus_helper
from exodus_helper.dir_exodus import _attr_inputs
from exodus_helper.reconfigure_mesh import IDXS_EDGES_4


# --------------------------------------------------------------------------- #

# Testing the core module --------------------------------------------------- #

def test_getters(dir_test_file, monkeypatch):
    """`test_getters` copies the `test_full.g` mesh and iterates
    through each of its `get` methods, which are then called with appropriate
    inputs hard coded in `dir_exodus.py`. `test_getters` is used to test proper
    exceptions handling rather than functionality and can print a message to
    the stdout when it encounters a method that hasn't been implemented yet,
    or is still in some form of development.
    """
    file_path = os.path.join(dir_test_file, 'test_full.g')
    mesh_getters = exodus_helper.Exodus(file_path)

    names = [a[0] for a in _dir_exodus_full]
    names_attr = _dir_exodus_full[-names[::-1].index('_'):]

    monkeypatch.setattr('builtins.input', lambda _: 'y')
    for a in names_attr:
        attr = getattr(mesh_getters, a, None)
        if attr is None:
            # print(f'Need to add: {a}')
            raise AttributeError(f'Need to add: {a}')
        prefix = attr.__name__[:3]
        if '__call__' in dir(attr) and prefix != 'put' and prefix != 'set':
            try:
                # print(f'Current Attribute: {attr.__name__}')
                result = attr(*(_attr_inputs[a]))
            except Warning:
                pass  # The function hasn't been implemented yet.
            except KeyError:
                print(f'\nAttribute not found: {attr.__name__}', 1)
                raise
            if attr.__name__ == 'copy':  # Delete the file made by copy()
                result.close()
                os.remove(_attr_inputs[a][0])


def test_putters(dir_test_file):
    """`test_putters` copies the `test_full.g` mesh and iterates
    through each of its `put` method, which are then called with appropriate
    inputs hard coded in `dir_exodus.py`. `test_putters` is used to test proper
    exceptions handling rather than functionality and can print a message to
    the stdout when it encounters a method that hasn't been implemented yet,
    or is still in some form of development.
    """
    file_path = os.path.join(dir_test_file, 'test_full.g')
    mesh_test_full = exodus_helper.Exodus(file_path)
    path_copy = os.path.join(dir_test_file, 'test_full_copy.g')
    mesh_copy = mesh_test_full.copy(path_copy)

    names = [a[0] for a in _dir_exodus_full]
    names_attr = _dir_exodus_full[-names[::-1].index('_'):]

    try:
        for a in names_attr:
            attr = getattr(mesh_copy, a, None)
            prefix = attr.__name__[:3]
            if '__call__' in dir(attr) and prefix in ('put', 'set'):
                try:
                    attr(*(_attr_inputs[a]))
                except exodus_helper.core.DatabaseError:
                    pass
                except KeyError:
                    print(f'Attribute not found: {attr.__name__}')
                    raise
        mesh_copy.close()
    finally:
        if os.path.isfile(path_copy):
            os.remove(path_copy)  # delete the copied mesh file.


def test_initialization(dir_test_file, monkeypatch):
    """This test exists to expand test coverage of the Exodus __init__ method.
    Some aspects of the constructor aren't covered by the rest of the tests
    in this class so this test creates meshes with various kwarg inputs to
    ensure every line of the constructor is executed during testing.
        Test 1: a mesh file is created when passing mode='w'.
        Test 2: a mesh file is overwritten after checking input -> 'y'.
        Test 3: a mesh file is opened read-only after checking input -> 'n'.
        Test 4: elem_type must be passed or deduced from num_nodes_per_els.
    """

    # Test 1
    file_path = os.path.join(dir_test_file, 'delete_me.g')
    monkeypatch.setattr('builtins.input', lambda _: 'y')
    mesh_init = exodus_helper.Exodus(
        file_path,
        mode='w',
        elem_type='HEX8',
        num_el_in_blks=[1],
        num_nodes_per_els=[1],
        num_glo_var=1,
        num_elem_var=1,
        time_step=1)
    mesh_init.close()
    assert os.path.isfile(file_path)

    # Test 2
    mesh_init = exodus_helper.Exodus(file_path, mode='w')
    assert mesh_init.mode == 1
    mesh_init.close()

    # Test 3
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    mesh_init = exodus_helper.Exodus(file_path, mode='w')
    assert mesh_init.mode == 2
    mesh_init.close()
    os.remove(file_path)

    # Test 4
    try:
        # init will deduce elem_type for 4,8,10 but not 12 nodes per element
        mesh_init = exodus_helper.Exodus(
            file_path,
            mode='w',
            num_el_in_blks=[1, 2, 3, 4],
            num_nodes_per_els=[4, 8, 10, 12],
            num_el_blk=4)
        assert False
    except ValueError:
        assert True
    finally:
        os.remove(file_path)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_deprecated_methods(mesh):
    """Both the `get_elem_num_map` and `get_node_num_map` methods have been
    deprecated and replaced by `get_elem_id_map` and `get_node_id_map`
    respectively.

    The deprecated method signatures have been left in the class for
    backwards compatability reasons but their functionality has been removed.
    These methods now throw a ``DeprecationWarning`` and return None.

    This test ensures that both of these methods return ``None`` as intended.
    """
    assert mesh.get_elem_num_map() is None
    assert mesh.get_node_num_map() is None


def test_eq(dir_test_file, monkeypatch):
    """This test determines if Exodus.__eq__ is working correctly."""
    try:
        file_path = os.path.join(dir_test_file, 'test_full.g')
        mesh_eq = exodus_helper.Exodus(file_path)
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        file_path_copy = os.path.join(dir_test_file, 'test_full_copy.g')
        mesh_copy = mesh_eq.copy(file_path_copy)
        assert mesh_eq == mesh_copy
    finally:
        mesh_eq.close()
        mesh_copy.close()
        os.remove(file_path_copy)


def test_properties(mesh):
    """This test exists to get full coverage on the Exodus properties unused by
    the other tests in this module."""
    assert mesh.numDim == 3
    assert mesh.numElemBlk == 1
    assert mesh.shape == (1, 1, 2)
    mesh.put_time(1, 1)
    assert mesh.times[0] == 1
    assert mesh.title() == 'rectangular_prism'


def test_add_variable(dir_test_file):
    path_from = os.path.join(dir_test_file, 'test_add_variable.e')
    mesh = exodus_helper.Exodus(path_from)
    values = [1] * mesh.numElem
    idx_time = 1
    path_add = os.path.join(dir_test_file, 'mesh_add.e')
    name_add = 'var_add'
    mesh_add = exodus_helper.add_variable(
        mesh, path_add, name_add, idx_time, values)
    dict_attrs, dict_dims, _ = mesh._get_dicts_netcdf()
    dict_attrs_add, _, _ = mesh_add._get_dicts_netcdf()

    assert dict_attrs == dict_attrs_add
    for k in dict_dims:
        if k in ['num_qa_rec', 'num_elem_var']:
            continue
        d = mesh.dataset.dimensions[k]
        d_add = mesh_add.dataset.dimensions[k]
        assert d.isunlimited() == d_add.isunlimited()
        assert d.size == d_add.size

    variables = mesh.dataset.variables
    variables_add = mesh_add.dataset.variables
    set_variables = set(variables.keys())
    set_variables_add = set(variables_add.keys())
    id_blk = mesh.get_elem_blk_ids()[0]
    name_netcdf = mesh_add.get_name_elem_variable_netcdf(name_add, id_blk)
    assert set_variables_add - set_variables == set([name_netcdf])
    variables_skip = [
        'qa_records', 'num_elem_var', 'elem_var_tab', name_netcdf,
        'name_elem_var']
    for k in variables:
        if k in variables_skip:
            continue
        v = variables[k]
        v_add = variables_add[k]
        assert v.dtype == v_add.dtype
        assert v.size == v_add.size
        assert v.dimensions == v_add.dimensions
        assert v.ncattrs() == v_add.ncattrs()
        # Check that all variable values are the same
        assert np.all(v[:].compressed() == v_add[:].compressed())
    os.remove(path_add)


# Getters ------------------------------------------------------------------- #

def test_get_all_global_variable_values(mesh):
    # Dimensions and variables to store global variable names and numbers
    mesh.dataset.createDimension('num_glo_var', size=2)
    mesh.dataset.createVariable(
        'name_glo_var', np.dtype('S1'),
        dimensions=('num_glo_var', 'len_string'))
    mesh.dataset.createVariable(
        'vals_glo_var', np.dtype('float'),
        dimensions=('time_step', 'num_glo_var'))

    # Put and get a values for each global variable
    assert mesh.put_all_global_variable_values(1, [10, 100])
    glo_vals = mesh.get_all_global_variable_values(1)
    assert isinstance(glo_vals, np.ndarray)
    assert glo_vals[0] == 10 and glo_vals[1] == 100


@pytest.mark.unwritten
def test_get_all_node_set_params():
    assert False


@pytest.mark.unwritten
def test_get_all_side_set_params():
    assert False


def test_get_coord(mesh):
    # Assert correct return type
    coord = mesh.get_coord(1)
    for c in coord:
        assert isinstance(c, np.ndarray)

    # Assert that the correct values are fetched
    coord_vals = np.arange(mesh.get_num_nodes())
    mesh.put_coords(coord_vals, coord_vals, coord_vals)
    for i, v in enumerate(coord_vals):
        assert np.all(np.stack(mesh.get_coord(i + 1)) == float(v))


def test_get_coords(mesh):
    # Assert correct return type
    coords = np.stack(mesh.get_coords()).T
    for coord in coords:
        assert isinstance(coord, np.ndarray)

    # Assert that the correct values are fetched
    coord_vals = np.arange(mesh.get_num_nodes())
    for i, v in enumerate(coord_vals):
        assert np.all(coords[i] == float(v))


def test_get_coord_names(mesh):
    # Assert correct return type and values
    names = mesh.get_coord_names()
    assert isinstance(names, list)
    assert isinstance(names[0], str)
    assert names[0] == 'x'
    assert names[1] == 'y'
    assert names[2] == 'z'


@pytest.mark.insufficient
def test_get_data_exodus(dir_test_file):
    file_path = os.path.join(
        dir_test_file, 'test_full.g')
    mesh_1 = exodus_helper.get_data_exodus(file_path)
    mesh_1.close()

    # Check exception handling
    try:
        exodus_helper.get_data_exodus(1)
        assert False
    except ValueError:
        assert True


# get_elem ------------------------------------------------------------------ #

def test_get_elem_attr(mesh):
    mesh.put_elem_attr(1, [1., 2., 3., 4.])
    attrs = mesh.get_elem_attr(1)
    for i in range(4):
        assert attrs[i] == i + 1


def test_get_elem_attr_all(mesh):
    attrs = mesh.get_elem_attr_all()
    for i in range(4):
        assert attrs[i] == i + 1


def test_get_elem_attr_names(mesh):
    test_names = ['test_name1', 'test_name2']
    mesh.put_elem_attr_names(1, test_names)
    names = mesh.get_elem_attr_names(1)
    assert names == test_names


def test_get_elem_attr_values(mesh):
    test_vals = [1., 3.]
    vals = mesh.get_elem_attr_values(1, 'test_name1')
    assert np.allclose(vals, test_vals)
    test_vals = [2., 4.]
    vals = mesh.get_elem_attr_values(1, 'test_name2')
    assert np.allclose(vals, test_vals)


def test_get_elem_attr_values_all(mesh):
    test_vals = [1., 3.]
    vals = mesh.get_elem_attr_values_all('test_name1')
    assert np.allclose(vals, test_vals)
    test_vals = [2., 4.]
    vals = mesh.get_elem_attr_values(1, 'test_name2')
    assert np.allclose(vals, test_vals)


def test_get_elem_blk_ids(mesh):
    blk_ids = mesh.get_elem_blk_ids()
    assert blk_ids[0] == 1


def test_get_elem_blk_idx(mesh):
    blk_idx = mesh.get_elem_blk_idx(1)
    assert blk_idx == 0


def test_get_elem_blk_info(mesh):
    assert mesh.get_elem_blk_info(1) == ('HEX8', 2, 8, 2)


def test_get_elem_blk_name(mesh):
    name = 'test_blk_name'
    mesh.put_elem_blk_name(1, name)
    blk_name = mesh.get_elem_blk_name(1)
    assert blk_name == name


def test_get_elem_blk_names(mesh):
    names = mesh.get_elem_blk_names()
    assert names[0] == 'test_blk_name'


def test_get_elem_connectivity(mesh):
    connectivity = mesh.get_elem_connectivity(1)
    expected = np.asarray([1, 2, 4, 3, 5, 6, 8, 7, 5, 6, 8, 7, 9, 10, 12, 11])
    assert np.allclose(connectivity[0], expected)


@pytest.mark.insufficient
def test_get_elem_connectivity_full(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh = exodus_helper.Exodus(file_path)
    connectivity_full_0 = mesh.get_element_connectivity_full()
    connectivity_full_1 = mesh.get_elem_connectivity_full()
    assert np.allclose(connectivity_full_0, connectivity_full_1)


def test_get_elem_id_map(mesh):
    id_map = mesh.get_elem_id_map()
    assert id_map[0] == 1


def test_get_elem_order_map(mesh):
    order_map = mesh.get_elem_order_map()
    assert order_map[0] == 1


@pytest.mark.unwritten
def test_get_elem_property_names(mesh):
    mesh.get_elem_property_names()
    assert False


@pytest.mark.unwritten
def test_get_elem_property_value(mesh):
    mesh.get_elem_property_value()
    assert False


def test_get_idx_ss(mesh):
    ss_idx = mesh.get_idx_ss(1)
    assert ss_idx == 0


def test_get_elem_type(mesh):
    elem_type = mesh.get_elem_type(1)
    assert elem_type == 'HEX8'


def test_get_elem_variable_names(mesh):
    # Create the dimensions and variable where var names and values are kept

    mesh.dataset.createDimension('num_elem_var', size=2)
    mesh.dataset.createVariable(
        'name_elem_var', np.dtype('S1'),
        dimensions=('num_elem_var', 'len_name'))

    # Put 2 test variable names and values in the mesh
    assert mesh.put_elem_variable_name('test_var1', 1)
    assert mesh.put_elem_variable_name('test_var2', 2)
    assert mesh.put_elem_variable_values(1, 'test_var1', 1, [1.])
    assert mesh.put_elem_variable_values(1, 'test_var2', 1, [2.])

    # Get names
    names = mesh.get_elem_variable_names()
    assert names[0] == 'test_var1' and names[1] == 'test_var2'


def test_get_elem_variable_number(mesh):
    assert mesh.get_elem_variable_number() == 2


@pytest.mark.unwritten
def test_get_elem_variable_truth_table(mesh):
    mesh.get_elem_variable_truth_table()
    assert False


def test_get_elem_variable_values(mesh):
    val1 = mesh.get_elem_variable_values(1, 'test_var1', 1)
    val2 = mesh.get_elem_variable_values(1, 'test_var2', 1)
    assert val1[0] == 1 and val2[0] == 2


def test_get_elem_variable_values_all(mesh):
    vals = mesh.get_elem_variable_values_all('test_var1')
    assert np.allclose(vals, 1)
    vals = mesh.get_elem_variable_values_all('test_var2')
    assert np.allclose(vals, 2)


def test_get_elem_variable_values_block(mesh):
    vals = mesh.get_elem_variable_values_block(1, 'test_var1')
    assert np.allclose(vals, 1)
    vals = mesh.get_elem_variable_values_block(1, 'test_var2')
    assert np.allclose(vals, 2)


# get_element --------------------------------------------------------------- #

def test_get_element_attribute(mesh):
    attrs = mesh.get_element_attribute(1)
    assert attrs[0] == 1.
    assert attrs[1] == 2.
    assert attrs[2] == 3.
    assert attrs[3] == 4.


def test_get_element_attribute_all(mesh):
    attrs = mesh.get_element_attribute_all()
    assert attrs[0] == 1.
    assert attrs[1] == 2.
    assert attrs[2] == 3.
    assert attrs[3] == 4.


def test_get_element_attribute_names(mesh):
    test_names = ['test_name1', 'test_name2']
    names = mesh.get_element_attribute_names(1)
    assert names == test_names


def test_get_element_attribute_values(mesh):
    test_vals = [1., 3.]
    vals = mesh.get_element_attribute_values(1, 'test_name1')
    assert np.allclose(vals, test_vals)
    test_vals = [2., 4.]
    vals = mesh.get_element_attribute_values(1, 'test_name2')
    assert np.allclose(vals, test_vals)


def test_get_element_attribute_values_all(mesh):
    test_vals = [1., 3.]
    vals = mesh.get_element_attribute_values_all('test_name1')
    assert np.allclose(vals, test_vals)
    test_vals = [2., 4.]
    vals = mesh.get_element_attribute_values(1, 'test_name2')
    assert np.allclose(vals, test_vals)


def test_get_element_blk_ids(mesh):
    blk_ids = mesh.get_element_blk_ids()
    assert blk_ids[0] == 1


def test_get_element_blk_idx(mesh):
    blk_idx = mesh.get_element_blk_idx(1)
    assert blk_idx == 0


def test_get_element_blk_info(mesh):
    assert mesh.get_element_blk_info(1) == ('HEX8', 2, 8, 2)


def test_get_element_blk_name(mesh):
    name = 'test_blk_name'
    blk_name = mesh.get_element_blk_name(1)
    assert blk_name == name


def test_get_element_blk_names(mesh):
    names = mesh.get_element_blk_names()
    assert names[0] == 'test_blk_name'


def test_get_element_connectivity(mesh):
    connectivity = mesh.get_element_connectivity(1)
    expected = np.asarray([1, 2, 4, 3, 5, 6, 8, 7, 5, 6, 8, 7, 9, 10, 12, 11])
    assert np.allclose(connectivity[0], expected)


def test_get_element_id_map(mesh):
    id_map = mesh.get_element_id_map()
    assert id_map[0] == 1


def test_get_element_order_map(mesh):
    order_map = mesh.get_element_order_map()
    assert order_map[0] == 1


@pytest.mark.unwritten
def test_get_element_property_names():
    assert False


@pytest.mark.unwritten
def test_get_element_property_value():
    assert False


def test_get_element_type(mesh):
    elem_type = mesh.get_element_type(1)
    assert elem_type == 'HEX8'


def test_get_element_variable_names(mesh):
    names = mesh.get_element_variable_names()
    assert names[0] == 'test_var1' and names[1] == 'test_var2'


def test_get_element_variable_number(mesh):
    assert mesh.get_element_variable_number() == 2


@pytest.mark.unwritten
def test_get_element_variable_truth_table():
    assert False


def test_get_element_variable_values(mesh):
    val1 = mesh.get_element_variable_values(1, 'test_var1', 1)
    val2 = mesh.get_element_variable_values(1, 'test_var2', 1)
    assert val1[0] == 1 and val2[0] == 2


def test_get_element_variable_values_all(mesh):
    vals = mesh.get_element_variable_values_all('test_var1')
    assert np.allclose(vals, 1)
    vals = mesh.get_element_variable_values_all('test_var2')
    assert np.allclose(vals, 2)


def test_get_element_variable_values_block(mesh):
    vals = mesh.get_element_variable_values_block(1, 'test_var1')
    assert np.allclose(vals, 1)
    vals = mesh.get_element_variable_values_block(1, 'test_var2')
    assert np.allclose(vals, 2)


# get_global ---------------------------------------------------------------- #

def test_get_global_variable_names(mesh):
    assert mesh.put_global_variable_name('test_gv_name1', 1)
    assert mesh.put_global_variable_name('test_gv_name2', 2)
    names = mesh.get_global_variable_names()
    assert names[0] == 'test_gv_name1' and names[1] == 'test_gv_name2'


def test_get_global_variable_number(mesh):
    assert mesh.get_global_variable_number() == 2


def test_get_global_variable_value(mesh):
    # Put a value in the first time step of each global variable
    mesh.put_global_variable_value('test_gv_name1', 1, 10.)
    mesh.put_global_variable_value('test_gv_name2', 1, 20.)
    value1 = mesh.get_global_variable_value('test_gv_name1', 1)
    value2 = mesh.get_global_variable_value('test_gv_name2', 1)
    assert value1 == 10 and value2 == 20


def test_get_global_variable_values(mesh):
    # Put a value in the second time step for each global varible
    mesh.put_global_variable_value('test_gv_name1', 2, 100.)
    mesh.put_global_variable_value('test_gv_name2', 2, 200.)

    # Get both global variables values to ensure correct indexing
    values1 = mesh.get_global_variable_values('test_gv_name1')
    values2 = mesh.get_global_variable_values('test_gv_name2')
    assert values1[0] == 10 and values1[1] == 100
    assert values2[0] == 20 and values2[1] == 200


# --------------------------------------------------------------------------- #

def test_get_ids_elem_in_blk(mesh):
    assert mesh.get_ids_elem_in_blk(1)[0] == 1


def test_get_idxs_elem_start_blk(mesh):
    idxs = mesh.idxs_elem_start_blk
    assert idxs[0] == 0 and idxs[1] == 2


def test_get_idxs_elem(mesh):
    ids = mesh.get_elem_id_map()
    np.random.shuffle(ids)
    mesh.put_elem_id_map(ids)
    idxs = mesh.get_idxs_elem(ids)
    assert np.all(idxs == np.arange(mesh.get_num_elems()))
    mesh.put_elem_id_map(np.sort(ids))


def test_get_idxs_node(mesh):
    ids = mesh.get_node_id_map()
    np.random.shuffle(ids)
    mesh.put_node_id_map(ids)
    idxs = mesh.get_idxs_node(ids)
    assert np.all(idxs == np.arange(mesh.get_num_nodes()))
    mesh.put_node_id_map(np.sort(ids))


def test_get_info_records(mesh):
    assert 'info_records' in mesh.dataset.variables
    assert 'num_info' in mesh.dataset.dimensions

    # Blank before assigning info
    assert mesh.get_info_records()[0] == ''

    # Put and get info
    mesh.put_info_records(['info'])
    info = mesh.get_info_records()
    assert info[0] == 'info'


# get_node ------------------------------------------------------------------ #

def test_get_node_id_map(mesh):
    id_map = mesh.get_node_id_map()
    assert np.all(id_map == np.arange(1, len(id_map) + 1))


def test_get_node_set_dist_facts(mesh):
    facts = mesh.get_node_set_dist_facts(1)
    assert np.all(facts == 1)


def test_get_node_set_ids(mesh):
    ids = mesh.get_node_set_ids()
    assert np.all(ids == np.arange(1, len(ids) + 1))


def test_get_node_set_name(dir_test_file, monkeypatch):
    # Put and get a node set name

    try:
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w', numNodeSets=2)
        mesh.put_node_set_params(99, 1)
        mesh.put_node_set_name(99, 'test_ns_name1')
        assert mesh.get_node_set_name(99) == 'test_ns_name1'
    finally:
        mesh.close()
        os.remove(file_path)


def test_get_node_set_names(mesh):
    assert mesh.put_node_set_name(1, 'test_ns_name1')
    assert mesh.put_node_set_name(2, 'test_ns_name2')
    names = mesh.get_node_set_names()
    assert names[0] == 'test_ns_name1'
    assert names[1] == 'test_ns_name2'


def test_get_node_set_nodes(mesh):
    nodes = mesh.get_node_set_nodes(1)
    # The node set should be [1, 3, 5, 7]
    for i in range(4):
        assert nodes[i] == (i * 2) + 1


def test_get_node_set_params(mesh):
    params = mesh.get_node_set_params(1)
    assert params[0] == 6 and params[1] == 6


@pytest.mark.unwritten
def test_get_node_set_property_names():
    assert False


@pytest.mark.unwritten
def test_get_node_set_property_value():
    assert False


@pytest.mark.unwritten
def test_get_node_set_variable_names():
    assert False


@pytest.mark.unwritten
def test_get_node_set_variable_number():
    assert False


@pytest.mark.unwritten
def test_get_node_set_variable_truth_table():
    assert False


@pytest.mark.unwritten
def test_get_node_set_variable_values():
    assert False


def test_get_node_variable_names(mesh):
    mesh.put_node_variable_name('test_nv_name', 1)
    mesh.put_node_variable_name('test_nv_name2', 2)
    names = mesh.get_node_variable_names()
    assert isinstance(names, list)
    assert names[0] == 'test_nv_name'
    assert names[1] == 'test_nv_name2'


def test_get_node_variable_number(mesh):
    node_var_num = mesh.get_node_variable_number()
    assert node_var_num == 2


def test_get_node_variable_values(mesh):
    values = [float(i + 1) for i in range(12)]
    mesh.put_node_variable_values('test_nv_name', 1, values)
    vals = mesh.get_node_variable_values('test_nv_name', 1)
    assert np.allclose(vals, values)

    # Assert proper exception handling
    with pytest.raises(KeyError):
        mesh.get_node_variable_values('bad name', 1)


def test_get_node_variable_values_all(mesh):
    vals_all = mesh.get_node_variable_values_all('test_nv_name')
    expected_values = [float(i + 1) for i in range(12)]
    assert np.allclose(vals_all[0], expected_values)


def test_get_nodes_on_surface(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh = exodus_helper.RectangularPrism(file_path)
    coordinates = np.column_stack(mesh.get_coords())
    for i in range(1, 7):
        nodes = mesh.get_nodes_on_surface(i) - 1
        side, dim = exodus_helper.topology.PICK_SURFACE[i]
        assert np.allclose(side(coordinates[:, dim]), coordinates[nodes, dim])


# get_num ------------------------------------------------------------------- #

def test_get_num_attr(mesh):
    assert mesh.get_num_attr(1) == 2


def test_get_num_blks(mesh):
    assert mesh.get_num_blks() == 1


def test_get_num_dimensions(mesh):
    assert mesh.get_num_dimensions()


def test_get_num_elems(mesh):
    assert mesh.get_num_elems() == 2


def test_get_num_elems_in_blk(mesh):
    assert mesh.get_num_elems_in_blk(1) == 2


def test_get_num_faces_in_side_set(mesh):
    assert mesh.get_num_faces_in_side_set(1) == 2


def test_get_num_info_records(mesh):
    assert mesh.get_num_info_records() == 1


def test_get_num_node_sets(mesh):
    assert mesh.get_num_node_sets() == 14


def test_get_num_nodes(mesh):
    assert mesh.get_num_nodes() == 12


def test_get_num_nodes_in_node_set(mesh):
    assert mesh.get_num_nodes_in_node_set(1) == 6


def test_get_num_nodes_per_elem(dir_test_file, mesh, monkeypatch):
    assert mesh.get_num_nodes_per_elem(1) == 8

    # Test exception handling
    try:
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')
        num_nodes = mesh.get_num_nodes_per_elem(1)
        assert num_nodes.size == 0
    finally:
        mesh.close()
        os.remove(file_path)


def test_get_num_side_sets(mesh):
    assert mesh.get_num_side_sets() == 6


def test_get_num_times(mesh):
    assert mesh.get_num_times() == 2


def test_get_num_qa_records(mesh):
    assert mesh.get_num_qa_records() == 1


# --------------------------------------------------------------------------- #

def test_get_qa_records(mesh):
    # Assert correct record type
    records = mesh.get_qa_records()
    assert isinstance(records, list)

    # Assert correct record values
    record = records[0]
    assert record[0] == 'exodus_helper'
    assert record[1] == exodus_helper.__version__
    assert record[2] == time.strftime('%Y-%m-%d')

    # The time stamp changes with each definition of the mesh.
    # Instead of asserting the value, only the type is checked.
    assert isinstance(record[3], str)


def test_get_resolution(dir_test_file, monkeypatch):
    try:
        shape = (3, 4, 5)
        resolution = (0.3, 0.2, 0.1)
        file_path = os.path.join(dir_test_file, 'test_get_shape.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.RectangularPrism(
            file_path, shape=shape, resolution=resolution, mode='w')
        assert np.allclose(mesh.get_resolution(), resolution)
        mesh.close()
    finally:
        os.remove(file_path)


def test_get_shape(dir_test_file, monkeypatch):
    try:
        shape = (3, 4, 5)
        file_path = os.path.join(dir_test_file, 'test_get_shape.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.RectangularPrism(file_path, shape=shape, mode='w')
        assert mesh.get_shape() == shape
        mesh.close()
    finally:
        os.remove(file_path)


def test_get_shape_surface(dir_test_file, monkeypatch):
    try:
        shape = (3, 4, 5)
        file_path = os.path.join(dir_test_file, 'test_get_shape_surface.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.RectangularPrism(file_path, shape=shape, mode='w')
        assert mesh.get_shape_surface(1) == (4, 5)
        assert mesh.get_shape_surface(2) == (4, 5)
        assert mesh.get_shape_surface(3) == (3, 5)
        assert mesh.get_shape_surface(4) == (3, 5)
        assert mesh.get_shape_surface(5) == (3, 4)
        assert mesh.get_shape_surface(6) == (3, 4)
        mesh.close()
    finally:
        os.remove(file_path)


# get_side ------------------------------------------------------------------ #

def test_get_side():
    assert exodus_helper.topology.PICK_SURFACE[1][0]([0, 1]) == 0
    assert exodus_helper.topology.PICK_SURFACE[2][0]([0, 1]) == 1
    assert exodus_helper.topology.PICK_SURFACE[3][0]([0, 1]) == 0
    assert exodus_helper.topology.PICK_SURFACE[4][0]([0, 1]) == 1
    assert exodus_helper.topology.PICK_SURFACE[5][0]([0, 1]) == 0
    assert exodus_helper.topology.PICK_SURFACE[6][0]([0, 1]) == 1


def test_get_side_set(mesh):
    id_elem, id_side = mesh.get_side_set(1)
    assert id_elem[0] == 1 and id_elem[1] == 2
    assert id_side[0] == 4 and id_side[1] == 4


@pytest.mark.insufficient
def test_get_side_set_dist_fact(mesh):
    assert np.allclose(mesh.get_side_set_dist_fact(1), 1.)


def test_get_side_set_ids(mesh):
    set_ids = mesh.get_side_set_ids()
    for i in range(6):
        assert set_ids[i] == i + 1


def test_get_side_set_name(mesh):
    # Put and get a name to each side set
    for i in range(1, 7):
        assert mesh.put_side_set_name(i, f'test_ss_name{i}')
        name = mesh.get_side_set_name(i)
        assert name == f'test_ss_name{i}'


def test_get_side_set_names(mesh):
    names = mesh.get_side_set_names()
    for i in range(1, 7):
        assert names[i - 1] == f'test_ss_name{i}'


@pytest.mark.unwritten
def test_get_side_set_node_list():
    assert False


def test_get_side_set_params(mesh, dir_test_file):
    num_faces, num_dist_facts = mesh.get_side_set_params(1)
    assert num_faces == 2 and num_dist_facts == 4

    # Test exception handling in the function
    try:
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        mesh_test = exodus_helper.Exodus(file_path, mode='w')
        num_faces, num_dist_facts = mesh_test.get_side_set_params(1)
        assert num_faces == 0 and num_dist_facts == 0
        mesh_test.close()
    finally:
        os.remove(file_path)


@pytest.mark.unwritten
def test_get_side_set_property_names():
    assert False


@pytest.mark.unwritten
def test_get_side_set_property_value():
    assert False


@pytest.mark.unwritten
def test_get_side_set_variable_names():
    assert False


@pytest.mark.unwritten
def test_get_side_set_variable_number():
    assert False


@pytest.mark.unwritten
def test_get_side_set_variable_truth_table():
    assert False


@pytest.mark.unwritten
def test_get_side_set_variable_values():
    assert False


def test_get_sides_on_surface(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh = exodus_helper.RectangularPrism(file_path)
    sides = mesh.get_sides_on_surface(1)
    assert np.allclose(sides, [5, 5, 5, 5])
    sides = mesh.get_sides_on_surface(2)
    assert np.allclose(sides, [6, 6, 6, 6])
    sides = mesh.get_sides_on_surface(3)
    assert np.allclose(sides, [1, 1, 1, 1])
    sides = mesh.get_sides_on_surface(4)
    assert np.allclose(sides, [3, 3, 3, 3])
    sides = mesh.get_sides_on_surface(5)
    assert np.allclose(sides, [2, 2, 2, 2])
    sides = mesh.get_sides_on_surface(6)
    assert np.allclose(sides, [4, 4, 4, 4])


def test_get_times(mesh):
    # Place and then fetch values at different time steps
    assert mesh.put_times([1., 2.])
    times = mesh.get_times()
    assert times[0] == 1 and times[1] == 2


def test_get_title(mesh):
    assert mesh.get_title() == 'rectangular_prism'


def test_get_version_num(mesh):
    assert mesh.get_version_num()


# External Functions: Setters ----------------------------------------------- #

@pytest.mark.unwritten
def test_set_element_variable_number():
    assert False


@pytest.mark.unwritten
def test_set_element_variable_truth_table():
    assert False


@pytest.mark.unwritten
def test_set_global_variable_number():
    assert False


@pytest.mark.unwritten
def test_set_node_set_variable_number():
    assert False


@pytest.mark.unwritten
def test_set_node_set_variable_truth_table():
    assert False


@pytest.mark.unwritten
def test_set_node_variable_number():
    assert False


@pytest.mark.unwritten
def test_set_side_set_variable_number():
    assert False


@pytest.mark.unwritten
def test_set_side_set_variable_truth_table():
    assert False


# External Functions: Putters ----------------------------------------------- #

def test_put_all_global_variable_values(mesh):
    # Put and get values for time steps 1 and 2
    assert mesh.put_all_global_variable_values(1, [10., 20.])
    assert mesh.put_all_global_variable_values(2, [100., 200.])
    vals1 = mesh.get_all_global_variable_values(1)
    vals2 = mesh.get_all_global_variable_values(2)
    assert vals1[0] == 10 and vals1[1] == 20
    assert vals2[0] == 100 and vals2[1] == 200

    # Test exception handling within function
    try:
        mesh.put_all_global_variable_values(1, [9])
        assert False
    except AssertionError:
        assert True


def test_put_concat_elem_blk(dir_test_file, monkeypatch):
    # Function can only be used once on a new mesh
    file_path = os.path.join(dir_test_file, 'delete_me.g')
    monkeypatch.setattr('builtins.input', lambda _: 'y')
    mesh = exodus_helper.Exodus(file_path, mode='w')

    # Test exception handling
    try:
        mesh.put_concat_elem_blk([1], ['HEX8'], [2], [8], [0, 1], True)
        assert False
    except AssertionError:
        assert True
    try:
        assert mesh.put_concat_elem_blk([1], ['HEX8'], [2], [8], [0], True)
        assert mesh.dataset.variables['eb_status'][0] == 1
        assert mesh.dataset.variables['eb_prop1'][0] == 1
    finally:
        mesh.close()
        os.remove(file_path)


def test_put_coord_names(mesh):
    mesh.put_coord_names(['x', 'y', 'z'])
    names = mesh.get_coord_names()
    assert names[0] == 'x'
    assert names[1] == 'y'
    assert names[2] == 'z'


def test_put_coords(mesh):
    coordx = np.arange(12)
    coordy = 10 * coordx
    coordz = 100 * coordx
    mesh.put_coords(coordx, coordy, coordz)
    coords = mesh.get_coords()
    assert np.allclose(coordx, coords[0])
    assert np.allclose(coordy, coords[1])
    assert np.allclose(coordz, coords[2])


# put_elem ------------------------------------------------------------------ #

def test_put_elem_attr(mesh, dir_test_file, monkeypatch):
    try:
        # Putter is tested in the getter test
        test_get_elem_attr(mesh)

        # Ensure variable create happens properly
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')
        assert not mesh.put_elem_attr(1, [1])
    finally:
        mesh.close()
        os.remove(file_path)


def test_put_elem_attr_names(mesh, dir_test_file, monkeypatch):
    try:
        # Putter is tested in the getter test
        test_get_elem_attr_names(mesh)

        # Test exception handling
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')
        assert not mesh.put_elem_attr_names(1, ['name'])
    finally:
        mesh.close()
        os.remove(file_path)


def test_put_elem_attr_values(mesh):
    mesh.put_elem_attr_values(1, 'test_name1', [10, 20])
    mesh.put_elem_attr_values(1, 'test_name2', [30, 40])
    vals1 = mesh.get_elem_attr_values(1, 'test_name1')
    vals2 = mesh.get_elem_attr_values(1, 'test_name2')
    assert vals1[0] == 10 and vals1[1] == 20
    assert vals2[0] == 30 and vals2[1] == 40

    # Test exception handling
    try:  # Bad values length
        mesh.put_elem_attr_values(1, 'test_name1', [10, 20, 30])
        assert False
    except TypeError:
        assert True


def test_put_elem_blk_info(dir_test_file, monkeypatch):
    try:
        # Create a new mesh so that new element block info can be written
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_new = exodus_helper.Exodus(file_path, mode='w')

        # Add new element block info to the new mesh dataset
        dimensions = mesh_new.dataset.dimensions
        variables = mesh_new.dataset.variables
        mesh_new.put_elem_blk_info(1, 'HEX8', 2, 8, 2)

        # Assert that the dimensions and variables were created properly
        assert dimensions['num_el_in_blk1'].size == 2
        assert dimensions['num_nod_per_el1'].size == 8
        assert dimensions['num_att_in_blk1'].size == 2

        assert variables['connect1'].shape == (2, 8)
        assert variables['connect1'].elem_type == 'HEX8'
        assert variables['attrib1'].shape == (2, 2)
    finally:
        mesh_new.close()
        os.remove(file_path)


def test_put_elem_blk_name(mesh):
    mesh.put_elem_blk_name(1, 'new_test_name')
    name = mesh.get_elem_blk_name(1)
    assert name == 'new_test_name'


def test_put_elem_blk_names(mesh, dir_test_file, monkeypatch):
    mesh.put_elem_blk_names(['new_test_name'])
    name = mesh.get_elem_blk_names()
    assert name[0] == 'new_test_name'

    try:
        # Test exception handling
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_new = exodus_helper.Exodus(file_path, mode='w')
        assert not mesh_new.put_elem_blk_names(['name1', 'name2'])
    finally:
        mesh_new.close()
        os.remove(file_path)


def test_put_elem_connectivity(mesh):
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 10, 11, 12]])
    mesh.put_elem_connectivity(1, conn)
    new_conn = mesh.get_elem_connectivity(1)
    assert np.allclose(new_conn[0], conn.reshape(1, 16)[0])
    assert new_conn[1] == 2
    assert new_conn[2] == 8


def test_put_elem_id_map(mesh):
    mesh.put_elem_id_map([2, 1])
    id_map = mesh.get_elem_id_map()
    assert id_map[0] == 2 and id_map[1] == 1


@pytest.mark.unwritten
def test_put_elem_property_value(mesh):
    mesh.put_elem_property_value()
    assert False


def test_put_elem_variable_name(mesh):

    # Put 2 test variable names
    assert mesh.put_element_variable_name('new_test_var1', 1)
    assert mesh.put_element_variable_name('new_test_var2', 2)
    # Check for new names
    names = mesh.get_elem_variable_names()
    assert names[0] == 'new_test_var1' and names[1] == 'new_test_var2'
    # Check that variables can be accessed by the new names
    assert mesh.put_element_variable_values(1, 'new_test_var1', 1, [1.])
    assert mesh.put_element_variable_values(1, 'new_test_var2', 1, [2.])

    # Ensure KeyError is raised
    with pytest.raises(IndexError):
        mesh.put_elem_variable_name('test_var1', 5)


def test_put_elem_variable_values(mesh):
    mesh.put_elem_variable_values(1, 'new_test_var1', 1, [3, 4])
    vals = mesh.get_elem_variable_values(1, 'new_test_var1', 1)
    assert vals[0] == 3 and vals[1] == 4


# put_element --------------------------------------------------------------- #

def test_put_element_attribute(mesh):
    mesh.put_element_attribute(1, [1., 2., 3., 4.])
    attrs = mesh.get_element_attribute(1)
    for i in range(4):
        assert attrs[i] == i + 1


def test_put_element_attribute_names(mesh):
    test_names = ['attribute_name1', 'attribute_name2']
    mesh.put_element_attribute_names(1, test_names)
    names = mesh.get_element_attribute_names(1)
    assert names == test_names


def test_put_element_attribute_values(mesh):
    mesh.put_element_attribute_values(1, 'attribute_name1', [10, 20])
    mesh.put_element_attribute_values(1, 'attribute_name2', [30, 40])
    vals1 = mesh.get_element_attribute_values(1, 'attribute_name1')
    vals2 = mesh.get_element_attribute_values(1, 'attribute_name2')
    assert vals1[0] == 10 and vals1[1] == 20
    assert vals2[0] == 30 and vals2[1] == 40

    # Test exception handling
    with pytest.raises(KeyError):
        mesh.put_element_attribute_values(1, 'bad_name', [1, 2])


def test_put_element_blk_info(dir_test_file, monkeypatch):
    try:
        # Create a new mesh so that new element block info can be written
        file_path = os.path.join(dir_test_file, 'delete_me_too.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_new = exodus_helper.Exodus(file_path, mode='w')

        # Add new element block info to the new mesh dataset
        dimensions = mesh_new.dataset.dimensions
        variables = mesh_new.dataset.variables
        mesh_new.put_element_blk_info(1, 'HEX8', 2, 8, 2)

        # Assert that the dimensions and variables were created properly
        assert dimensions['num_el_in_blk1'].size == 2
        assert dimensions['num_nod_per_el1'].size == 8
        assert dimensions['num_att_in_blk1'].size == 2

        assert variables['connect1'].shape == (2, 8)
        assert variables['connect1'].elem_type == 'HEX8'
        assert variables['attrib1'].shape == (2, 2)
    finally:
        mesh_new.close()
        os.remove(file_path)


def test_put_element_blk_name(mesh):
    mesh.put_element_blk_name(1, 'new_elem_blk_name')
    name = mesh.get_element_blk_name(1)
    assert name == 'new_elem_blk_name'


def test_put_element_blk_names(mesh):
    mesh.put_element_blk_names(['new_elem_blk_name'])
    name = mesh.get_element_blk_names()
    assert name[0] == 'new_elem_blk_name'


def test_put_element_connectivity(mesh):
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 10, 11, 12]])
    mesh.put_element_connectivity(1, conn)
    new_conn = mesh.get_element_connectivity(1)
    assert np.allclose(new_conn[0], conn.reshape(1, 16)[0])
    assert new_conn[1] == 2
    assert new_conn[2] == 8


def test_put_element_id_map(mesh):
    mesh.put_element_id_map([2, 1])
    id_map = mesh.get_element_id_map()
    assert id_map[0] == 2 and id_map[1] == 1


@pytest.mark.unwritten
def test_put_element_property_value():
    assert False


def test_put_element_variable_values(mesh):
    mesh.put_element_variable_values(1, 'new_test_var1', 1, [3, 4])
    vals = mesh.get_element_variable_values(1, 'new_test_var1', 1)
    assert vals[0] == 3 and vals[1] == 4


# put_global ---------------------------------------------------------------- #

def test_put_global_variable_name(mesh):
    assert mesh.put_global_variable_name('new_test_gv_name1', 1)
    name = mesh.get_global_variable_names()
    assert name[0] == 'new_test_gv_name1'


def test_put_global_variable_value(mesh):
    mesh.put_global_variable_value('new_test_gv_name1', 1, 100.)
    value = mesh.get_global_variable_value('new_test_gv_name1', 1)
    assert value == 100


# put_info ------------------------------------------------------------------ #

@pytest.mark.unwritten
def test_put_info():
    assert False


def test_put_info_records(dir_test_file, monkeypatch):
    try:
        # Create a mesh with num_info=2
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w', num_info=2)

        # Put and get info records
        test_info = ['info', 'records']
        mesh.put_info_records(test_info)
        info = mesh.get_info_records()
        assert info[0] == test_info[0]
        assert info[1] == test_info[1]

    finally:
        mesh.close()
        os.remove(file_path)


# put_node ------------------------------------------------------------------ #

def test_put_node_id_map(mesh):
    id_map = mesh.get_node_id_map()
    test_map = list(reversed(id_map))
    mesh.put_node_id_map(test_map)
    new_map = mesh.get_node_id_map()
    assert np.allclose(new_map, test_map)


def test_put_node_set(mesh, dir_test_file, monkeypatch):
    ns = [1, 3, 5, 7, 11, 9]
    mesh.put_node_set(1, ns)
    assert np.allclose(mesh.dataset.variables['node_ns1'][:].data, ns)
    try:
        # Test exception handling
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_new = exodus_helper.Exodus(file_path, mode='w')
        assert not mesh_new.put_node_set(10, ns)
    finally:
        mesh_new.close()
        os.remove(file_path)


def test_put_node_set_dist_fact(mesh):
    test_dist_facts = [2, 2, 2, 2, 2, 2]
    mesh.put_node_set_dist_fact(1, test_dist_facts)
    dist_facts = mesh.get_node_set_dist_facts(1)
    assert np.allclose(test_dist_facts, dist_facts)

    # Test exception handling
    with pytest.raises(KeyError):
        mesh.put_node_set_dist_fact(100, test_dist_facts)


def test_put_node_set_name(mesh):
    mesh.put_node_set_name(1, 'new_test_ns_name1')
    name = mesh.get_node_set_name(1)
    assert name == 'new_test_ns_name1'


def test_put_node_set_names(mesh):
    # Make a list of test names
    test_names = ['new_test_ns_name1', 'test_ns_name2']
    for i in range(3, 15):
        test_names.append(str(i))

    # Put and get names
    mesh.put_node_set_names(test_names)
    assert np.all(test_names == mesh.get_node_set_names())

    # Ensure AssertionError is thrown for bad name list length
    try:
        bad_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        mesh.put_node_set_names(bad_names)
        assert False
    except AssertionError:
        assert True


def test_put_node_set_params(dir_test_file, monkeypatch):
    try:
        # Params can only be used once on a fresh mesh
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w', numNodeSets=1)
        mesh.put_node_set_params(1, 1, numSetDistFacts=1)

        # Assert that dimensions and variables were created properly
        assert mesh.dataset.dimensions['num_nod_ns1'].size == 1

        v1 = mesh.dataset.variables['node_ns1']
        assert v1.dtype == mesh.int_type
        assert v1.size == 1

        v2 = mesh.dataset.variables['dist_fact_ns1']
        assert v2.dtype == np.dtype('float')
        assert v2[0].data == 1

    finally:
        mesh.close()
        os.remove(file_path)


@pytest.mark.unwritten
def test_put_node_set_property_value():
    assert False


@pytest.mark.unwritten
def test_put_node_set_variable_name():
    assert False


@pytest.mark.unwritten
def test_put_node_set_variable_values():
    assert False


@pytest.mark.insufficient
def test_put_node_variable_name(mesh):
    # Put method tested in the getter test
    test_get_node_variable_names(mesh)


@pytest.mark.insufficient
def test_put_node_variable_values(mesh):
    # Put method tested in the getter test
    test_get_node_variable_values(mesh)


def test_put_qa_records(mesh, dir_test_file, monkeypatch):
    try:
        # Create a new mesh with no qa records
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_copy = mesh.copy(file_path)

        # Assert that the former qa records are preserved
        qa_records = mesh_copy.get_qa_records()
        assert qa_records[0] == mesh.get_qa_records()[0]

        new_record = qa_records[1]
        assert new_record[0] == 'exodus_helper'
        assert new_record[1] == exodus_helper.__version__
        assert new_record[2] == time.strftime('%Y-%m-%d')

        # The time stamp changes with each definition of the mesh
        # Instead of asserting the value, only the type is checked
        assert isinstance(new_record[3], str)
    finally:
        mesh_copy.close()
        os.remove(file_path)

    # Ensure errors are thrown
    try:  # Wrong length
        mesh.put_qa_records(['bad list'])
        assert False
    except AssertionError:
        assert True
    try:  # Elements are not strings
        mesh.put_qa_records([('1', '2', '3', 4)])
        assert False
    except TypeError:
        assert True


# put_side ------------------------------------------------------------------ #

def test_put_side_set(mesh):
    test_ss_sides = [2, 2]
    test_ss_elems = [2, 1]
    mesh.put_side_set(1, test_ss_elems, test_ss_sides)
    ss_elems = mesh.dataset.variables['elem_ss1'][:]
    ss_sides = mesh.dataset.variables['side_ss1'][:]
    assert ss_elems[0] == 2 and ss_elems[1] == 1
    assert ss_sides[0] == 2 and ss_sides[1] == 2


def test_put_side_set_dist_fact(mesh):
    mesh.put_side_set_dist_fact(1, [0.1, 0.1, 0.1, 0.1])
    ss_df = mesh.get_side_set_dist_fact(1)
    assert ss_df[0] == 0.1 and ss_df[1] == 0.1


def test_put_side_set_name(mesh):
    # Putter tested along side the getter
    test_get_side_set_name(mesh)


def test_put_side_set_names(mesh):
    # Put 6 test names
    ss_names = [f'test_ss_name{i}' for i in range(1, 7)]
    mesh.put_side_set_names(ss_names)
    names = mesh.get_side_set_names()

    # Assert all 6 test names are correct
    assert np.all(names == ss_names)

    # Ensure AssertionError is thrown for bad name list length
    try:
        bad_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        mesh.put_side_set_names(bad_names)
        assert False
    except AssertionError:
        assert True


def test_put_side_set_params(dir_test_file, monkeypatch):
    try:
        # Params can only be used once on a fresh mesh
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w', num_side_sets=1)
        mesh.put_side_set_params(1, 1, numSetDistFacts=2)

        # Assert that dimensions and variables were created properly
        assert mesh.dataset.dimensions['num_side_ss1'].size == 1
        assert mesh.dataset.dimensions['num_df_ss1'].size == 2

        v1 = mesh.dataset.variables['side_ss1']
        assert v1.dtype == mesh.int_type
        assert v1.size == 1
        v2 = mesh.dataset.variables['elem_ss1']
        assert v2.dtype == mesh.int_type
        assert v2.size == 1
        v3 = mesh.dataset.variables['dist_fact_ss1']
        assert v3.dtype == np.dtype('float')
        assert v3.size == 2
    finally:
        mesh.close()
        os.remove(file_path)


@pytest.mark.unwritten
def test_put_side_set_property_value():
    assert False


@pytest.mark.unwritten
def test_put_side_set_variable_name():
    assert False


@pytest.mark.unwritten
def test_put_side_set_variable_values():
    assert False


def test_put_time(dir_test_file, monkeypatch):
    try:
        # Test on a new mesh to ensure dimensions and variables are created
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')

        # The time_step dimensions should dynamically resize as times are added
        mesh.put_time(1, 1)
        mesh.put_time(2, 2)
        assert mesh.numTimes == 2
        times = mesh.get_times()
        assert times[0] == 1 and times[1] == 2

        # Times should also be able to skip integers and fill the intermediate
        # time values with 0's
        mesh.put_time(5, 5)
        times = mesh.get_times()
        assert mesh.numTimes == 5
        assert times[2] == 0 and times[3] == 0
        assert times[4] == 5
    finally:
        mesh.close()
        os.remove(file_path)


def test_put_times(dir_test_file, monkeypatch):
    try:
        # Test on a new mesh to ensure dimensions and variables are created
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')

        # Put and get times
        test_times = [1, 2, 3, 4, 5]
        mesh.put_times(test_times)
        times = mesh.get_times()

        assert mesh.numTimes == 5
        assert np.all(times == test_times)

    finally:
        mesh.close()
        os.remove(file_path)


# External Functions: Special Methods --------------------------------------- #

def test_close(mesh):
    mesh.close()
    try:
        print(mesh.dataset.variables['time_whole'])
        success = False
    except RuntimeError:
        success = True
    assert success


def test_copy(dir_test_file, monkeypatch):
    """ Assert that the copied mesh dataset is identical to the original. """

    try:
        # Copy mesh
        file_path = os.path.join(dir_test_file, 'test_full.g')
        file_path_copy = os.path.join(dir_test_file, 'delete_me.g')
        mesh = exodus_helper.Exodus(file_path)
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh_copy = mesh.copy(file_path_copy)
        assert mesh == mesh_copy

        # Test 2
        title_old = mesh.dataset.getncattr('title')
        mesh_copy.dataset.setncattr('title', title_old + 'x')
        assert mesh != mesh_copy
        mesh_copy.dataset.setncattr('title', title_old)
        assert mesh == mesh_copy

        # Test 3
        name_old = mesh.dataset.dimensions['num_dim'].name
        mesh_copy.dataset.renameDimension(name_old, name_old + 'x')
        assert mesh != mesh_copy
        mesh_copy.dataset.renameDimension(name_old + 'x', name_old)
        assert mesh == mesh_copy

        # Test 4
        # num_dim_old = mesh.dataset.dimensions['num_dim'].size
        # mesh_copy.put_time(2,2)
        # assert mesh != mesh_copy

        # Test 5
        mesh_copy.put_time(2, 2)
        assert mesh != mesh_copy
    finally:
        mesh_copy.close()
        os.remove(file_path_copy)


def test_decode(dir_test_file, monkeypatch):
    try:
        file_path = os.path.join(dir_test_file, 'delete_me.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.Exodus(file_path, mode='w')

        # Decode is used to get coord names
        coord_names = mesh.get_coord_names()
        assert coord_names[0] == 'x'
        assert coord_names[1] == 'y'
        assert coord_names[2] == 'z'
    finally:
        mesh.close()
        os.remove(file_path)


# Testing the element_calculations module ----------------------------------- #

def test_calculate_volume_element():

    s = 2.
    p = 0.5 * s
    a = 1.

    coordinates_1 = [
        [-p, -p, -p],
        [p, -p, -p],
        [p, p, -p],
        [-p, p, -p],
        [-p, -p, a - p],
        [p, -p, p],
        [p, p, p],
        [-p, p, p]]

    coordinates_2 = [
        [-p, -p, -p],
        [p, -p, -p],
        [p, p, -p],
        [-p, p, -p],
        [-p, -p, a + p],
        [p, -p, p],
        [p, p, p],
        [-p, p, p]]

    v = 2 * s**3
    v_1 = exodus_helper.calculate_volume_element(coordinates_1)
    v_2 = exodus_helper.calculate_volume_element(coordinates_2)
    assert np.isclose(v, v_1 + v_2)


def test_calculate_volume_hexahedron():
    s = 2.
    a = 1.
    v = s**2 * (s - a) / 3 + a * s**2
    p = 0.5 * s

    coordinates = [
        [-p, -p, -p],
        [p, -p, -p],
        [p, p, -p],
        [-p, p, -p],
        [-p, -p, a - p],
        [p, -p, p],
        [p, p, a - p],
        [-p, p, p]]

    v_test = exodus_helper.calculate_volume_hexahedron(coordinates)
    assert np.isclose(v, v_test)


def test_calculate_volumes_element(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_calculate_volumes_element.g')
    mesh = exodus_helper.Exodus(file_path)
    volumes = exodus_helper.calculate_volumes_element(mesh, 1)
    assert np.allclose(6., volumes)
    assert np.isclose(48., np.sum(volumes))


def test_calculate_volumes_block(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_calculate_volumes_block.g')
    mesh = exodus_helper.Exodus(file_path)
    volumes = exodus_helper.calculate_volumes_block(mesh)
    assert np.allclose(.125, volumes)
    assert np.isclose(1., np.sum(volumes))


# Testing the reconfigure_mesh module --------------------------------------- #

def test_convert_tet4_tet10(dir_test_file, monkeypatch):
    # monkeypatch.setattr('builtins.input', lambda _: 'y')
    file_path = os.path.join(dir_test_file, 'test_convert_tet.g')
    mesh_converted = exodus_helper.convert_tet4_tet10(file_path)
    connectivity = mesh_converted.get_elem_connectivity_full()[:]
    coords = np.stack(mesh_converted.get_coords()).T
    for ids_node in connectivity:
        for idx, edge in enumerate(IDXS_EDGES_4):
            coords_0 = coords[ids_node[edge[0]] - 1]
            coords_1 = coords[ids_node[edge[1]] - 1]
            coords_2 = coords[ids_node[idx + 4] - 1]
            assert np.allclose(0.5 * (coords_0 + coords_1), coords_2)
    for idx_coords, coords_now in enumerate(coords[:-1]):
        norms = np.linalg.norm(coords[idx_coords + 1:] - coords_now, axis=1)
        assert not np.any(np.isclose(norms, 0))
    os.remove(os.path.join(dir_test_file, 'test_convert_tet_tet10.g'))


def test_create_sets_canonical(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_create_sets_canonical.g')
    exodus_helper.create_sets_canonical(file_path)
    try:
        file_path = os.path.join(
            dir_test_file, 'test_create_sets_canonical_canonical.g')
        mesh = exodus_helper.Exodus(file_path)
        coords_x, coords_y, coords_z = mesh.get_coords()
        min_x = np.where(np.isclose(-0.5, coords_x))[0] + 1
        max_x = np.where(np.isclose(0.5, coords_x))[0] + 1
        min_y = np.where(np.isclose(-0.5, coords_y))[0] + 1
        max_y = np.where(np.isclose(0.5, coords_y))[0] + 1
        min_z = np.where(np.isclose(-0.5, coords_z))[0] + 1
        max_z = np.where(np.isclose(0.5, coords_z))[0] + 1
        assert set(mesh.get_node_set_nodes(1)) == set(min_x)
        assert set(mesh.get_node_set_nodes(2)) == set(max_x)
        assert set(mesh.get_node_set_nodes(3)) == set(min_y)
        assert set(mesh.get_node_set_nodes(4)) == set(max_y)
        assert set(mesh.get_node_set_nodes(5)) == set(min_z)
        assert set(mesh.get_node_set_nodes(6)) == set(max_z)
        test_set = set(np.intersect1d(np.intersect1d(min_x, min_y), min_z))
        assert set(mesh.get_node_set_nodes(7)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(min_x, min_y), max_z))
        assert set(mesh.get_node_set_nodes(8)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(min_x, max_y), min_z))
        assert set(mesh.get_node_set_nodes(9)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(min_x, max_y), max_z))
        assert set(mesh.get_node_set_nodes(10)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(max_x, min_y), min_z))
        assert set(mesh.get_node_set_nodes(11)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(max_x, min_y), max_z))
        assert set(mesh.get_node_set_nodes(12)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(max_x, max_y), min_z))
        assert set(mesh.get_node_set_nodes(13)) == test_set
        test_set = set(np.intersect1d(np.intersect1d(max_x, max_y), max_z))
        assert set(mesh.get_node_set_nodes(14)) == test_set
    finally:
        mesh.close()
        os.remove(file_path)


# Testing the render_mesh module -------------------------------------------- #

@pytest.mark.insufficient
def test_render_mesh(dir_test_file, monkeypatch):
    try:
        file_path = os.path.join(dir_test_file, 'test_render_mesh.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        sample_ratio = 0.5
        shape = (3, 4, 5)
        resolution = (3, 2, 1)
        mesh = exodus_helper.RectangularPrism(
            file_path, shape=shape, resolution=resolution, mode='w')
        lengths = np.array([(s - 1) * r for s, r in zip(shape, resolution)])
        for surface in range(1, 7):
            colors = exodus_helper.render_mesh(
                mesh, surface, sample_ratio=sample_ratio, return_coords=False)
            dims = exodus_helper.topology.get_dims_surface(surface)
            shape_target = np.array(mesh.shape)[dims]
            num_elems = np.sqrt(np.prod(shape_target))
            lengths_target = lengths[dims]
            ratio_length = np.sqrt(lengths_target[1] / lengths_target[0])
            shape_target = (
                int(num_elems * ratio_length / sample_ratio) + 1,
                int(num_elems / ratio_length / sample_ratio) + 1)
            shape_test = colors.shape
            tests = shape_test == shape_target
            message = f'{shape_test} vs. {shape_target}'
            assert np.all(tests), f'Failed on surface {surface}: {message}'
    finally:
        mesh.close()
        os.remove(file_path)


def test_map_points_to_elements(dir_test_file, monkeypatch):
    try:
        file_path = os.path.join(dir_test_file, 'test_map_points.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        shape = (3, 4, 5)
        resolution = (3, 2, 1)
        mesh = exodus_helper.RectangularPrism(
            file_path, shape=shape, resolution=resolution, mode='w')
        points = np.random.rand(4, 3) * shape * resolution
        parents = exodus_helper.map_points_to_elements(mesh, points)
        nodes = mesh.get_elem_connectivity_full()
        coords = np.column_stack(mesh.get_coords())
        for point, parent in zip(points, parents):
            mins = np.min(coords[nodes[parent] - 1], axis=0)
            maxs = np.max(coords[nodes[parent] - 1], axis=0)
            assert np.all(point >= mins)
            assert np.all(point <= maxs)
    finally:
        mesh.close()
        os.remove(file_path)


# Testing the topology module ----------------------------------------------- #

def test_get_element_centroids(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh_test = exodus_helper.Exodus(file_path)
    centroids = mesh_test.get_element_centroids()
    assert np.allclose(0.0, np.mean(centroids, axis=0))
    centroids = mesh_test.get_elem_centroids()
    assert np.allclose(0.0, np.mean(centroids, axis=0))


def test_get_elements_on_surface(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh_test = exodus_helper.RectangularPrism(file_path)
    elems = mesh_test.get_elements_on_surface(1)
    assert set(elems) == set([1, 2, 3, 4])
    elems = mesh_test.get_elements_on_surface(2)
    assert set(elems) == set([5, 6, 7, 8])
    elems = mesh_test.get_elements_on_surface(3)
    assert set(elems) == set([1, 2, 5, 6])
    elems = mesh_test.get_elements_on_surface(4)
    assert set(elems) == set([3, 4, 7, 8])
    elems = mesh_test.get_elements_on_surface(5)
    assert set(elems) == set([2, 4, 6, 8])
    elems = mesh_test.get_elements_on_surface(6)
    assert set(elems) == set([1, 3, 5, 7])


def test_get_elements_on_surface_sorted(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh_test = exodus_helper.RectangularPrism(file_path)
    elements = mesh_test.get_elements_on_surface_sorted(1)
    assert np.allclose(elements, [2, 4, 1, 3])
    elements = mesh_test.get_elements_on_surface_sorted(2)
    assert np.allclose(elements, [6, 8, 5, 7])
    elements = mesh_test.get_elements_on_surface_sorted(3)
    assert np.allclose(elements, [2, 6, 1, 5])
    elements = mesh_test.get_elements_on_surface_sorted(4)
    assert np.allclose(elements, [4, 8, 3, 7])
    elements = mesh_test.get_elements_on_surface_sorted(5)
    assert np.allclose(elements, [2, 6, 4, 8])
    elements = mesh_test.get_elements_on_surface_sorted(6)
    assert np.allclose(elements, [1, 5, 3, 7])


def test_get_elements_sides_on_surface(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh_test = exodus_helper.RectangularPrism(file_path)
    for i in range(1, 7):
        elems, sides = mesh_test.get_elements_sides_on_surface(i)
        assert np.all(
            elems == mesh_test.get_elements_on_surface(i))
        assert np.all(
            sides == mesh_test.get_sides_on_surface(i))


def test_get_elements_sorted(dir_test_file):
    file_path = os.path.join(dir_test_file, 'test_get_x_on_surface.g')
    mesh_test = exodus_helper.RectangularPrism(file_path)
    elems_sorted = mesh_test.get_elements_sorted()
    assert np.all(elems_sorted == [2, 6, 4, 8, 1, 5, 3, 7])


def test_rectangular_prism(dir_test_file, monkeypatch):
    try:
        file_path = os.path.join(
            dir_test_file, 'test_rectangular_prism.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mesh = exodus_helper.RectangularPrism(
            file_path, shape=(1, 2, 3), resolution=(.1, .2, .3), mode='w')
        volumes = exodus_helper.calculate_volumes_element(mesh, 1)
        assert np.allclose(.006, volumes)
        assert np.allclose(.036, np.sum(volumes))
        mesh.close()
    finally:
        os.remove(file_path)

    try:
        file_path = os.path.join(
            dir_test_file, 'test_rectangular_prism.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        elem_blk_info = (
            [1, 2], ['HEX8', 'HEX8'], [32, 32], [8, 8], [0, 0], True)
        mesh = exodus_helper.RectangularPrism(
            file_path, shape=(4, 4, 4), mode='w', elem_blk_info=elem_blk_info)
        for i in mesh.get_side_set_ids():
            assert mesh.get_side_set(i)[0].size == 16
            assert mesh.get_node_set_nodes(i).size == 25
        mesh.close()
    finally:
        os.remove(file_path)


def test_scale_mesh(dir_test_file, monkeypatch):
    try:
        file_path = os.path.join(
            dir_test_file, 'test_calculate_volumes_element.g')
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        exodus_helper.scale_mesh(file_path, scale=(.1, 2., .5))
        mesh = exodus_helper.Exodus(file_path.replace('.g', '_scaled.g'))
        volumes = exodus_helper.calculate_volumes_element(mesh, 1)
        assert np.allclose(.6, volumes)
        assert np.isclose(4.8, np.sum(volumes))
        mesh.close()
    finally:
        os.remove(file_path.replace('.g', '_scaled.g'))

# --------------------------------------------------------------------------- #
