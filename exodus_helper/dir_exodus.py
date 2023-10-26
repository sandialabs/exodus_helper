
# --------------------------------------------------------------------------- #

import numpy as np

# --------------------------------------------------------------------------- #

# This file contains a list of the python bindings to functions in the exodus
#   library, version 2.0 from the seacas package
#   https://github.com/gsjaardema/seacas
#
# To update the list, follow the steps below:
#
# With access to seacas (and exodus.py therein), in the Python interpreter, do
# >>> from exodus import exodus
# List the attributes of the exodus class
# >>> dir(exodus)
# List the attributes with a call method
# >>> [s for s in dir(exodus) if hasattr(getattr(exodus,s),'__call__')]

_dir_exodus_full = [
    '__class__',
    '__delattr__',
    '__dict__',
    '__dir__',
    '__doc__',
    '__eq__',
    '__format__',
    '__ge__',
    '__getattribute__',
    '__gt__',
    '__hash__',
    '__init__',
    '__init_subclass__',
    '__le__',
    '__lt__',
    '__module__',
    '__ne__',
    '__new__',
    '__reduce__',
    '__reduce_ex__',
    '__repr__',
    '__setattr__',
    '__sizeof__',
    '__str__',
    '__subclasshook__',
    '__weakref__',
    '_ex_get_info_recs_quietly',
    '_exodus__copy_file',
    '_exodus__create',
    '_exodus__ex_get_all_times',
    '_exodus__ex_get_assemblies',
    '_exodus__ex_get_assembly',
    '_exodus__ex_get_attributes',
    '_exodus__ex_get_blob',
    '_exodus__ex_get_block',
    '_exodus__ex_get_coord',
    '_exodus__ex_get_coord_names',
    '_exodus__ex_get_elem_attr',
    '_exodus__ex_get_elem_attr_names',
    '_exodus__ex_get_elem_conn',
    '_exodus__ex_get_elem_num_map',
    '_exodus__ex_get_elem_order_map',
    '_exodus__ex_get_id_map',
    '_exodus__ex_get_ids',
    '_exodus__ex_get_info',
    '_exodus__ex_get_info_recs',
    '_exodus__ex_get_name',
    '_exodus__ex_get_names',
    '_exodus__ex_get_node_num_map',
    '_exodus__ex_get_node_set',
    '_exodus__ex_get_node_set_dist_fact',
    '_exodus__ex_get_object_truth_vector',
    '_exodus__ex_get_one_attr',
    '_exodus__ex_get_partial_coord',
    '_exodus__ex_get_partial_var',
    '_exodus__ex_get_prop',
    '_exodus__ex_get_prop_names',
    '_exodus__ex_get_qa',
    '_exodus__ex_get_reduction_variable_name',
    '_exodus__ex_get_reduction_variable_names',
    '_exodus__ex_get_reduction_variable_param',
    '_exodus__ex_get_reduction_vars',
    '_exodus__ex_get_set_param',
    '_exodus__ex_get_side_set',
    '_exodus__ex_get_side_set_dist_fact',
    '_exodus__ex_get_side_set_node_list',
    '_exodus__ex_get_side_set_node_list_len',
    '_exodus__ex_get_time',
    '_exodus__ex_get_truth_table',
    '_exodus__ex_get_var',
    '_exodus__ex_get_variable_name',
    '_exodus__ex_get_variable_names',
    '_exodus__ex_get_variable_param',
    '_exodus__ex_inquire_float',
    '_exodus__ex_inquire_int',
    '_exodus__ex_put_assemblies',
    '_exodus__ex_put_assembly',
    '_exodus__ex_put_attribute',
    '_exodus__ex_put_block',
    '_exodus__ex_put_concat_elem_blk',
    '_exodus__ex_put_coord',
    '_exodus__ex_put_coord_names',
    '_exodus__ex_put_elem_attr',
    '_exodus__ex_put_elem_attr_names',
    '_exodus__ex_put_elem_conn',
    '_exodus__ex_put_id_map',
    '_exodus__ex_put_info',
    '_exodus__ex_put_info_recs',
    '_exodus__ex_put_name',
    '_exodus__ex_put_names',
    '_exodus__ex_put_node_set',
    '_exodus__ex_put_node_set_dist_fact',
    '_exodus__ex_put_one_attr',
    '_exodus__ex_put_prop',
    '_exodus__ex_put_qa',
    '_exodus__ex_put_reduction_variable_name',
    '_exodus__ex_put_reduction_variable_param',
    '_exodus__ex_put_reduction_vars',
    '_exodus__ex_put_set_param',
    '_exodus__ex_put_side_set',
    '_exodus__ex_put_side_set_dist_fact',
    '_exodus__ex_put_time',
    '_exodus__ex_put_truth_table',
    '_exodus__ex_put_var',
    '_exodus__ex_put_variable_name',
    '_exodus__ex_put_variable_param',
    '_exodus__ex_update',
    '_exodus__open',
    # 'close',
    'copy',
    'elem_blk_info',
    'elem_type',
    'get_all_global_variable_values',
    'get_all_node_set_params',
    'get_all_side_set_params',
    'get_assemblies',
    'get_assembly',
    'get_attribute_count',
    'get_attributes',
    'get_blob',
    'get_coord',
    'get_coord_names',
    'get_coords',
    'get_elem_attr',
    'get_elem_attr_values',
    'get_elem_blk_ids',
    'get_elem_blk_name',
    'get_elem_blk_names',
    'get_elem_connectivity',
    'get_elem_id_map',
    'get_elem_order_map',
    'get_element_attribute_names',
    'get_element_property_names',
    'get_element_property_value',
    'get_element_variable_names',
    'get_element_variable_number',
    'get_element_variable_truth_table',
    'get_element_variable_values',
    'get_global_variable_names',
    'get_global_variable_number',
    'get_global_variable_value',
    'get_global_variable_values',
    'get_id_map',
    'get_ids',
    'get_info_records',
    'get_name',
    'get_names',
    'get_node_id_map',
    'get_node_set_dist_facts',
    'get_node_set_ids',
    'get_node_set_name',
    'get_node_set_names',
    'get_node_set_nodes',
    'get_node_set_params',
    'get_node_set_property_names',
    'get_node_set_property_value',
    'get_node_set_variable_names',
    'get_node_set_variable_number',
    'get_node_set_variable_truth_table',
    'get_node_set_variable_values',
    'get_node_variable_names',
    'get_node_variable_number',
    'get_node_variable_values',
    'get_partial_element_variable_values',
    'get_partial_node_set_variable_values',
    'get_partial_node_variable_values',
    'get_partial_side_set_variable_values',
    'get_qa_records',
    'get_reduction_variable_names',
    'get_reduction_variable_number',
    'get_reduction_variable_values',
    'get_set_params',
    'get_side_set',
    'get_side_set_dist_fact',
    'get_side_set_ids',
    'get_side_set_name',
    'get_side_set_names',
    'get_side_set_node_list',
    'get_side_set_params',
    'get_side_set_property_names',
    'get_side_set_property_value',
    'get_side_set_variable_names',
    'get_side_set_variable_number',
    'get_side_set_variable_truth_table',
    'get_side_set_variable_values',
    'get_times',
    'get_variable_names',
    'get_variable_number',
    'get_variable_truth_table',
    'inquire',
    'num_assembly',
    'num_attr',
    'num_blks',
    'num_blob',
    'num_dimensions',
    'num_elems',
    'num_elems_in_blk',
    'num_faces_in_side_set',
    'num_info_records',
    'num_node_sets',
    'num_nodes',
    'num_nodes_in_node_set',
    'num_nodes_per_elem',
    'num_qa_records',
    'num_side_sets',
    'num_times',
    'put_all_global_variable_values',
    'put_assemblies',
    'put_assembly',
    'put_attribute',
    'put_concat_elem_blk',
    'put_coord_names',
    'put_coords',
    'put_elem_attr',
    'put_elem_attr_names',
    'put_elem_attr_values', 
    'put_elem_blk_info', 
    'put_elem_blk_name',
    'put_elem_blk_names',
    'put_elem_connectivity',
    'put_elem_face_conn',
    'put_elem_id_map',
    'put_element_attribute_names',
    'put_element_property_value',
    'put_element_variable_name', 
    'put_element_variable_values',
    'put_face_count_per_polyhedra',
    'put_face_node_conn',
    'put_global_variable_name',
    'put_global_variable_value',
    'put_id_map',
    'put_info',
    'put_info_ext',
    'put_info_records',
    'put_name',
    'put_names',
    'put_node_count_per_face',
    'put_node_id_map',
    'put_node_set',
    'put_node_set_dist_fact',
    'put_node_set_name',
    'put_node_set_names',
    'put_node_set_params',
    'put_node_set_property_value',
    'put_node_set_variable_name',
    'put_node_set_variable_values',
    'put_node_variable_name',
    'put_node_variable_values',
    'put_polyhedra_elem_blk',
    'put_polyhedra_face_blk',
    'put_qa_records',
    'put_reduction_variable_name',
    'put_reduction_variable_values',
    'put_set_params',
    'put_side_set',
    'put_side_set_dist_fact',
    'put_side_set_name',
    'put_side_set_names',
    'put_side_set_params',
    'put_side_set_property_value',
    'put_side_set_variable_name',
    'put_side_set_variable_values',
    'put_time',
    'put_variable_name',
    'set_element_variable_number',
    'set_element_variable_truth_table',
    'set_global_variable_number',
    'set_node_set_variable_number',
    'set_node_set_variable_truth_table',
    'set_node_variable_number',
    'set_reduction_variable_number',
    'set_side_set_variable_number',
    'set_side_set_variable_truth_table',
    'set_variable_number',
    'set_variable_truth_table',
    'summarize',
    'title',
    'version_num']

_attr_inputs = {a: [] for a in _dir_exodus_full}

# _attr_inputs['close'] = []
_attr_inputs['copy'] = ['test_file.g']
_attr_inputs['elem_blk_info'] = [1]
_attr_inputs['elem_type'] = [1]
_attr_inputs['get_all_global_variable_values'] = [1]
# _attr_inputs['get_all_node_set_params'] = []
# _attr_inputs['get_all_side_set_params'] = []
# _attr_inputs['get_assemblies'] = []
# _attr_inputs['get_assembly'] = []
# _attr_inputs['get_attribute_count'] = []
# _attr_inputs['get_attributes'] = []
# _attr_inputs['get_blob'] = []
_attr_inputs['get_coord'] = [1]
# _attr_inputs['get_coord_names'] = []
# _attr_inputs['get_coords'] = []
_attr_inputs['get_elem_attr'] = [1]
_attr_inputs['get_elem_attr_values'] = [1, 'test_attr']
# _attr_inputs['get_elem_blk_ids'] = []
_attr_inputs['get_elem_blk_name'] = [1]
# _attr_inputs['get_elem_blk_names'] = []
_attr_inputs['get_elem_connectivity'] = [1]
# _attr_inputs['get_elem_id_map'] = []
# _attr_inputs['get_elem_num_map'] = []
# _attr_inputs['get_elem_order_map'] = []
_attr_inputs['get_element_attribute_names'] = [1]
# _attr_inputs['get_element_property_names'] = []
_attr_inputs['get_element_property_value'] = [1, 'ID']
# _attr_inputs['get_element_variable_names'] = []
_attr_inputs['get_element_variable_number'] = []
# _attr_inputs['get_element_variable_truth_table'] = []
_attr_inputs['get_element_variable_values'] = [1, 'var_elem_test', 1]
# _attr_inputs['get_global_variable_names'] = []
# _attr_inputs['get_global_variable_number'] = []
_attr_inputs['get_global_variable_value'] = ['momentum', 1]
_attr_inputs['get_global_variable_values'] = ['momentum']
# _attr_inputs['get_id_map'] = []
# _attr_inputs['get_ids'] = []
# _attr_inputs['get_info_records'] = []
# _attr_inputs['get_name'] = []
# _attr_inputs['get_names'] = []
# _attr_inputs['get_node_id_map'] = []
# _attr_inputs['get_node_num_map'] = []
_attr_inputs['get_node_set_dist_facts'] = [1]
# _attr_inputs['get_node_set_ids'] = []
_attr_inputs['get_node_set_name'] = [1]
# _attr_inputs['get_node_set_names'] = []
_attr_inputs['get_node_set_nodes'] = [1]
_attr_inputs['get_node_set_params'] = [1]
# _attr_inputs['get_node_set_property_names'] = []
# _attr_inputs['get_node_set_property_value'] = []
# _attr_inputs['get_node_set_variable_names'] = []
# _attr_inputs['get_node_set_variable_number'] = []
# _attr_inputs['get_node_set_variable_truth_table'] = []
# _attr_inputs['get_node_set_variable_values'] = []
_attr_inputs['get_node_variable_names'] = []
_attr_inputs['get_node_variable_number'] = []
_attr_inputs['get_node_variable_values'] = ['var_node_test', 1]
# _attr_inputs['get_partial_element_variable_values'] = []
# _attr_inputs['get_partial_node_set_variable_values'] = []
# _attr_inputs['get_partial_node_variable_values'] = []
# _attr_inputs['get_partial_side_set_variable_values'] = []
# _attr_inputs['get_qa_records'] = []
# _attr_inputs['get_reduction_variable_names'] = []
# _attr_inputs['get_reduction_variable_number'] = []
# _attr_inputs['get_reduction_variable_values'] = []
# _attr_inputs['get_set_params'] = []
_attr_inputs['get_side_set'] = [1]
_attr_inputs['get_side_set_dist_fact'] = [1]
# _attr_inputs['get_side_set_ids'] = []
_attr_inputs['get_side_set_name'] = [1]
# _attr_inputs['get_side_set_names'] = []
# _attr_inputs['get_side_set_node_list'] = []
_attr_inputs['get_side_set_params'] = [1]
# _attr_inputs['get_side_set_property_names'] = []
_attr_inputs['get_side_set_property_value'] = [1, 'ID']
# _attr_inputs['get_side_set_variable_names'] = []
# _attr_inputs['get_side_set_variable_number'] = []
# _attr_inputs['get_side_set_variable_truth_table'] = []
# _attr_inputs['get_side_set_variable_values'] = []
# _attr_inputs['get_times'] = []
# _attr_inputs['get_variable_names'] = []
# _attr_inputs['get_variable_number'] = []
# _attr_inputs['get_variable_truth_table'] = []
# _attr_inputs['inquire'] = []
# _attr_inputs['num_assembly'] = []
_attr_inputs['num_attr'] = [1]
# _attr_inputs['num_blks'] = []
# _attr_inputs['num_blob'] = []
# _attr_inputs['num_dimensions'] = []
# _attr_inputs['num_elems'] = []
_attr_inputs['num_elems_in_blk'] = [1]
_attr_inputs['num_faces_in_side_set'] = [1]
# _attr_inputs['num_info_records'] = []
# _attr_inputs['num_node_sets'] = []
# _attr_inputs['num_nodes'] = []
_attr_inputs['num_nodes_in_node_set'] = [1]
_attr_inputs['num_nodes_per_elem'] = [1]
# _attr_inputs['num_qa_records'] = []
# _attr_inputs['num_side_sets'] = []
# _attr_inputs['num_times'] = []
_attr_inputs['put_all_global_variable_values'] = [1, [1.]]
# _attr_inputs['put_assemblies'] = []
# _attr_inputs['put_assembly'] = []
# _attr_inputs['put_attribute'] = []
_attr_inputs['put_concat_elem_blk'] = [[1], ['HEX8'], [3], [8], [1], True]
_attr_inputs['put_coord_names'] = [['x', 'y', 'z']]
_attr_inputs['put_coords'] = [1, 1, 1]
_attr_inputs['put_elem_attr'] = [1, 1.]
_attr_inputs['put_elem_attr_names'] = [1, ['test_name']]
_attr_inputs['put_elem_attr_values'] = [1, 'test_name', [1]]
_attr_inputs['put_elem_blk_info'] = [1, 'HEX8', 1, 1, 1]
_attr_inputs['put_elem_blk_name'] = [1, 'test_name']
_attr_inputs['put_elem_blk_names'] = [['test_name_1']]
_attr_inputs['put_elem_connectivity'] = [1, np.zeros((1, 8), dtype=int)]
# _attr_inputs['put_elem_face_conn'] = []
_attr_inputs['put_elem_id_map'] = [[1]]
_attr_inputs['put_element_attribute_names'] = [1, ['test_attr_name']]
_attr_inputs['put_element_property_value'] = [1, 'ID', 1]
_attr_inputs['put_element_variable_name'] = ['test_var_name', 0]
_attr_inputs['put_element_variable_values'] = [1, 'test_var_name', 0, [1.]]
# _attr_inputs['put_face_count_per_polyhedra'] = []
# _attr_inputs['put_face_node_conn'] = []
_attr_inputs['put_global_variable_name'] = ['glo_var_name', 1]
_attr_inputs['put_global_variable_value'] = ['glo_var_name', 1, 1.]
# _attr_inputs['put_id_map'] = []
# _attr_inputs['put_info'] = []
# _attr_inputs['put_info_ext'] = []
_attr_inputs['put_info_records'] = [['']]
# _attr_inputs['put_name'] = []
# _attr_inputs['put_names'] = []
# _attr_inputs['put_node_count_per_face'] = []
_attr_inputs['put_node_id_map'] = [[1, 2, 3, 4, 5, 6, 7, 8]]
_attr_inputs['put_node_set'] = [1, [1, 2, 3, 4]]
_attr_inputs['put_node_set_dist_fact'] = [1, [1, 2, 3, 4]]
_attr_inputs['put_node_set_name'] = [1, 'test_ns_name']
_attr_inputs['put_node_set_names'] = [[str(i) for i in range(1, 15)]]
_attr_inputs['put_node_set_params'] = [1, 1]
# _attr_inputs['put_node_set_property_value'] = []
# _attr_inputs['put_node_set_variable_name'] = []
# _attr_inputs['put_node_set_variable_values'] = []
_attr_inputs['put_node_variable_name'] = ['test_nv_name', 1]
_attr_inputs['put_node_variable_values'] = [
    'test_nv_name', 1, [1, 2, 3, 4, 5, 6, 7, 8]]
# _attr_inputs['put_polyhedra_elem_blk'] = []
# _attr_inputs['put_polyhedra_face_blk'] = []
_attr_inputs['put_qa_records'] = [[
    ('str1', 'str2', 'str3', 'str4'), ('str5', 'str6', 'str7', 'str8')]]
# _attr_inputs['put_reduction_variable_name'] = []
# _attr_inputs['put_reduction_variable_values'] = []
# _attr_inputs['put_set_params'] = []
_attr_inputs['put_side_set'] = [1, [1], [1]]
_attr_inputs['put_side_set_dist_fact'] = [1, [1]]
_attr_inputs['put_side_set_name'] = [1, 'test_ss_name']
_attr_inputs['put_side_set_names'] = [[str(i) for i in range(1, 7)]]
_attr_inputs['put_side_set_params'] = [1, 1]
_attr_inputs['put_side_set_property_value'] = [1, 'ID', 1]
# _attr_inputs['put_side_set_variable_name'] = []
# _attr_inputs['put_side_set_variable_values'] = []
_attr_inputs['put_time'] = [2, 2.]
# _attr_inputs['put_variable_name'] = []
_attr_inputs['set_element_variable_number'] = [1]
_attr_inputs['set_element_variable_truth_table'] = [[1]]
_attr_inputs['set_global_variable_number'] = [1]
_attr_inputs['set_node_set_variable_number'] = [0]
_attr_inputs['set_node_set_variable_truth_table'] = [[1]]
# _attr_inputs['set_node_variable_number'] = []
# _attr_inputs['set_reduction_variable_number'] = []
_attr_inputs['set_side_set_variable_number'] = [1]
_attr_inputs['set_side_set_variable_truth_table'] = [[1]]
# _attr_inputs['set_variable_number'] = []
# _attr_inputs['set_variable_truth_table'] = []
# _attr_inputs['summarize'] = []
# _attr_inputs['title'] = []
# _attr_inputs['version_num']
