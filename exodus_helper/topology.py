"""This module contains the RectangularPrism class, derived from the core
Exodus class, and external functions related to querying objects of this type.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import re

import numpy as np

from .core import CONNECTIVITY_SIDES
from .core import NUM_NODES_SIDE
from .core import Exodus


# --------------------------------------------------------------------------- #

# exodus_helper rectangular prism surface id canonical order
# 1:x-, 2:x+, 3:y-, 4:y+, 5:z-, 6:z+
PICK_SURFACE = {
    1: (np.min, 0),
    2: (np.max, 0),
    3: (np.min, 1),
    4: (np.max, 1),
    5: (np.min, 2),
    6: (np.max, 2)}


# --------------------------------------------------------------------------- #

class RectangularPrism(Exodus):
    """Specialization of the Exodus class for rectangular prism geometries.
        This geometry simplifies eg identifying elements on a surface"""

    def __init__(
            self, filename, shape=(1, 1, 1), resolution=(1., 1., 1.),
            num_attr=0, mode='r', **kwargs):
        """Create an `Exodus` mesh object for a rectangular prism.

        Args:
            filename (str): File containing mesh data.
            shape (tuple(int), optional): The shape of the prism as a length-3
                `tuple`. Defaults to (1, 1, 1).
            resolution (tuple(float), optional): The (spatially invariant)
                resolution of each element. Defaults to (1., 1., 1.).
            num_attr (int, optional): The number of attributes for each element
                block. Defaults to 0.
            **kwargs: Variable length keyword arguments to pass to
                Exodus.__init__.

        Returns:
            Exodus: An `Exodus` mesh for a rectangular prism.
        """

        if mode == 'r':
            super().__init__(filename, mode='r', **kwargs)
            return

        num_elements = np.prod(shape)
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

        coords_x = resolution[0] * idxs_xg
        coords_y = resolution[1] * idxs_yg
        coords_z = resolution[2] * idxs_zg

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

        connectivity = np.zeros((num_elements, 8), dtype=int)
        num_xy = num_nodes_x * num_nodes_y
        for idx_z in range(shape[2]):
            for idx_y in range(shape[1]):
                for idx_x in range(shape[0]):
                    i = idx_z * shape[0] * shape[1] + idx_y * shape[0] + idx_x
                    base = idx_y * num_nodes_x + idx_x + idx_z * num_xy
                    connectivity[i, :] = [
                        base, base + 1,
                        base + num_nodes_x + 1,
                        base + num_nodes_x,
                        base + num_xy,
                        base + 1 + num_xy,
                        base + num_nodes_x + 1 + num_xy,
                        base + num_nodes_x + num_xy]

        num_node_nss = [
            num_nodes_y * num_nodes_z,
            num_nodes_y * num_nodes_z,
            num_nodes_x * num_nodes_z,
            num_nodes_x * num_nodes_z,
            num_nodes_x * num_nodes_y,
            num_nodes_x * num_nodes_y,
            1, 1, 1, 1, 1, 1, 1, 1]

        num_side_sss = [
            shape[1] * shape[2],
            shape[1] * shape[2],
            shape[0] * shape[2],
            shape[0] * shape[2],
            shape[0] * shape[1],
            shape[0] * shape[1]]

        elem_blk_info = kwargs.get(
            'elem_blk_info',
            ([1], ['HEX8'], [num_elements], [8], [num_attr], True))

        num_el_blk = len(elem_blk_info[0])

        super().__init__(
            filename,
            mode='w',
            num_info=1,
            time_step=1,
            num_dim=3,
            num_nodes=num_nodes,
            num_elem=num_elements,
            num_el_blk=num_el_blk,
            num_node_sets=14,
            num_side_sets=6,
            num_processors=1,
            num_procs_file=1,
            num_nod_nss=num_node_nss,
            num_side_sss=num_side_sss,
            title='rectangular_prism',
            **kwargs)

        variables = self.dataset.variables

        self.put_coords(coords_x, coords_y, coords_z)
        self.put_concat_elem_blk(*elem_blk_info)
        if num_el_blk == 1:
            variables['connect1'][:] = connectivity + 1

        node_set_nodes = []
        side_set_elements = []
        side_set_sides = []
        for id_surface in range(1, 7):
            node_set_nodes.append(self.get_nodes_on_surface(id_surface))
            elems, sides = self.get_elems_sides_on_surface(id_surface)
            side_set_elements.append(elems)
            side_set_sides.append(sides)
        for x in range(2):
            for y in range(2, 4):
                for z in range(4, 6):
                    nodes_xy = np.intersect1d(
                        node_set_nodes[x], node_set_nodes[y])
                    intersect = np.intersect1d(nodes_xy, node_set_nodes[z])
                    node_set_nodes.append(intersect)

        for idx, nodes in enumerate(node_set_nodes):
            self.put_node_set_params(idx + 1, len(nodes), numSetDistFacts=1)
            self.put_node_set(idx + 1, nodes)

        for idx, element in enumerate(side_set_elements):
            num_sides = len(side_set_sides[idx])
            self.put_side_set_params(idx + 1, num_sides, numSetDistFacts=4)
            self.put_side_set(idx + 1, element, side_set_sides[idx])

        self.put_elem_id_map(range(1, num_elements + 1))
        self.put_node_id_map(range(1, num_nodes + 1))

    @property
    def resolution(self):
        return self.get_resolution()

    @property
    def shape(self):
        """tuple(int, int, int): The shape of the mesh."""
        return self.get_shape()

    def get_elems_on_surface(self, surface):
        return self.get_elems_sides_on_surface(surface)[0]

    def get_elements_on_surface(self, surface):
        return self.get_elems_on_surface(surface)

    def get_elems_on_surface_sorted(self, surface):
        elems = self.get_elems_on_surface(surface)
        centroids = self.get_elem_centroids()[elems - 1]
        dims = get_dims_surface(surface)
        idxs_sort = np.lexsort([centroids[:, d] for d in dims])
        return elems[idxs_sort]

    def get_elements_on_surface_sorted(self, surface):
        return self.get_elems_on_surface_sorted(surface)

    def get_elems_sides_on_surface(self, surface):
        nodes_surface = self.get_nodes_on_surface(surface)
        idxs_start = self.get_idxs_elem_start_blk()
        elems = []
        sides = []
        for id_blk, idx_start in zip(self.get_elem_blk_ids(), idxs_start):
            elem_type, _, num_nodes, _ = self.get_elem_blk_info(id_blk)
            num_nodes = NUM_NODES_SIDE[elem_type]
            connectivity_info = self.get_elem_connectivity(id_blk)
            connectivity = connectivity_info[0].reshape(connectivity_info[1:])
            idxs_surf_elem = get_idxs_surface_elem(
                nodes_surface, connectivity, idx_start)
            faces_surface = [
                i for i in idxs_surf_elem if len(i[1]) == num_nodes]
            connectivity_sides = CONNECTIVITY_SIDES[
                re.sub(r'\d+', '', elem_type)]
            num_check = len(list(connectivity_sides.keys())[0])
            for face in faces_surface:
                elems.append(face[0])
                sides.append(connectivity_sides[tuple(face[1][:num_check])])
        return np.array(elems, dtype=int), np.array(sides, dtype=int)

    def get_elements_sides_on_surface(self, surface):
        return self.get_elems_sides_on_surface(surface)

    def get_elems_sorted(self, order='xyz'):
        if isinstance(order, str):
            dims = {'x': 0, 'y': 1, 'z': 2}
            order = [dims[o] for o in order]
        centroids = self.get_elem_centroids()
        return np.lexsort([centroids[:, o] for o in order]) + 1

    def get_elements_sorted(self, order='xyz'):
        return self.get_elems_sorted(order=order)

    def get_nodes_on_surface(self, surface):
        coordinates = np.column_stack(self.get_coords())
        side, dim = PICK_SURFACE[surface]
        nodes = np.where(
            np.isclose(side(coordinates[:, dim]), coordinates[:, dim]))[0] + 1
        return nodes

    def get_patches_on_surface(self, surface):
        nodes_surface = self.get_nodes_on_surface(surface)
        coordinates = np.column_stack(self.get_coords())
        idxs_start = self.get_idxs_elem_start_blk()
        patches = []
        for id_blk, idx_start in zip(self.get_elem_blk_ids(), idxs_start):
            elem_type, _, num_nodes, _ = self.get_elem_blk_info(id_blk)
            num_nodes = NUM_NODES_SIDE[elem_type]
            connectivity_info = self.get_elem_connectivity(id_blk)
            connectivity = connectivity_info[0].reshape(connectivity_info[1:])
            nodes_surface_elem = get_nodes_surface_elem(
                nodes_surface, connectivity, idx_start)
            faces_surface = [
                n for n in nodes_surface_elem if len(n[1]) == num_nodes]
            connectivity_sides = CONNECTIVITY_SIDES[
                re.sub(r'\d+', '', elem_type)]
            num_check = len(list(connectivity_sides.keys())[0])
            for face in faces_surface:
                patches.append(np.array(
                    [coordinates[f - 1] for f in face[1][:num_check]]))
        return patches

    def get_resolution(self):
        conn = self.get_elem_connectivity_full()
        coords = np.column_stack(self.get_coords())
        return np.max(np.abs(np.diff(coords[conn[0] - 1], axis=0)), axis=0)

    def get_shape(self):
        num_xs = self.numElem // self.get_elements_on_surface(1).size
        num_ys = self.numElem // self.get_elements_on_surface(3).size
        num_zs = self.numElem // self.get_elements_on_surface(5).size
        assert num_xs * num_ys * num_zs == self.numElem
        return (num_xs, num_ys, num_zs)

    def get_shape_surface(self, surface):
        dims = get_dims_surface(surface)
        shape = self.get_shape()
        return tuple(shape[d] for d in dims)

    def get_sides_on_surface(self, surface):
        return self.get_elems_sides_on_surface(surface)[1]


def get_dims_surface(surface):
    """Get the indices of the dimensions that span a surface.

    The surface ID's for a rectangular prism are:

        1:x-, 2:x+, 3:y-, 4:y+, 5:z-, 6:z+

    Args:
        surface (int): Surface ID.

    Returns:
        A `numpy.ndarray` of dimension indicies (ints).
    """
    dim = PICK_SURFACE[surface][1]
    return np.sort([(dim - 1) % 3, (dim + 1) % 3])


def get_idxs_surface_elem(nodes, connectivity, idx_start=0):
    shift = idx_start + 1
    itr = enumerate(connectivity)
    return [(i + shift, np.where(np.in1d(c, nodes))[0] + 1) for i, c in itr]


def get_nodes_surface_elem(nodes, connectivity, idx_start=0):
    idxs = get_idxs_surface_elem(nodes, connectivity)
    return [(i[0] + idx_start, connectivity[i[0] - 1, i[1] - 1]) for i in idxs]

# --------------------------------------------------------------------------- #
