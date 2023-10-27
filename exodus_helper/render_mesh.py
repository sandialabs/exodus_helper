"""This module contains functionality related to producing graphical
representations of finite element meshes.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from .topology import get_dims_surface


# --------------------------------------------------------------------------- #

def render_mesh(mesh, surface, sample_ratio=0.5, return_coords=False):
    """Render the surface of a mesh as a binary image.

    Given a surface, numbered as such: 1:x-, 2:x+, 3:y-, 4:y+, 5:z-, 6:z+
    find all the elements associated with that surface, and create a binary
    image with 1's within the mesh, and 0's outside.

    Args:
        mesh (`Exodus`): A finite element mesh.
        surface (`int`): The surface to render.
        sample_ratio(`float`, optional): ratio of pixel size to element size

    Returns:
        A `numpy.ndarray` of binary values indicating whether each pixel falls
        within the mesh boundaries.
    """

    # Get all elements on the given surface and get their locations
    elems = mesh.get_elems_on_surface_sorted(surface)
    centroids = mesh.get_elem_centroids()[elems - 1]

    # Determine the spatial extent of the data
    dims = get_dims_surface(surface)
    x_min = np.min(centroids[:, dims[0]])
    l_x = np.max(centroids[:, dims[0]]) - x_min
    y_min = np.min(centroids[:, dims[1]])
    l_y = np.max(centroids[:, dims[1]]) - y_min

    # Compute the resolution of the rendered image
    d_a = l_x * l_y / len(centroids)
    d_x = np.sqrt(d_a) * sample_ratio
    num_x = int(l_x / d_x) + 1 if d_x > 0. else 1
    num_y = int(l_y / d_x) + 1 if d_x > 0. else 1

    # Create a rectangular grid of pixels
    (x_g, y_g) = np.meshgrid(
        x_min + d_x * np.arange(num_x), y_min + d_x * np.arange(num_y))
    coords_pixel = np.column_stack((x_g.flat, y_g.flat))

    # For each centroid, how far is each pixel
    distances = cdist(centroids[:, dims], coords_pixel)

    # Find the nodes associated with each element and triangulate
    nodes = mesh.get_elem_connectivity_full()[elems - 1]
    coords = np.column_stack(mesh.get_coords())
    hulls = [Delaunay(coords[np.ix_(n - 1, dims)]) for n in nodes]

    # For each pixel, get the closest neighboring elements
    neighborhoods = np.argsort(distances.T, axis=1)[:, :5]

    # Start with a blank image and fill in points that reside within an element
    colors = np.zeros((num_x * num_y))
    for idx, neighbors in enumerate(neighborhoods):
        for neighbor in neighbors:
            if hulls[neighbor].find_simplex(coords_pixel[idx]) >= 0:
                colors[idx] = 1
                break

    if return_coords:
        return (colors.reshape((num_y, num_x)), coords_pixel)

    return colors.reshape((num_y, num_x))


def map_points_to_elements(mesh, points):
    centroids = mesh.get_elem_centroids()
    distances = cdist(centroids, points)
    neighborhoods = np.argsort(distances.T, axis=1)
    nodes = mesh.get_elem_connectivity_full()
    coords = np.column_stack(mesh.get_coords())
    hulls = [Delaunay(coords[n - 1]) for n in nodes]
    parents = -np.ones(len(points), dtype=int)
    for idx, neighbors in enumerate(neighborhoods):
        for neighbor in neighbors:
            if hulls[neighbor].find_simplex(points[idx]) >= 0:
                parents[idx] = neighbor
                break
        if parents[idx] == -1:
            raise ValueError
    return parents

# --------------------------------------------------------------------------- #
