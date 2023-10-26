"""This module contains the import statements that build the core
exodus_helper package attributes.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import importlib.metadata
from packaging.version import parse
__version__ = importlib.metadata.version(__package__)
__version_info__ = parse(__version__).release

from .core import CONNECTIVITY_SIDES
from .core import Exodus
from .core import add_variable
from .core import get_data_exodus

from .element_calculations import calculate_volume_element
from .element_calculations import calculate_volume_hexahedron
from .element_calculations import calculate_volumes_block
from .element_calculations import calculate_volumes_element

from .render_mesh import render_mesh
from .render_mesh import map_points_to_elements

from .topology import RectangularPrism

from .reconfigure_mesh import create_sets_canonical
from .reconfigure_mesh import scale_mesh

# --------------------------------------------------------------------------- #
