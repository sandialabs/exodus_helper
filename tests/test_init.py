"""This module contains tests related to the initialization of the
exodus_helper package.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import re

import exodus_helper


# --------------------------------------------------------------------------- #

def test_version_attribute():
    '''Test that version attribute has expected format'''
    version = exodus_helper.__version__
    pattern = r'\d+\.\d+\.\d.*'
    match = re.search(pattern, version, re.M)
    divs = match.group().split('.')
    ndot = len(divs)
    assert isinstance(version, str)
    assert ndot >= 3
    assert isinstance(float(divs[0]), float)
    assert isinstance(float(divs[1]), float)
    assert isinstance(float(divs[2][0]), float)

# --------------------------------------------------------------------------- #
