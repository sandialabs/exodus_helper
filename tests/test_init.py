'''Module for testing initialization of exodus_helper'''

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
