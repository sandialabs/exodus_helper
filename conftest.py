"""This module contains pytest fixtures and related functionality for unit
testing the exodus_helper package.

Part of exodus_helper 1.0: Copyright 2023 Sandia Corporation
This Software is released under the BSD license detailed in the file
`license.txt` in the top-level directory"""

# --------------------------------------------------------------------------- #

import datetime
import fnmatch
import os
import shutil

import git
import pytest

from exodus_helper import RectangularPrism


# --------------------------------------------------------------------------- #

def get_files(
        path='.', recursive=True, ignore_dirs=['.*'], ignore_files=['*.pyc']):

    files = []
    walker = os.walk(path)
    if not recursive:
        walker = [next(walker)]
    for root, dirnames, filenames in walker:
        hierarchy = root.split(os.path.sep)[1:]
        skip = False
        for ignore in ignore_dirs:
            if any([fnmatch.fnmatch(d, ignore) for d in hierarchy]):
                skip = True
                break
        if skip:
            continue
        filtered = []
        for ignore in ignore_files:
            filtered.extend(fnmatch.filter(filenames, ignore))
        files.extend([
            os.path.join(root, f) for f in filenames if f not in filtered])
    return files


def pytest_addoption(parser):

    try:
        parser.addoption(
            '--no_commit', action='store_true',
            help='Do not retrieve current commit of local repository HEAD')
    except ValueError:
        pass

    try:
        parser.addoption(
            '--no_timestamp', action='store_true',
            help='Do not print timestamp during test collection')
    except ValueError:
        pass


# called after command line parse
def pytest_configure(config):

    no_commit = config.getoption('no_commit')
    no_timestamp = config.getoption('no_timestamp')

    if not no_commit or not no_timestamp:

        shape_terminal = shutil.get_terminal_size()
        tag = 'metadata'
        tag_p = ' ' + tag + ' '
        pad = (shape_terminal[0] - len(tag_p)) // 2
        print('=' * pad + tag_p + '=' * pad)

    if not no_commit:

        repo = git.Repo(path=config.rootdir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f'Git repository at {sha}')
        config.option.no_commit = True

    if not no_timestamp:
        date = datetime.datetime.today().isoformat(' ')
        print(f'Session initialized at {date}')
        config.option.no_timestamp = True


@pytest.fixture()
def shape_terminal():
    return shutil.get_terminal_size()


@pytest.fixture(scope='module')
def dir_test_file(request):
    return request.fspath.dirname


@pytest.fixture(scope='module')
def mesh(dir_test_file):
    filename = os.path.join(dir_test_file, 'test_mesh.g')
    shape = (1, 1, 2)
    res = (1., 1., 1.)
    num_attr = 2
    yield RectangularPrism(
        filename, shape=shape, resolution=res, num_attr=num_attr,
        num_nod_var=2, mode='w')
    os.remove(filename)


@pytest.fixture(autouse=True)
def run_around_tests(dir_test_file):
    # Check working directory before the test
    dir_before = os.getcwd()
    # Record files existing before the test
    ignore_files = ['missing_tests.txt', '.nfs*', '*.pyc']
    files_before = set(get_files(dir_test_file, ignore_files=ignore_files))
    # A test function will be run at this point
    yield
    # Check working directory after the test
    dir_after = os.getcwd()
    # Make sure no test changes the working directory
    assert dir_before == dir_after
    # Record files existing after the test
    files_after = set(get_files(dir_test_file, ignore_files=ignore_files))
    # Make sure there are no changes
    assert files_before.symmetric_difference(files_after) == set()


@pytest.fixture(autouse=True, scope='session')
def run_around_session():
    # Record files existing before the test session
    ignore_files = ['missing_tests.txt', '.nfs*', '*.pyc']
    files_before = set(get_files(ignore_files=ignore_files))
    # The test session will be run at this point
    yield
    # Record files existing after the test session
    files_after = set(get_files(ignore_files=ignore_files))
    # Make sure there are no changes
    assert files_before.symmetric_difference(files_after) == set()

# --------------------------------------------------------------------------- #
