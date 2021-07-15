#!/usr/bin/env python

"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
import os
import sys
import subprocess
import warnings
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy
# from Cython.Build import cythonize

# ------------------------------ Env check utils -------------------------------

def shell_call(cmd):
    """Run shell command `cmd` in the shell and return output. Returns empty
    string if `cmd` returns non-zero.
    """
    try:
        output = subprocess.check_output(cmd, shell=sys.platform.startswith('win'))
    except (OSError, subprocess.CalledProcessError):
        return str()
    else:
        return output.decode(sys.stdout.encoding or 'utf-8').strip()
        # return str().strip()


def get_gsl_config():
    """Retrieve information about GNU GSL configuration, using shell cmd
    'gsl-config', which is assumed to be in the $PATH. Returns dictionary
    containing GSL version, library dir, and include dir.
    """
    cfg = {
        'version': shell_call(['gsl-config', '--version']).split('.'),
        'libs': shell_call(['gsl-config', '--libs']).split(),
        'cflags': shell_call(['gsl-config', '--cflags']).split(),
        'prefix': shell_call(['gsl-config', '--prefix']),
    }
    return cfg

# ------------------------------------------------------------------------------

# Initialize variables
repo_url = 'https://github.com/eho-tacc/episimlab'
pkg_dir = 'episimlab'

# Determines whether to cythonize extensions or compile from *.c
USE_CYTHON = True
src_ext = '.pyx' if USE_CYTHON else '.c'

# Check GNU GSL install
gsl_config = get_gsl_config()
try:
    assert int(gsl_config['version'][0]) > 0
except (AssertionError, IndexError, ValueError) as err:
    raise Exception("GNU GSL does not appear to be installed; could not find " +
                    "`gsl-config` in $PATH. Please add `gsl-config` to your " +
                    "$PATH, or install GNU GSL if necessary. To install GSL " +
                    "on Mac OS X: brew install gsl.")

# cy_model extensions
gsl_lib = dict(language='c',
               libraries=['gsl'],
               include_dirs=[os.path.join(gsl_config['prefix'], 'include')],
               library_dirs=[os.path.join(gsl_config['prefix'], 'lib')])
extensions = [
    Extension('episimlab.cy_utils.cy_utils',
              sources=[f"episimlab/cy_utils/cy_utils{src_ext}"],
              **gsl_lib),
    Extension('episimlab.seir.bf_cython_engine',
              sources=[f"episimlab/seir/bf_cython_engine{src_ext}"],
              **gsl_lib),
    Extension('episimlab.foi.bf_cython_engine',
              sources=[f"episimlab/foi/bf_cython_engine.pyx"],
              **gsl_lib)

]

# Cythonize extensions
if USE_CYTHON is True:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, annotate=True,
                           compiler_directives={'language_level': "3"})

# Create version file
VERSION = "1.0.0"
with open(os.path.join(pkg_dir,'version.py'), 'w') as VF:
    cnt = """# THIS FILE IS GENERATED FROM SETUP.PY\nversion = '%s'"""
    VF.write(cnt%(VERSION))

# Get README contents
README_fp = "README.md"
if os.path.isfile(README_fp):
    with open(README_fp, 'r') as f:
        README_contents = f.read()
else:
    README_contents = ""


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='episimlab',  # package name
    version=VERSION,  # Required
    description='Environment for epidemiological model development',
    long_description=README_contents,
    long_description_content_type='text/markdown',
    url=repo_url,
    author='Ethan Ho',
    author_email='eho@tacc.utexas.edu',
    classifiers=[  # https://pypi.org/classifiers/
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha', # update this as necessary
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: BSD License',
	# Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='seir model covid-19',
    packages=["episimlab"],
    package_data={},
    include_package_data=True,
    python_requires='>=3.6, <4',
    install_requires=[
        'pandas>=1,<2',
        'scipy',
        'numpy',
        'xarray',
        'xarray-simlab',
        'Cython'
    ],
    extras_require={ #   $ pip install sampleproject[dev]
        'dev': ['pytest','tox'],
        'test': ['pytest','tox']
    },
    entry_points={  # Optional
        'console_scripts': [
            'episimlab=episimlab:main'
        ],
    },
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={  # Optional
        'Bug Reports': os.path.join(repo_url,'issues'),
        'COVID portal': 'https://covid-19.tacc.utexas.edu/projections/',
        'Meyers Lab': 'http://www.bio.utexas.edu/research/meyers/'
    },
    # Extensions and included libs
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
)
