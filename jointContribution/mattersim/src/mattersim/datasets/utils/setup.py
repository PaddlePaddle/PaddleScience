from distutils.core import Extension
from distutils.core import setup

import numpy
from Cython.Build import cythonize

package = Extension(
    "threebody_indices", ["threebody_indices.pyx"], include_dirs=[numpy.get_include()]
)
setup(ext_modules=cythonize([package]))
