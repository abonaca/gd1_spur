from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("interact", ["interact_wrap.c", "interact.c", "nr_rand.c", "utility.c"])

setup(ext_modules = [c_ext], 
      include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs())
