from distutils.core import setup, Extension

module1 = Extension('slic',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                     include_dirs = ['/Users/Chris/miniconda/envs/ilastik-devel/include/',
                                      '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/',
                                      '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/numpy',
                                      '/usr/local/include/'],
                    libraries = ['opencv_core','opencv_highgui'],
                    library_dirs = ['/usr/local/lib/'],
                    sources = ['slicmodule.cpp', 'LKM.cpp', 'utils.cpp']) #, language='c++')

setup (name = 'slic',
       version = '1.0',
       description = 'This is a python wrapper for the SLIC library, ',
       author = 'Christian Jaques',
       author_email = 'christian.jaques@gmail.com',
       url = 'http://cvlab.epfl.ch',
       long_description = '''
       C.f. lab website''',
       ext_modules = [module1])
