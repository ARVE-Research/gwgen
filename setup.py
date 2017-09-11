from setuptools import find_packages
from setuptools.command.test import test as TestCommand
from numpy.distutils.core import Extension
import sys
import os
import os.path as osp


# conda skeleton applies a patch that is not realised by
# numpy.distutils.core.setup. Therefore we give the user the possibility to
# choose to use the setuptools. Note that conda skeleton not actually installs
# the package but just saves the information from the setup call
if os.getenv('RUNNING_SKELETON', None):
    from setuptools import setup
else:
    from numpy.distutils.core import setup

parseghcnrow = Extension(
    name='gwgen._parseghcnrow', sources=[
        osp.join('gwgen', 'mo_parseghcnrow.f90')])
parseeecra = Extension(
    name='gwgen._parseeecra', sources=[osp.join('gwgen', 'mo_parseeecra.f90')],
    f2py_options=['only:', 'parse_file', 'extract_data', ':'],
    extra_f90_compile_args=["-fopenmp"], extra_link_args=["-lgomp"])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


def readme():
    with open('README.rst') as f:
        return f.read()


# read the version from version.py
with open(osp.join('gwgen', 'version.py')) as f:
    exec(f.read())


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('gwgen')
    config.add_data_dir(osp.join('gwgen', 'src'))
    config.add_data_dir(osp.join('gwgen', 'data'))

    return config


install_requires = ['f90nml', 'psyplot', 'scipy', 'sqlalchemy', 'psycopg2',
                    'statsmodels', 'docrep', 'model-organization', 'xarray',
                    'six']


setup(name='gwgen',
      version=__version__,
      description='A global weather generator for daily data',
      long_description=readme(),
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: Unix',
        'Operating System :: MacOS',
      ],
      keywords='wgen weathergen ghcn eecra richardson geng',
      url='https://github.com/ARVE-Research/gwgen',
      author='Philipp Sommer',
      author_email='philipp.sommer@unil.ch',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      install_requires=install_requires,
      package_data={'gwgen': [
          'gwgen/src/*',
          'gwgen/data/*',
          ]},
      include_package_data=True,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      entry_points={'console_scripts': ['gwgen=gwgen.main:main']},
      zip_safe=False,
      ext_modules=[parseghcnrow, parseeecra],
      configuration=configuration)
