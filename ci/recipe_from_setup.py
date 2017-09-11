"""Add the missing informations to the conda recipe and build the conda package

This script fills in the necessary informations into meta.yaml and builds the
conda package of the given package in the psyplot framework.

Call signature::

    python build_package.py <path-to-package>
"""
import os.path as osp
import sys
import yaml
import argparse


def get_directory(s):
    return osp.dirname(osp.join(s, ''))


parser = argparse.ArgumentParser(
    description='Create a conda receipt from python source files')

parser.add_argument('package', help="The path to the Python package",
                    type=get_directory)
parser.add_argument('outdir', type=get_directory, help="""
    The output directory. The recipe will be in `outdir`/`basename package`.
    """)


args = parser.parse_args()

# Will be set down below from version.py
__version__ = None

#: The path to the package
path = args.package

#: The path for the output. It already has to include a file called
#: 'meta.template'. Otherwise, conda skeleton will be run
outdir = args.outdir


#: The name of the package
package = osp.basename(path)

# set __version__
with open(osp.join(path, package.replace('-', '_'), 'version.py')) as f:
    exec(f.read())

assert __version__ is not None, (
    "__version__ has not been set in version.py!")

version = __version__

template_file = osp.join(outdir, package, 'meta.template')

# Read the meta.template
with open(template_file) as f:
    meta = yaml.load(f)

# fill in the missing informations
meta['package']['version'] = version

if sys.platform.startswith('darwin'):
    meta['build']['script'] = (
        "unset LDFLAGS && python setup.py install "
        "--single-version-externally-managed --record record.txt  [osx]"
        )
else:
    meta['build']['script'] = (
        "python setup.py install --single-version-externally-managed "
        "--record record.txt"
        )

# write out the recipe
with open(osp.join(outdir, package, 'meta.yaml'), 'w') as f:
    yaml.dump(meta, f, default_flow_style=False)
