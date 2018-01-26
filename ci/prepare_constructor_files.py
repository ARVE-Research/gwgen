# Python script to prepare the psyplot-conda constructor files
import sys
import six
import os
import os.path as osp
import shutil
from itertools import chain
from collections import OrderedDict
import yaml
import pkg_resources
import datetime as dt
import subprocess as spr
import argparse
import glob
import pathlib


def file2html(fname):
    return pathlib.Path(osp.abspath(fname)).as_uri()


def pkg_type(s):
    return list(map(lambda s: osp.dirname(osp.join(s, '')), glob.glob(s)))

parser = argparse.ArgumentParser(
    description='Create files for the conda constructor package')

parser.add_argument('packages', nargs='*', type=pkg_type,
                    help=('The paths to the recipes to build and include as '
                          'locally built packages. They have to be installed '
                          'already!'))
parser.add_argument('-n', '--name', help=(
    "The conda environment name. If None, the current environment is used"))
parser.add_argument('-v', '--version', help='The version of the installer')
parser.add_argument('-i', '--input-dir', default='ci/constructor-files',
                    help=("The directory that contains the raw files for the "
                          "conda constructor. Default: %(default)s"))
parser.add_argument('-o', '--output-dir', default='gwgen-conda',
                    help=("The directory for the final constructor files. "
                          "Default: %(default)s"))
parser.add_argument('-f', '--environment-file', default='ci/environment.yml',
                    help=("The conda environment yaml file that contains the "
                          "specs to use for the constructor files. Default "
                          "%(default)s"))
parser.add_argument('--no-build', help="Do not build packages",
                    action='store_true')

args = parser.parse_args()

# get the version name
version = args.version or ''
if version.startswith('v'):
    version = version[1:]


def get_all_versions(name=None):
    list_cmd = 'conda list -e'.split()
    if name is not None:
        list_cmd += ['-n', name]
    all_versions_str = yaml.load(spr.check_output(list_cmd).decode('utf-8'))
    return {t[0]: t[1:] for t in map(lambda s: s.split('='),
                                     all_versions_str.split())}


all_versions = get_all_versions(args.name)
root_versions = get_all_versions('root')
all_versions['conda'] = root_versions['conda']


def get_version(mod, d=all_versions):
    try:
        return ' '.join(d[mod])
    except KeyError:
        return pkg_resources.get_distribution(mod).version


local_packages = list(chain.from_iterable(args.packages))

src_dir = args.input_dir

build_dir = args.output_dir

local_versions = OrderedDict([(pkg, get_version(pkg)) for pkg in
                              map(osp.basename, local_packages)])

with open(args.environment_file) as f:
    other_pkgs = [
        pkg.split('=')[0] for pkg in yaml.load(f)['dependencies']
        if isinstance(pkg, six.string_types)]

other_versions = {pkg: get_version(pkg) for pkg in other_pkgs}


# delete old psyplot-conda files
try:
    shutil.rmtree(build_dir)
except Exception as e:
    pass

# copy raw files
shutil.copytree(src_dir, build_dir)


# ----------------------- End User License Agreement --------------------------

with open(osp.join(build_dir, 'intro.rst')) as f:
    eula = f.read()

py_version = '.'.join(map(str, sys.version_info[:3]))

# replace versions
replacements = {}
replacements['VERSIONS'] = '\n'.join(
    '- %s: %s' % t for t in chain([('python', py_version)],
                                  sorted(local_versions.items()),
                                  sorted(other_versions.items())))

replacements['CONSTRUCTOR'] = get_version('constructor',
                                          get_all_versions('root'))

replacements['TIME'] = dt.datetime.now()

eula = eula.format(**replacements)

with open('LICENSE') as f:
    eula += '\n\n' + f.read()

with open(osp.join(build_dir, 'EULA.txt'), 'w') as f:
    f.write(eula)

# ---------------------------- construct.yaml ---------------------------------

with open(osp.join(build_dir, 'construct.yaml')) as f:
    construct = yaml.load(f)

if version:
    construct['version'] = version

# use all installed packages in the given environment
construct['specs'] = ['python %s*' % py_version] + [
    '%s %s' % (name, ' '.join(v)) for name, v in all_versions.items()
    if name != 'python']

if sys.platform.startswith('win'):
    post_file = 'post_win.bat'
elif sys.platform.startswith('darwin'):
    post_file = 'post_osx.sh'
else:
    post_file = 'post_linux.sh'
if osp.exists(osp.join(build_dir, post_file)):
    construct['post_install'] = post_file

# for packages in the psyplot framework, we use our own local builds
if not args.no_build and local_packages:
    spr.check_call(['conda', 'build', '--no-test'] + list(local_packages),
                   stdout=sys.stdout, stderr=sys.stderr)
if local_packages:

    builds = spr.check_output(
        ['conda', 'build', '--output'] + list(local_packages)).decode(
                'utf-8').splitlines()

    for fname in map(osp.basename, builds):
        construct['specs'].append(
            ' '.join(fname.replace('.tar.bz2', '').rsplit('-', 2)))

    conda_bld_dir = file2html(osp.dirname(osp.dirname(builds[0])))

    construct['channels'] = [conda_bld_dir] + construct['channels']


with open(osp.join(build_dir, 'construct.yaml'), 'w') as f:
    yaml.dump(construct, f, default_flow_style=False)
