# Replace the necessary parts in the files for conda constructor
#
# This script replaces some of the parts in the files in ci/gwgen_conda,
# mainly versions but also the build for gwgen
#
# Necessary environment variables are
#
# GWGEN_FILE
#     The path to the created gwgen build
# GWGEN_VERSION
#     The version of gwgen
# PYTHON_VERSION
#     The version of python
# TRAVIS_OS_NAME
#     The current OS name

set -e

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    BASH_RC=.bash_profile
    INPLACE='-i ""'
else
    BASH_RC=.bashrc
    INPLACE='-i'
fi

sed $INPLACE "s/PYTHON_VERSION/${PYTHON_VERSION}*/; s/GWGEN_VERSION/${GWGEN_VERSION}/; s#GWGEN_FILE#${GWGEN_FILE}#" ci/gwgen_conda/construct.yaml
sed $INPLACE "s/<<<BASH_RC>>>/${BASH_RC}/" ci/gwgen_conda/post.sh
sed $INPLACE "s#CREATION TIME#`date`#; s/PYTHON_VERSION/${PYTHON_VERSION}/; s/GWGEN_VERSION/${GWGEN_VERSION}/; s/CONSTRUCTOR VERSION/`constructor -V`/" ci/gwgen_conda/EULA.txt
