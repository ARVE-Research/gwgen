# Replace the necessary parts in the files for conda constructor
#
# This script replaces some of the parts in the files in ci/gwgen_conda,
# mainly versions but also builds for psyplot and gwgen
#
# Necessary environment variables are
#
# PSYPLOT_FILE
#     The path to the created psyplot build
# GWGEN_FILE
#     The path to the created psyplot build
# GWGEN_VERSION
#     The version of gwgen
# PYTHON_VERSION
#     The version of python
# TRAVIS_OS_NAME
#     The current OS name

set -e

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    BASH_RC=.bash_profile
else
    BASH_RC=.bashrc
fi

sed -i "s/PYTHON_VERSION/${PYTHON_VERSION}*/; s/GWGEN_VERSION/${GWGEN_VERSION}/; s#GWGEN_FILE#${GWGEN_FILE}#; s#PSYPLOT_FILE#${PSYPLOT_FILE}#" ci/gwgen_conda/construct.yaml
sed -i "s#PSY_SIMPLE_FILE#${PSY_SIMPLE_FILE}#; s#PSY_REG_FILE#${PSY_REG_FILE}#" ci/gwgen_conda/construct.yaml
sed -i "s/<<<BASH_RC>>>/${BASH_RC}/" ci/gwgen_conda/post.sh
sed -i "s#CREATION TIME#`date`#; s/PYTHON_VERSION/${PYTHON_VERSION}/; s/GWGEN_VERSION/${GWGEN_VERSION}/; s/CONSTRUCTOR VERSION/`constructor -V`/" ci/gwgen_conda/EULA.txt
