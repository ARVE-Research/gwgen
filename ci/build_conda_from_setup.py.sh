#!/bin/bash

function pypi_skeleton() {

    DIR=$1
    ODIR=$2

    WORK=`pwd`

    cd $DIR
    FULL_DIR=`pwd`

    PKG=`basename ${FULL_DIR}`
    PYPKG=${PKG/-/_}

    VERSION=`python -c "import ${PYPKG}; print(${PYPKG}.__version__)"`
    # build the tarball
    python setup.py sdist
    FNAME=`ls -rt dist/* | tail -n 1`
    # create the receipt
    cd $WORK
    mkdir $ODIR
    echo "{% set version = '$VERSION' %}" > $ODIR/meta.yaml
    cat ci/${PKG}_meta.yaml >> $ODIR/meta.yaml

}


function build_conda() {
    DIR=$1

    # build
    conda build --python ${PYTHON_VERSION} $DIR --no-test
    # print the output
    BUILD_FNAME=$(conda build --python ${PYTHON_VERSION} $DIR --output)

}
