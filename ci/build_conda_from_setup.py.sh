#!/bin/bash

function pypi_skeleton() {

    DIR=$1
    ODIR=$2

    WORK=`pwd`

    cd $DIR
    # build the tarball
    python setup.py sdist
    FNAME=`ls -rt dist/* | tail -n 1`
    # create the receipt
    mkdir tmp
    RUNNING_SKELETON=1 conda skeleton pypi --output-dir $ODIR --python-version ${PYTHON_VERSION} \
        --manual-url file://`pwd`/${FNAME}

    cd $WORK

}


function build_conda() {
    DIR=$1

    # build
    conda build --python ${PYTHON_VERSION} $DIR --no-test
    # print the output
    BUILD_FNAME=$(conda build --python ${PYTHON_VERSION} $DIR --output)

}
