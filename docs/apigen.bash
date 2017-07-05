#!/bin/bash
# script to automatically generate the psyplot api documentation using
# sphinx-apidoc and sed
sphinx-apidoc -f -M -e  -T -o api ../gwgen/
# replace chapter title in psyplot.rst
sed -i '' -e 1,1s/.*/'Python API Reference'/ api/gwgen.rst
sed -i '' -e 2,2s/.*/'===================='/ api/gwgen.rst
