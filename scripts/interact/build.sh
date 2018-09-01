rm -r build
rm *.so
python setup.py build_ext --inplace
