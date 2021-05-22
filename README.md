# cvtk
computer vision toolkit of pytorch

* PyTorch 1.7+

## packaging
```sh
# pip install --upgrade setuptools wheel
# pip install --upgrade twine
rm -rf build dist *.egg-info
python setup.py sdist bdist_wheel
twine upload dist/*
```

## installation
```sh
pip install cvtk
# https://pip.pypa.io/en/latest/reference/pip_install/#git
pip install git+https://github.com/flystarhe/cvtk.git@hash
pip install git+https://github.com/flystarhe/cvtk.git@main
pip install git+https://github.com/flystarhe/cvtk.git@v1.0
```
