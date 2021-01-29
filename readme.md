# cvtk
computer vision toolkit of pytorch

## packaging
```
# pip install --upgrade setuptools wheel
# pip install --upgrade twine
python setup.py sdist bdist_wheel
twine upload dist/*
pip install cvtk
```

git:
```
# https://pip.pypa.io/en/latest/reference/pip_install/#git
pip install git+https://github.com/flystarhe/cvtk.git@hash
pip install git+https://github.com/flystarhe/cvtk.git@main
pip install git+https://github.com/flystarhe/cvtk.git@v1.0
```
