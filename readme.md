# cvtk
computer vision toolkit of pytorch

## packaging
```
# python -m pip install --upgrade pip setuptools wheel
# python -m pip install --upgrade twine
python setup.py sdist bdist_wheel
python -m twine upload dist/*
python -m pip install cvtk
```
