# After updates to the repository, run these commands to deploy Persimmon to Pypi

pip install twine
python setup.py sdist
pip install wheel
python setup.py bdist_wheel

# twine upload dist/*