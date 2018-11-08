# Deployment procedures

Follow these steps when deploying a new version to Pypi

1. Remove `.dev` tag from version number in `setup.py`
2. Add release notes for the new version in `RELEASE.txt`
3. Run the following commands to upload the new version to pypi

```
pip install -U twine
python setup.py sdist
pip install wheel
python setup.py bdist_wheel
```

```
twine upload dist/*
```

4. Check [pypi.python.org](pypi.python.org) that the new version is present.
5. Increment version number and give `.dev` tag. 


# Notes

We use semver for versioning as best as we know how. The current working development should be labeled with a `.dev` tag.


Helpful instructions can be found [here](https://github.com/fhamborg/news-please/wiki/PyPI---How-to-upload-a-new-version)
