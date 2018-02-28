# Deployment procedures

Helpful instructions can be found [here](https://github.com/fhamborg/news-please/wiki/PyPI---How-to-upload-a-new-version)


1. Increment the version number in `setup.py`
2. Add notes for the new version in `RELEASE.txt`
3. Run the following commands to upload the new version to pypi

```
python setup.py sdist
python setup.py sdist upload
```

4. Check [pypi.python.org](pypi.python.org) that the new version is present.
