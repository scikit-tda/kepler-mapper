# Deployment procedures

To update a new version to pypi, simply create a github release. This will trigger the github-actions workflow to deploy a new package for you.

Make sure to increment the versions in `kmapper/_version.py` and add notes to
the `RELEASE.txt` file.

Thank you!

# Notes

We use semver for versioning as best as we know how. The current working development should be labeled with a `.dev` tag.
