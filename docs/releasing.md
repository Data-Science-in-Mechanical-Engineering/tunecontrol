# Publishing TuneControl to PyPI

The repository builds a source distribution and a wheel using the root
`pyproject.toml`. The supported interpreter is currently Python 3.12.

## One-time setup

1. Sign in to [PyPI](https://pypi.org/) and open
   [Publishing](https://pypi.org/manage/account/publishing/). Register a pending
   GitHub publisher for a new project (or add a trusted publisher under the
   existing project's Publishing settings if you already own it):

   | Field | Value |
   | --- | --- |
   | PyPI project name | `tunecontrol` |
   | GitHub owner | `Data-Science-in-Mechanical-Engineering` |
   | Repository | `tunecontrol` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

   A pending publisher does not reserve a package name. If another owner has
   already registered it, resolve ownership or choose a different distribution
   name before release.
2. Create a GitHub environment named `pypi` in repository settings. Restrict
   deployment to release tags (`v*`) and configure any desired required reviewers.
3. Push the packaging changes and workflow to GitHub. No PyPI API token is needed;
   publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## Validate a release locally

From the repository root, in a Python 3.12 virtual environment:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
python -m venv /tmp/tunecontrol-release-test
/tmp/tunecontrol-release-test/bin/python -m pip install dist/tunecontrol-0.1.0-py3-none-any.whl
/tmp/tunecontrol-release-test/bin/python -I -m unittest discover -s "$PWD/tests" -v
```

Use a fresh output directory (or remove old build artifacts) for each release and
substitute the new wheel filename when the version changes. The default build
creates the wheel from the source distribution, checking both release formats.
The tests exercise all 34 registered tasks and deterministic repeatability.

## Publish

1. Update `project.version` in `pyproject.toml` and `version` in `CITATION.cff`
   together. The first release is `0.1.0`.
2. Merge the changes and confirm that the build/test workflow passes.
3. Create and publish a GitHub release tagged `v0.1.0` at the reviewed commit.
   Publishing the release triggers the PyPI upload. A tag push alone does not.
4. Confirm the workflow succeeds and verify installation in a fresh Python 3.12
   environment with `python -m pip install tunecontrol`.
5. Make the PyPI command the primary README install instruction once the release
   is available.

Pull requests, main-branch pushes, and manual workflow runs build and test only.
The release job rejects tags that do not match the package version. PyPI versions
cannot be overwritten; use a new version for a subsequent release.
