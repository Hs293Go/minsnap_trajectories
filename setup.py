"""Legacy setup.py, kept so this package installs where PEP 517 cannot reach.

pyproject.toml stays the source of truth: The build backend builds the published wheels
, and no PEP 517 frontend ever reads this file. It exists for build tooling that invokes
setup.py directly, e.g. colcon.

Metadata is parsed out of pyproject.toml using built-in regex to avoid manually
repeating metadata.
"""

import pathlib
import re

from setuptools import find_packages, setup

# resolve() before .parent, because this file is not always executed where it
# lives: colcon's --symlink-install symlinks setup.py into the build directory
# and runs it from there, and .parent would be that build directory, which has
# no pyproject.toml. Resolving the symlink lands back in the source tree.
_SOURCE_DIR = pathlib.Path(__file__).resolve().parent
_PYPROJECT_PATH = _SOURCE_DIR / "pyproject.toml"
if not _PYPROJECT_PATH.is_file():
    raise RuntimeError(
        f"{_PYPROJECT_PATH} does not exist. setup.py parses its metadata from pyproject.toml "
        "and expects to find it beside the resolved location of this file."
    )
_PYPROJECT = _PYPROJECT_PATH.read_text(encoding="utf-8")


def _field(pattern: str) -> str:
    """Pull one field out of pyproject.toml, failing loudly if it moved."""
    match = re.search(pattern, _PYPROJECT, re.MULTILINE | re.DOTALL)
    if match is None:
        raise RuntimeError(
            f"pyproject.toml has no field matching {pattern!r}. setup.py parses its metadata "
            "from there; update the pattern to match the new layout."
        )
    return match.group(1)


setup(
    name=_field(r'^name = "([^"]+)"'),
    version=_field(r'^version = "([^"]+)"'),
    python_requires=_field(r'^requires-python = "([^"]+)"'),
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=re.findall(r'"([^"]+)"', _field(r"^dependencies = \[(.*?)\]")),
)
