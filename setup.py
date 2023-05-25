from pathlib import Path
from typing import List

from setuptools import find_namespace_packages, setup
from setuptools._distutils.dist import Distribution
from setuptools.command.install import install


class GitVersionChecker(install):
    """Class used to check git version before installation."""

    def __init__(self, dist: Distribution) -> None:
        super().__init__(dist)
        self._check_git_version()

    def _check_git_version(self) -> None:
        """Call git --version and check that the version is 2.10 or higher."""
        # pylint: disable=import-outside-toplevel
        import subprocess

        try:
            git_version_bytes = subprocess.check_output(["git", "--version"])
            git_version = git_version_bytes.decode("utf-8").strip()
            if git_version.startswith("git version"):
                git_version = git_version.split(" ")[2]
            else:
                raise RuntimeError("Unable to get git version")
        except Exception as e:
            raise RuntimeError("Unable to get git version") from e

        if git_version < "2.10":
            raise RuntimeError("Git version 2.10 or higher is required for installation")

    def run(self) -> None:
        """Run the installation."""
        install.run(self)


BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
dependency_links: List[str] = []
required_packages = []
with open(Path(BASE_DIR, "requirements.txt")) as file:
    for line in file.readlines():
        line = line.strip()
        required_packages.append(line)

with open(Path(BASE_DIR, "quadra", "__init__.py")) as file:
    for line in file:
        if line.startswith("__version__"):
            __version__ = line.split("=")[1].strip().strip("\"'")
            break


test_packages = [
    "pytest==7.2.*",
    "pytest-cov==4.0.*",
]

dev_packages = [
    "interrogate==1.5.*",
    "black==22.12.*",
    "isort==5.11.*",
    "pre-commit==3.0.*",
    "pylint==2.16.*",
    "bump2version==1.0.*",
    "types-PyYAML==6.0.12.*",
    "mypy==1.0.*",
    "ruff==0.0.257",
    "pandas-stubs==1.5.3.*",
]

docs_packages = [
    "mkdocs==1.4.*",
    "mkdocs-material==9.0.*",
    "mkdocstrings-python==0.8.*",
    "mkdocs-gen-files==0.4.*",
    "mkdocs-literate-nav==0.6.*",
    "mkdocs-section-index==0.3.*",
    "mike==1.1.*",
]

setup(
    name="quadra",
    version=str(__version__),
    license="MIT",
    description="Deep Learning experiment orchestration Library",
    url='https://github.com/orobix/quadra',
    author="Alessandro Polidori, Federico Belotti, Lorenzo Mammana, Refik Can Malli, Silvia Bianchetti",
    author_email="refikcan.malli@orobix.com",
    python_requires=">=3.8,<3.10",
    entry_points={"console_scripts": ["quadra = quadra.main:main"]},
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    dependency_links=dependency_links,
        
    include_package_data=True,
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
    cmdclass={"install": GitVersionChecker},
)
