from setuptools import setup
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


# We keep setup function for compatibility with older setuptools
# mentioned here https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
# Also, in the future we may move 'cmdclass' to pyproject.toml if supported.
setup(
    cmdclass={"install": GitVersionChecker},
)
