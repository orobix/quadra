import os

import dotenv
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class QuadraSearchPathPlugin(SearchPathPlugin):
    """Generic Search Path Plugin class."""

    def __init__(self):
        try:
            os.getcwd()
        except FileNotFoundError:
            # This may happen when running tests
            return

        if os.path.exists(os.path.join(os.getcwd(), ".env")):
            dotenv.load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Plugin used to add custom config to searchpath to be discovered by quadra."""
        # This can be global or taken from the .env
        quadra_search_path = os.environ.get("QUADRA_SEARCH_PATH", None)

        # Path should be specified as a list of hydra path separated by ";"
        # E.g pkg://package1.configs;file:///path/to/configs
        if quadra_search_path is not None:
            for i, path in enumerate(quadra_search_path.split(";")):
                search_path.append(provider=f"quadra-searchpath-plugin-{i}", path=path)
