import collections

__VersionInfo = collections.namedtuple("VersionInfo", ("major", "minor", "micro"))

__version__ = "0.0.1"
__version_info__ = __VersionInfo(*(map(int, __version__.split("."))))

from t5_vae.model import *
from t5_vae.train import *
