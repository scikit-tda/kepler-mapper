from .kmapper import KeplerMapper
from .kmapper import cluster
from .cover import Cover
from .nerve import GraphNerve

# Enable access to version number
import pkg_resources

__version__ = pkg_resources.get_distribution("kmapper").version
