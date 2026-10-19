"""Root of the destacking data tree.

All scripts read and write under DATA_ROOT, which is the environment variable
DESTACKING_DATA if set, else ~/destacking_data:

    DATA_ROOT/real_data   real cluster and shear catalogues and their products
    DATA_ROOT/sims        simulated cluster catalogues and background arrays
    DATA_ROOT/output      simulation and fit runs
"""
import os

DATA_ROOT = os.path.abspath(os.path.expanduser(os.environ.get("DESTACKING_DATA", "~/destacking_data")))


def expand_path(value):
    """Expand ~, $DESTACKING_DATA (defaulting to DATA_ROOT) and other environment variables in a config path.

    Non-string values are returned unchanged.
    """
    if not isinstance(value, str):
        return value
    value = value.replace("${DESTACKING_DATA}", DATA_ROOT).replace("$DESTACKING_DATA", DATA_ROOT)
    return os.path.expandvars(os.path.expanduser(value))
