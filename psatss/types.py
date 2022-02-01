"""This module contains some datatypes that are used in the application
"""

from mpyc.runtime import mpc
import numpy as np

sectypes = {
            "int_64": mpc.SecInt(64), 
            "int_32": mpc.SecInt(32), 
            "fxp_64": mpc.SecFxp(64)
            }

num_types = (int, float, np.int32, np.int64, np.float64)