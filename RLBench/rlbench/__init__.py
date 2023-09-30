__version__ = '1.1.0'

import numpy as np
import pyrep

pr_v = np.array(pyrep.__version__.split('.'), dtype=int)
if pr_v.size < 4 or np.any(pr_v < np.array([4, 1, 0, 2])):
    raise ImportError(
        'PyRep version must be greater than 4.1.0.2. Please update PyRep.')


from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.observation_config import CameraConfig
from rlbench.sim2real.domain_randomization import RandomizeEvery
from rlbench.sim2real.domain_randomization import VisualRandomizationConfig
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
#from rlbench.robosuite_interface import Robosuite_Interface
