import numpy as np

def init_random_state(random_state):
    return (random_state
        if isinstance(random_state, np.random.RandomState) else
        np.random.RandomState(random_state))

def check_attribute(module, attr_name):
    if not hasattr(module, attr_name):
        raise NotImplementedError(f"Attribute '{attr_name}' not implemented in module '{module.__name__}'")
