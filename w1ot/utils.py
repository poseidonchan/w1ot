import numpy as np

def ensure_numpy(data) -> np.ndarray:
    """
    Ensure the data is a NumPy ndarray.

    Parameters:
        data (Any): The data to be converted.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'toarray'):
        return data.toarray()
    else:
        return np.array(data)
