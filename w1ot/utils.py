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

def normalize_and_log_transform(data: np.ndarray) -> np.ndarray:
    """
    Normalize and log-transform the count data.
    """
    if np.max(data) > 50:
        # normalize the data to make sure the sum of each row is 10000
        row_sums = np.sum(data, axis=1, keepdims=True) + 1e-5
        data = data / row_sums * 10000
        # log-transform the normalized data
        data = np.log1p(data)
    else:
        print("Data is not normalized and log-transformed")
    return data