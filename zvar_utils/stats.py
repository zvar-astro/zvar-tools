import numpy as np
from scipy.stats import median_abs_deviation # noqa: F401

def calculate_cdf(data: np.ndarray) -> np.ndarray:
    # Filter data to include only values greater than 0
    filtered_data = data[data > 0]

    # Create a distribution of the filtered data
    hist, bin_edges = np.histogram(filtered_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the CDF of the distribution
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    return bin_centers, cdf


def find_extrema(data: np.ndarray, cdf_value: float = 0.999) -> np.ndarray:
    bin_centers, cdf = calculate_cdf(data)

    threshold_index = np.searchsorted(cdf, cdf_value)
    threshold = bin_centers[threshold_index]

    candidate_indices = data > threshold
    return np.array(candidate_indices)