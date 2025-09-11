"""
This file implements the heuristic baseline which computes the edge ratio of an image
as a proxy for complexity.
"""


import numpy as np
from skimage import io, color, feature
import matplotlib.pyplot as plt




def compute_edge_ratio(image):

    # --- Compute edges using Canny edge detector ---
    gray = color.rgb2gray(image)              # float in [0,1]
    edges = feature.canny(gray, sigma=2.0)  # tune sigma; also has low/high thresholds

    # --- Compute edge ratio ---
    num_edges = np.sum(edges)            # True counts as 1
    total_pixels = edges.size            # total number of pixels
    edge_ratio = num_edges / total_pixels
    return edge_ratio


