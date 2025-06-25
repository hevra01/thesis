import os
import sys

import numpy as np

from LID.utils import compute_knee

def estimate_LID_over_t_range(
    x,  # [data_dim] or [batch_size, data_dim]
    lid_estimator,
    t_values,
    hutchinson_sample_count,
    ambient_dim,
    device,
    return_info=False,
    **knee_kwargs
):
    """
    Estimates LID over a range of t values for a single instance or a batch.
    Returns the best LID using the knee algorithm.
    This function is intended to be used alone, when the lid of a single instance 
    or a batch is needed.
    """
    x = x.to(device)
    lid_curve = estimate_LID_over_t_range_batch(
            x=x,
            t_values=t_values,
            hutchinson_sample_count=hutchinson_sample_count,
            lid_estimator=lid_estimator
        )
    lid_curve = np.array(lid_curve)
    t_values = np.array(t_values)
    return lid_curve


def estimate_LID_over_t_range_batch_mean(x, t_values, hutchinson_sample_count, lid_estimator):
    """
    Performs LID estimation over a range of t values for a batch of data.
    This function is used by the estimate_LID_over_t_range_dataloader function. 
    """
    # this list will get filled with LID values for each t in t_values
    lid_curve = []

    # loop over each t value to estimate LID
    for t in t_values:
        lid_vals = lid_estimator._estimate_lid(x, t=t, hutchinson_sample_count=hutchinson_sample_count)

        # For a batch, we take the mean of LID values across the batch.
        # However, if the input is a single instance, we just take the value directly.
        lid_curve.append(lid_vals.mean().item())
        
    return lid_curve


def estimate_LID_over_t_range(x, t_values, hutchinson_sample_count, lid_estimator):
    """
    Performs LID estimation over a range of t values for a batch of data.
    """
    # this list will get filled with LID values for each t in t_values
    lid_over_t = {}

    # loop over each t value to estimate LID
    for t in t_values:
        lid_vals = lid_estimator._estimate_lid(x, t=t, hutchinson_sample_count=hutchinson_sample_count)

        lid_over_t[t] = lid_vals

    return lid_over_t

from typing import Dict, List

def compute_knees_for_all_data_points_in_batch(
    # Dictionary where keys are timesteps and values are lists of LID values for each data point.
    lid_over_t: Dict[float, List[float]],  
    ambient_dim: int,  # Original data dimension
    **knee_kwargs  # Additional arguments for the compute_knee function
):
    """
    Computes the knee for each data point using the LID curves derived from lid_over_t.

    Args:
        lid_over_t (dict): A dictionary where keys are timesteps and values are lists of LID values for each data point.
        ambient_dim (int): The ambient dimension.
        **knee_kwargs: Additional arguments to pass to the compute_knee function.

    Returns:
        List[dict]: A list of results from compute_knee for each data point.
    """
    # Extract timesteps and ensure they are sorted
    timesteps = sorted(lid_over_t.keys())
    # Convert the dictionary into a 2D array where each row corresponds to a data point's LID curve
    # Rows: Correspond to timesteps.
    # Columns: Correspond to data points.
    # To create LID curves for each data point, you need to transpose this structure. Transposing swaps rows and columns, so:
    # Rows: Correspond to data points.
    # Columns: Correspond to timesteps.
    lid_curves = np.array([lid_over_t[t].cpu().numpy() for t in timesteps]).T  

    results = []
    for lid_curve in lid_curves:
        # Call compute_knee for each LID curve
        result = compute_knee(
            timesteps=np.array(timesteps),
            lid_curve=lid_curve,
            ambient_dim=ambient_dim,
            **knee_kwargs
        )
        results.append(result)

    return results


def estimate_LID_over_t_range_dataloader(
    dataloader,
    lid_estimator,
    t_values,
    hutchinson_sample_count,
    ambient_dim,
    device,
    return_info=False,
    **knee_kwargs
):
    """
    Estimates LID over a range of t values for an entire dataset (dataloader).
    Processes the dataset batch-by-batch to avoid memory issues.
    Aggregates mean LID for each t across all batches, then finds the best LID using the knee algorithm.
    """
    # Initialize an array to accumulate the sum of mean LID values for each t
    lid_curve_sum = np.zeros(len(t_values), dtype=np.float64)
    # Counter for the number of batches processed
    lid_curve_count = 0

    # Outer loop: iterate over each batch in the dataloader
    # For each batch, we compute the mean LID for every t value.
    for batch in dataloader:
        images = batch["image"].to(device)

        # You accumulate these means across all batches.
        batch_lid_curve = []

        batch_lid_curve = estimate_LID_over_t_range_batch(
            x=images,
            t_values=t_values,
            hutchinson_sample_count=hutchinson_sample_count,
            lid_estimator=lid_estimator
        )

        # Add the batch's mean LID curve to the running sum
        # This operation adds the values element-wise (for each t).
        lid_curve_sum += np.array(batch_lid_curve)

        # Increment the batch counter
        lid_curve_count += 1

    # After all batches, compute the average LID curve over the dataset
    lid_curve = lid_curve_sum / lid_curve_count
    t_values = np.array(t_values)

    return lid_curve

    
