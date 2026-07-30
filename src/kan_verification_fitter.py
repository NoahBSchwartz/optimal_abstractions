# ============================================================================
# FITTING LINEAR SEGMENTS
# ============================================================================


import numpy as np
from src.custom_fastkan import FastKANLayer

def fit_line_through_points(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def find_bspline_segments_given_max_segments(layer, input_index, output_index, max_segments, min_x, max_x):
    """
    Used in both the vanilla fitting method and DP method.
    Given a single B-Spline in the network and a budget of segments,
    use DP to fit them optimally (minimizing the max error).
    """
    # Sample points from the spline (within defined x range)
    n_samples = 25
    x_points = np.linspace(min_x, max_x, n_samples)
    x_tensor, y_tensor = layer.plot_curve(input_index, output_index, num_pts=n_samples)
    x_np = x_tensor.detach().cpu().numpy()
    y_np = y_tensor.detach().cpu().numpy()
    mask = (x_np >= min_x) & (x_np <= max_x)
    x_points = x_np[mask]
    y_points = y_np[mask]
    n = len(x_points)
    if n < 2:
        return [], np.inf

    # Precompute errors for all possible segments
    errors = np.full((n, n), np.inf)
    for start_idx in range(n-1):
        max_len = min(n - start_idx, int(np.round((max_x - min_x) / ((max_x - min_x) / n))))
        x_start = x_points[start_idx]
        y_start = y_points[start_idx]
        for end_idx in range(start_idx + 1, start_idx + max_len):
            if end_idx >= n:
                break
            x_end = x_points[end_idx]
            y_end = y_points[end_idx]
            if abs(x_end - x_start) <= 1e-10:
                continue
            slope = (y_end - y_start) / (x_end - x_start)
            intercept = y_start - slope * x_start
            segment_x = x_points[start_idx:end_idx+1]
            segment_y = y_points[start_idx:end_idx+1]
            predicted_y = slope * segment_x + intercept
            segment_error = np.max(np.abs(predicted_y - segment_y))
            errors[start_idx, end_idx] = segment_error
    
    # Dynamic programming to find optimal segmentation
    dp = np.full((max_segments, n), np.inf)
    back = np.zeros((max_segments, n), dtype=int)
    for j in range(1, n):  # Base Case: one segment from start to j
        dp[0, j] = errors[0, j] 
    # TODO: add in recurrence relation
    for i in range(1, max_segments):
        for j in range(i+1, n):
            for k in range(i, j):
                if dp[i-1, k] == np.inf:
                    continue
                if errors[k, j] == np.inf:
                    continue
                curr_error = max(dp[i-1, k], errors[k, j])
                if curr_error < dp[i, j]:
                    dp[i, j] = curr_error
                    back[i, j] = k
    
    # Reconstruct the segments
    segments = []
    curr_seg = n - 1
    if dp[max_segments-1, curr_seg] == np.inf:
        return [], np.inf
    for i in range(max_segments-1, -1, -1):
        if i > 0:
            prev_seg = back[i, curr_seg]
        else:
            prev_seg = 0
        x1, y1 = x_points[prev_seg], y_points[prev_seg]
        x2, y2 = x_points[curr_seg], y_points[curr_seg]
        slope, intercept = fit_line_through_points(x1, y1, x2, y2)
        segments.insert(0, (x1, x2, slope, intercept))
        curr_seg = prev_seg
        if curr_seg <= 0:
            break
        
    return segments, dp[max_segments-1, n-1]

def fit_kan_vanilla(kan_model, segments_per_curve, min_x, max_x):
    """
    Our baseline fitting method for the KAN model.
    We assign the same number of segments to each curve for approximation.
    Then, use Gurobi to solve for output bounds and track time to get a target to beat.
    """
    all_segments = []
    for layer_idx, layer in enumerate(kan_model.layers):
        layer.cpu()
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Fitting Layer {layer_idx} ({input_dim} -> {output_dim})")
        for input_index in range(input_dim):
            for output_index in range(output_dim):
                segments, max_error = find_bspline_segments_given_max_segments(layer, input_index, output_index, segments_per_curve, min_x, max_x)
                all_segments.append({
                    'layer_idx': layer_idx,
                    'input_idx': input_index,
                    'output_idx': output_index,
                    'segments': segments,
                    'max_error': max_error
                })
    print(f"Generated data for {len(all_segments)} connections.")
    return all_segments

def calculate_bspline_lipschitz_constant(layer: 'FastKANLayer', input_idx: int, output_idx: int, num_pts: int = 1000, min_x: float = None, max_x: float = None) -> float:
    if min_x is None:
        min_x = layer.rbf.grid_min
    if max_x is None:
        max_x = layer.rbf.grid_max
    x_tensor, y_tensor = layer.plot_curve(input_idx, output_idx, num_pts=num_pts)
    x_np = x_tensor.detach().cpu().numpy()
    y_np = y_tensor.detach().cpu().numpy()
    mask = (x_np >= min_x) & (x_np <= max_x)
    x_np = x_np[mask]
    y_np = y_np[mask]

    dx = np.diff(x_np)
    dy = np.diff(y_np)
    nonzero_dx = dx != 0
    slopes = np.zeros_like(dx)
    slopes[nonzero_dx] = dy[nonzero_dx] / dx[nonzero_dx]
    lipschitz_constant = np.max(np.abs(slopes))
    return lipschitz_constant

def compute_dp_tables_lipschitz(kan_model, max_segments=15, min_x=-5.0, max_x=5.0):
    """
    Compute the error tables, segments tables, and Lipschitz constants for all B-splines in the KAN model.
    This is used for the DP method later.
    """
    error_tables = {}
    segments_tables = {}
    lipschitz_constants = {}
    
    # Iterate through all layers
    for layer_idx, layer in enumerate(kan_model.layers):
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Analyzing Layer {layer_idx}: {input_dim} inputs -> {output_dim} outputs")
        # Iterate through all B-splines in this layer
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                layer = kan_model.layers[layer_idx]
                error_table = {}
                segments_table = {}
                for num_segments in range(1, max_segments + 1):
                    segments, error = find_bspline_segments_given_max_segments(layer, input_idx, output_idx, num_segments, min_x, max_x)
                    error_table[num_segments] = error
                    segments_table[num_segments] = segments
                lipschitz_constant = calculate_bspline_lipschitz_constant(layer=layer, input_idx=input_idx, output_idx=output_idx)
                error_tables[spline_key] = error_table
                segments_tables[spline_key] = segments_table
                lipschitz_constants[spline_key] = lipschitz_constant
    
    return error_tables, segments_tables, lipschitz_constants

def get_all_downstream_splines(kan_model, layer_idx, output_idx, visited=None):
    """
    Find all downstream splines in the KAN model starting from a given spline (at position layer_idx, output_idx).
    This is used to compute the Lipschitz constant for a certain spline.
    """
    if visited is None:
        visited = set()
    # Base Case: last layer has no downstream splines
    if layer_idx >= len(kan_model.layers) - 1:
        return set()
    # The output of current spline becomes an input in the next layer
    next_input_idx = output_idx
    next_layer_idx = layer_idx + 1
    next_layer = kan_model.layers[next_layer_idx]
    downstream_splines = set()
    for next_output_idx in range(next_layer.output_dim):
        next_spline = (next_layer_idx, next_input_idx, next_output_idx)
        if next_spline not in visited:
            visited.add(next_spline)
            downstream_splines.add(next_spline)
            further_downstream = get_all_downstream_splines(kan_model, next_layer_idx, next_output_idx, visited)
            downstream_splines.update(further_downstream)
    return downstream_splines

def weight_dp_tables_lipschitz(kan_model, error_tables, segments_tables, lipschitz_constants):
    """
    Weight the error tables for every B-Spline by the product of Lipschitz constants of all downstream splines.
    (ie. weight the error tables by their overall effect on the network output bound)
    """
    weighted_error_tables = {}
    # Iterate through all splines in the network
    for layer_idx, layer in enumerate(kan_model.layers):
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(f"Weighing Layer {layer_idx}: {input_dim} inputs -> {output_dim} outputs")
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                downstream_splines = get_all_downstream_splines(kan_model, layer_idx, output_idx)
                # Weight the error table by computing the product of Lipschitz constants of all downstream splines
                lipschitz_product = 1.0
                for downstream_spline in downstream_splines:
                    if downstream_spline in lipschitz_constants:
                        lipschitz_product *= lipschitz_constants[downstream_spline]
                weighted_table = {}
                for num_segments, error in error_tables[spline_key].items():
                    weighted_table[num_segments] = error * lipschitz_product
                weighted_error_tables[spline_key] = weighted_table
    return weighted_error_tables

def fit_kan_optimally_lipschitz(weighted_error_tables, segments_tables, max_error, max_segments=50):
    """
    Use DP to find the optimal allocation of segments to splines in the KAN model.
    (Use the weighted error tables computed above).
    """
    used_backup_allocation = False
    all_splines = list(weighted_error_tables.keys())
    num_splines = len(all_splines)
    max_total_segments = num_splines * max_segments
    
    # Initialization: dp[i][j] = minimum error achievable when allocating j total segments to the first i splines
    dp = np.full((num_splines + 1, max_total_segments + 1), float('inf'))
    dp[0, 0] = 0  # Base case: 0 error when allocating 0 segments to 0 splines
    backtrack = np.zeros((num_splines + 1, max_total_segments + 1), dtype=int)
    for i in range(1, num_splines + 1):
        spline_key = all_splines[i-1]
        for j in range(max_total_segments + 1):
            # Try different allocations for the current spline (updating allocation if we find a better error)
            for segs in range(min(j + 1, max_segments + 1)):
                if segs not in weighted_error_tables[spline_key]:
                    continue
                spline_error = weighted_error_tables[spline_key][segs]
                prev_error = dp[i-1, j-segs]
                if prev_error != float('inf') and max(prev_error, spline_error) < dp[i, j]:
                    dp[i, j] = max(prev_error, spline_error)
                    backtrack[i, j] = segs
    
    # Backtrack: Find the min total segments to achieve error <= max_error
    optimal_total_segments = None
    achieved_error = float('inf')
    for j in range(max_total_segments + 1):
        if dp[num_splines, j] <= max_error:
            optimal_total_segments = j
            achieved_error = dp[num_splines, j]
            break
    if optimal_total_segments is None:
        min_error_idx = np.argmin(dp[num_splines, :])
        optimal_total_segments = min_error_idx
        achieved_error = dp[num_splines, min_error_idx]
        print(f"Could not find allocation satisfying the max error constraint ({max_error}).")
        print(f"Returning best allocation with error: {achieved_error}")
        used_backup_allocation = True
    optimal_allocation = {}
    actual_segments = {}
    remaining_segments = optimal_total_segments
    for i in range(num_splines, 0, -1):
        spline_key = all_splines[i-1]
        # Get the segment allocation for this spline and store the coordinates of its linear segments
        segs_for_spline = int(backtrack[i, remaining_segments])
        optimal_allocation[spline_key] = segs_for_spline
        actual_segments[spline_key] = segments_tables[spline_key].get(segs_for_spline, [])
        remaining_segments -= segs_for_spline
    total_segments = sum(optimal_allocation.values())
    print(f"  Achieved error: {achieved_error}")
    print(f"  Total segments allocated: {total_segments}")

    return optimal_allocation, actual_segments, total_segments, achieved_error, dp, used_backup_allocation


def fit_kan_optimally_lipschitz_memory_efficient(weighted_error_tables, segments_tables, max_error, max_segments=50):

    
    """
    Memory-efficient version of fit_kan_optimally_lipschitz using space-optimized DP.
    """
    problem_flag = False
    all_splines = list(weighted_error_tables.keys())
    num_splines = len(all_splines)

    if num_splines == 0:
        print("Warning: No splines found to optimize.")
        return {}, {}, 0, 0.0, False

    print(f"Optimizing segment allocation for {num_splines} B-splines...")


    estimated_total = num_splines * 5
    cap = 200000
    max_total_segments = min(estimated_total, cap)
    print(f"  Max_total_segments not provided. Using heuristic default: {max_total_segments}")

    print(f"  Max segments per spline: {max_segments}")

    prev_row = np.full(max_total_segments + 1, np.inf)
    prev_row[0] = 0.0
    backtrack_decisions = {}
    for i in range(1, num_splines + 1):
        spline_key = all_splines[i - 1]
        spline_weighted_errors = weighted_error_tables.get(spline_key, {})
        curr_row = np.full(max_total_segments + 1, np.inf)
        curr_backtrack = np.zeros(max_total_segments + 1, dtype=np.int16)

        if not spline_weighted_errors:
             curr_row = prev_row.copy()
             print(f"Warning: Spline {spline_key} has no weighted error entries. Allocating 0 segments.")

        else:
            for j in range(max_total_segments + 1):
                 min_achieved_error_for_j = np.inf
                 k = 0
                 error_k0 = spline_weighted_errors.get(0, np.inf)
                 prev_error_k0 = prev_row[j]
                 if prev_error_k0 != np.inf:
                      current_max_error = max(prev_error_k0, error_k0)
                      if current_max_error < min_achieved_error_for_j:
                           min_achieved_error_for_j = current_max_error
                           curr_backtrack[j] = 0
                 max_k_for_spline = min(j, max_segments)
                 for k in range(1, max_k_for_spline + 1):
                      spline_error_k = spline_weighted_errors.get(k, None)
                      if spline_error_k is None:
                          continue
                      prev_error = prev_row[j - k]
                      if prev_error != np.inf:
                          current_max_error = max(prev_error, spline_error_k)
                          if current_max_error < min_achieved_error_for_j:
                              min_achieved_error_for_j = current_max_error
                              curr_backtrack[j] = k
                 curr_row[j] = min_achieved_error_for_j

        backtrack_decisions[i] = curr_backtrack.copy()
        prev_row = curr_row
        if i % 100 == 0 or i == num_splines:
            print(f"  DP progress: Processed spline {i}/{num_splines}")

    final_dp_row = prev_row
    optimal_total_segments = -1
    achieved_error = np.inf
    for j in range(max_total_segments + 1):
        if final_dp_row[j] <= max_error:
            optimal_total_segments = j
            achieved_error = final_dp_row[j]
            break

    if optimal_total_segments == -1:
        problem_flag = True
        min_error_idx = np.argmin(final_dp_row)
        if np.isinf(final_dp_row[min_error_idx]):
             print(f"ERROR: Could not find *any* valid segment allocation within {max_total_segments} total segments.")
             return {}, {}, 0, np.inf, True
        else:
             optimal_total_segments = min_error_idx
             achieved_error = final_dp_row[optimal_total_segments]
             print(f"WARNING: Could not meet max_error constraint ({max_error:.4f}).")
             print(f"         Returning best possible allocation with {optimal_total_segments} segments and error {achieved_error:.4f}")

    optimal_allocation = {}
    actual_segments = {}
    remaining_segments = optimal_total_segments
    print(f"\nReconstructing allocation for {optimal_total_segments} total segments...")
    for i in range(num_splines, 0, -1):
        spline_key = all_splines[i - 1]
        if remaining_segments < 0:
             print(f"ERROR: Backtracking resulted in negative remaining segments at spline index {i}. Aborting reconstruction.")
             return optimal_allocation, actual_segments, -1, achieved_error, True
        try:
             segs_for_spline = int(backtrack_decisions[i][remaining_segments])
        except IndexError:
             print(f"ERROR: IndexError during backtracking. i={i}, remaining_segments={remaining_segments}. Max total segments might be too low or DP issue.")
             return optimal_allocation, actual_segments, -1, achieved_error, True
        except KeyError:
              print(f"ERROR: KeyError during backtracking. Spline index {i} not found in backtrack_decisions. DP issue.")
              return optimal_allocation, actual_segments, -1, achieved_error, True


        optimal_allocation[spline_key] = segs_for_spline
        spline_segs_table = segments_tables.get(spline_key, {})
        actual_segments[spline_key] = spline_segs_table.get(segs_for_spline, [])
        remaining_segments -= segs_for_spline


    if remaining_segments != 0:
        print(f"Warning: Backtracking finished with {remaining_segments} remaining segments != 0. Check DP logic.")


    total_segments_used = sum(optimal_allocation.values())

    print(f"\nOptimization Complete:")
    print(f"  Target Max Weighted Error: {max_error:.4f}")
    print(f"  Achieved Max Weighted Error: {achieved_error:.4f}")
    print(f"  Total Segments Used: {total_segments_used} (Constraint: {max_total_segments})")
    if problem_flag:
        print(f"  NOTE: Target error was not met.")

    return optimal_allocation, actual_segments, total_segments_used, achieved_error, None, problem_flag
