# ============================================================================
# MILP VERIFICATION
# ============================================================================

from src.kan_verification_grapher import *
from src.kan_verification_trainer import *
from src.kan_verification_fitter import *
from src.kan_verification_experiment_runner import *
import numpy as np
from pulp import (
    GUROBI_CMD, LpBinary, LpMaximize, LpMinimize, LpProblem, 
    LpStatus, LpVariable, lpSum
)
from typing import List, Tuple, Optional, Union

class SplineApproximation:
    """
    Class to represent a spline approximation for a single connection in the KAN model.
    """
    def __init__(self, segments: List[Tuple[float, float, float, float]], max_error: float):
        self.segments = sorted(segments, key=lambda s: s[0]) if segments else []
        self.original_max_error = max_error
        self.max_error_mip = max_error if np.isfinite(max_error) else 1e7 # For MIP solver stability

    def is_valid(self):
        return bool(self.segments) and np.isfinite(self.original_max_error)


def prep_segments_data(kan_segments_data: List[dict], input_dim: int, hidden_dim: int, output_dim: int):
    """
    Seperate the segments data by layer to make verification easier.
    """
    layer0_approximations = [[None for _ in range(hidden_dim)] for _ in range(input_dim)]
    layer1_approximations = [[None for _ in range(output_dim)] for _ in range(hidden_dim)]
    processed_indices_l0 = set()
    processed_indices_l1 = set()
    segments_layer0 = [s for s in kan_segments_data if s['layer_idx'] == 0]
    segments_layer1 = [s for s in kan_segments_data if s['layer_idx'] == 1]

    for seg_data in segments_layer0:
        i = seg_data['input_idx']
        h = seg_data['output_idx']
        approx = SplineApproximation(seg_data['segments'], seg_data['max_error'])
        layer0_approximations[i][h] = approx
        processed_indices_l0.add((i, h))

    for seg_data in segments_layer1:
        h = seg_data['input_idx']
        o = seg_data['output_idx']
        approx = SplineApproximation(seg_data['segments'], seg_data['max_error'])
        layer1_approximations[h][o] = approx
        processed_indices_l1.add((h, o))

    return layer0_approximations, layer1_approximations

def MIP_interval_analysis(
    layer0_approximations: List[List[Optional[SplineApproximation]]],
    layer1_approximations: List[List[Optional[SplineApproximation]]],
    x_min_vec: Union[List[float], np.ndarray],
    x_max_vec: Union[List[float], np.ndarray],
    M: float = 1e4,
):
    """
    Set up the MIP problem for the KAN model using the provided layer approximations and input bounds.
    Returns the results of the optimization for each output dimension.
    """
    if (not layer0_approximations or not layer0_approximations[0]) or (not layer1_approximations or not layer1_approximations[0]):
        print("Error: layer_approximations is empty or invalid.")
        return []
    input_dim = len(layer0_approximations)
    hidden_dim = len(layer0_approximations[0])
    output_dim = len(layer1_approximations[0])
    if len(x_min_vec) != input_dim or len(x_max_vec) != input_dim:
        raise ValueError(f"Input bounds vector length mismatch. Expected {input_dim}, got min:{len(x_min_vec)}, max:{len(x_max_vec)}")
    print(f"Setting up MIP for KAN [{input_dim}, {hidden_dim}, {output_dim}]...")
    print(f"Input bounds: {list(zip(x_min_vec, x_max_vec))}")


    # Define Variables
    x_vars = [LpVariable(f"x_{i}", lowBound=x_min_vec[i], upBound=x_max_vec[i]) for i in range(input_dim)]
    layer0_spline_outputs = [[LpVariable(f"l0_spline_{i}_{h}") for h in range(hidden_dim)] for i in range(input_dim)]
    hidden_activations = [LpVariable(f"hidden_act_{h}") for h in range(hidden_dim)]
    layer1_spline_outputs = [[LpVariable(f"l1_spline_{h}_{o}") for o in range(output_dim)] for h in range(hidden_dim)]
    output_vars = [LpVariable(f"output_{o}") for o in range(output_dim)] # Bounds are the goal


    # Binary indicators for segment selection (layer 0 defined first, then layer 1)
    layer0_indicators = []
    for i in range(input_dim):
        indicators_i = []
        for h in range(hidden_dim):
            approx = layer0_approximations[i][h]
            if approx and approx.segments:
                indicators_i.append([LpVariable(f"l0_ind_{i}_{h}_seg_{j}", cat=LpBinary) for j in range(len(approx.segments))])
            else:
                indicators_i.append([]) # No indicators if no segments
        layer0_indicators.append(indicators_i)
    layer1_indicators = []
    for h in range(hidden_dim):
        indicators_h = []
        for o in range(output_dim):
            approx = layer1_approximations[h][o]
            if approx and approx.segments:
                indicators_h.append([LpVariable(f"l1_ind_{h}_{o}_seg_{j}", cat=LpBinary) for j in range(len(approx.segments))])
            else:
                indicators_h.append([])
        layer1_indicators.append(indicators_h)


    # Define Constraints
    base_constraints = []
    # Layer 0 (Input -> Hidden)
    for i in range(input_dim):
        for h in range(hidden_dim):
            approx = layer0_approximations[i][h]
            x_input = x_vars[i] # Input for this spline connection
            spline_output_var = layer0_spline_outputs[i][h]
            indicators = layer0_indicators[i][h]
            if indicators: # 1. One segment must be active
                base_constraints.append(lpSum(indicators) == 1)
            for j, (start_x, end_x, slope, intercept) in enumerate(approx.segments): # 2. Constrain x bounds and output value
                indicator = indicators[j]
                y_approx = slope * x_input + intercept
                base_constraints.append(x_input >= start_x - M * (1 - indicator))
                base_constraints.append(x_input <= end_x + M * (1 - indicator))
                error = approx.max_error_mip
                base_constraints.append(spline_output_var <= y_approx + error + M * (1 - indicator))
                base_constraints.append(spline_output_var >= y_approx - error - M * (1 - indicator))
    # Hidden Layer
    for h in range(hidden_dim):
        incoming_splines = [layer0_spline_outputs[i][h] for i in range(input_dim)]
        base_constraints.append(hidden_activations[h] == lpSum(incoming_splines)) # Value of activation h in hidden layer = sum over incoming splines
    # Layer 1 (Hidden -> Output)
    for h in range(hidden_dim):
        for o in range(output_dim):
            approx = layer1_approximations[h][o]
            hidden_input = hidden_activations[h] # Input for this spline connection
            spline_output_var = layer1_spline_outputs[h][o]
            indicators = layer1_indicators[h][o]
            if indicators: # 1. One segment must be active
                 base_constraints.append(lpSum(indicators) == 1)
            for j, (start_x, end_x, slope, intercept) in enumerate(approx.segments): # 2. Constrain x bounds and output value
                indicator = indicators[j]
                y_approx = slope * hidden_input + intercept
                base_constraints.append(hidden_input >= start_x - M * (1 - indicator))
                base_constraints.append(hidden_input <= end_x + M * (1 - indicator))
                error = approx.max_error_mip
                base_constraints.append(spline_output_var <= y_approx + error + M * (1 - indicator))
                base_constraints.append(spline_output_var >= y_approx - error - M * (1 - indicator))
    # Final Output
    for o in range(output_dim):
        incoming_splines = [layer1_spline_outputs[h][o] for h in range(hidden_dim)]
        base_constraints.append(output_vars[o] == lpSum(incoming_splines))# Value of activation o in output layer = sum over incoming splines
    print(f"Defined {len(base_constraints)} base constraints.")


    # Solve for Min/Max of each Output
    results = []
    solver = GUROBI_CMD(path=None, keepFiles=False, mip=True, msg=True)
    for o in range(output_dim):
        print(f"\n--- Optimizing for Output Dimension {o} ---")
        target_output_var = output_vars[o]
        min_val = None
        max_val = None
    # Minimize
        prob_min = LpProblem(f"KAN_Output_{o}_Min", LpMinimize)
        prob_min += target_output_var # Objective
        for constraint in base_constraints:
            prob_min += constraint
        print(f"Solving for Min Output {o} using Gurobi...")
        prob_min.solve(solver)
        status_min = LpStatus[prob_min.status]
        if status_min == 'Optimal':
            min_val = target_output_var.varValue
            print(f"Min Output {o} found: {min_val:.6f}")
        else:
            print(f"Min Output {o} Optimization failed. Status: {status_min}")
    # Maximize
        prob_max = LpProblem(f"KAN_Output_{o}_Max", LpMaximize)
        prob_max += target_output_var # Objective
        for constraint in base_constraints:
            prob_max += constraint
        print(f"Solving for Max Output {o} using Gurobi...")
        prob_max.solve(solver) 
        status_max = LpStatus[prob_max.status]
        if status_max == 'Optimal':
            max_val = target_output_var.varValue
            print(f"Max Output {o} found: {max_val:.6f}")
        else:
            print(f"Max Output {o} Optimization failed. Status: {status_max}")
        results.append((min_val, max_val))

    return results

def compute_propagated_error(kan_model, all_segments):
    """
    Compute the overall error for the vanilla fitting method.
    This is used to set the target for the DP fitting.
    """
    segment_dict = {}
    for segment_data in all_segments:
        layer_idx = segment_data['layer_idx']
        input_idx = segment_data['input_idx']
        output_idx = segment_data['output_idx']
        spline_key = (layer_idx, input_idx, output_idx)
        segment_dict[spline_key] = segment_data
    
    # Calculate Lipschitz constants for each B-spline, then compute propagated error
    lipschitz_constants = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                lip_const = calculate_bspline_lipschitz_constant(layer=layer, input_idx=input_idx,output_idx=output_idx)
                lipschitz_constants[spline_key] = lip_const
    spline_errors = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                if spline_key not in segment_dict:
                    continue
                local_error = segment_dict[spline_key]['max_error']
                downstream_splines = get_all_downstream_splines(kan_model, layer_idx, output_idx)
                # Compute the product of Lipschitz constants of all downstream splines
                lipschitz_product = 1.0
                for ds_spline in downstream_splines:
                    if ds_spline in lipschitz_constants:
                        lipschitz_product *= lipschitz_constants[ds_spline]
                propagated_error = local_error * lipschitz_product
                spline_errors[spline_key] = propagated_error
    # Compute overall propagated error (max across all splines)
    if spline_errors:
        total_propagated_error = max(spline_errors.values())
    else:
        total_propagated_error = float('inf')
    
    return total_propagated_error, spline_errors, lipschitz_constants

def check_bounds(
    mip_min: Optional[float],
    mip_max: Optional[float],
    sampled_min: float,
    sampled_max: float,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3
) -> Tuple[bool, bool, str]:
    message = []
    lower_bound_ok = False
    upper_bound_ok = False

    if mip_min is None:
        message.append("MIP Min bound is None.")
    else:
        # Check if MIP lower bound is less than or equal to sampled lower bound (allowing tolerance)
        # Tolerance = abs_tol + rel_tol * |sampled_min|
        lower_tol = abs_tol + rel_tol * abs(sampled_min)
        if mip_min <= sampled_min + lower_tol:
            lower_bound_ok = True
        else:
            message.append(f"Lower bound failed: MIP_min ({mip_min:.6f}) > Sampled_min ({sampled_min:.6f}) + Tol ({lower_tol:.6f})")

    if mip_max is None:
        message.append("MIP Max bound is None.")
    else:
        # Check if MIP upper bound is greater than or equal to sampled upper bound (allowing tolerance)
        # Tolerance = abs_tol + rel_tol * |sampled_max|
        upper_tol = abs_tol + rel_tol * abs(sampled_max)
        if mip_max >= sampled_max - upper_tol:
            upper_bound_ok = True
        else:
            message.append(f"Upper bound failed: MIP_max ({mip_max:.6f}) < Sampled_max ({sampled_max:.6f}) - Tol ({upper_tol:.6f})")

    return lower_bound_ok, upper_bound_ok, ". ".join(message)
