import json
import os
import sys
import time
from typing import List

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gurobipy import GRB
from joblib import Parallel, delayed

log_filename    = "logs/verification_log.txt"
MODEL_DIR       = 'model_pkls'
# FILENAME_FILTER = ["xy", "exp100", "pinn", "exp4"]
FILENAME_FILTER = ["prosthetic", "weather", "acopf_ml4acopf_"]
# FILENAME_FILTER = ["func"]
EXCLUDE         = []
SEGMENT_COUNTS  = [2, 3, 4, 5, 6, 7]
INPUT_WIDTHS    = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
TIME_LIMIT      = 1000
MIP_GAP         = 0.05
MAX_SEGMENTS    = SEGMENT_COUNTS[-1]
FIXED_IW        = INPUT_WIDTHS[-1]
K_MID           = SEGMENT_COUNTS[len(SEGMENT_COUNTS) // 2]
MAX_NEURON_TO_SOLVE = 5

# Curve-sampling grid sizes for domain-aware fitting (bug fix 1)
N_GRID_UNIFORM  = 1000
N_GRID_DENSE    = 800
N_GRID_TAIL     = 240

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, denominator: float = None):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, use_base_update: bool = True, use_layernorm: bool = True, base_activation=F.silu, spline_weight_init_scale: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def eval_curves(self, input_idx: int, x_np: np.ndarray) -> np.ndarray:
        """Evaluate the learned univariate curves for one input index at
        arbitrary x locations (bug fix 1: curves must be sampled over each
        unit's actual reachable domain, not a fixed window)."""
        ng = self.rbf.num_grids
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float64)).to(self.spline_linear.weight.dtype)
        with torch.no_grad():
            basis = self.rbf(x)
            w = self.spline_linear.weight.view(self.output_dim, self.input_dim, ng)
            y = torch.einsum('pg, og -> po', basis, w[:, input_idx, :])
        return y.cpu().numpy()


class FastKAN(nn.Module):
    def __init__(self, layers_hidden: List[int], grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, use_base_update: bool = True, use_layernorm: bool = True, base_activation=F.silu, spline_weight_init_scale: float = 0.1) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.layers = nn.ModuleList([FastKANLayer(in_dim, out_dim, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids, use_base_update=use_base_update, use_layernorm=use_layernorm, base_activation=base_activation, spline_weight_init_scale=spline_weight_init_scale) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, use_layernorm=self.use_layernorm)
        return x


class AttentionWithFastKANTransform(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, head_dim: int, num_heads: int, gating: bool = True, use_layernorm: bool = True):
        super(AttentionWithFastKANTransform, self).__init__()
        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.use_layernorm = use_layernorm
        self.linear_q = FastKANLayer(q_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_k = FastKANLayer(k_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_v = FastKANLayer(v_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_o = FastKANLayer(total_dim, q_dim, use_layernorm=use_layernorm)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim, use_layernorm=use_layernorm)
        self.norm = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        wq = self.linear_q(q, use_layernorm=self.use_layernorm).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm
        wk = self.linear_k(k, use_layernorm=self.use_layernorm).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)
        att = (wq * wk).sum(-1).softmax(-2)
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]
        wv = self.linear_v(v, use_layernorm=self.use_layernorm).view(*v.shape[:-2], 1, v.shape[-2], self.num_heads, -1)
        o = (att[..., None] * wv).sum(-3)
        del att, wv
        o = o.view(*o.shape[:-2], -1)
        if self.linear_g is not None:
            g = self.linear_g(q, use_layernorm=self.use_layernorm)
            o = torch.sigmoid(g) * o
        o = self.linear_o(o, use_layernorm=self.use_layernorm)
        return o


def fit_line_through_points(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def find_bspline_segments_from_data(x_points, y_points, max_segments):
    n_samples = len(x_points)

    errors = np.full((n_samples, n_samples), np.inf)
    for start_idx in range(n_samples - 1):
        x_start = x_points[start_idx]
        y_start = y_points[start_idx]
        for end_idx in range(start_idx + 1, n_samples):
            x_end = x_points[end_idx]
            y_end = y_points[end_idx]
            slope, intercept = fit_line_through_points(x_start, y_start, x_end, y_end)
            segment_x = x_points[start_idx:end_idx + 1]
            segment_y = y_points[start_idx:end_idx + 1]
            predicted_y = slope * segment_x + intercept
            segment_error = np.max(np.abs(predicted_y - segment_y))
            errors[start_idx, end_idx] = segment_error

    dp_table = np.full((max_segments, n_samples), np.inf)
    backtrack = np.zeros((max_segments, n_samples), dtype=int)

    for j in range(1, n_samples):
        dp_table[0, j] = errors[0, j]

    for i in range(1, max_segments):
        for j in range(i + 1, n_samples):
            for k in range(i, j):
                if dp_table[i - 1, k] == np.inf: continue
                if errors[k, j] == np.inf: continue
                curr_error = max(dp_table[i - 1, k], errors[k, j])
                if curr_error < dp_table[i, j]:
                    dp_table[i, j] = curr_error
                    backtrack[i, j] = k

    results = {}
    for seg_count in range(1, max_segments + 1):
        row_idx = seg_count - 1
        final_error = dp_table[row_idx, n_samples - 1]
        if final_error == np.inf:
            results[seg_count] = ([], np.inf)
            continue

        segments = []
        curr_seg_end = n_samples - 1
        for i in range(row_idx, -1, -1):
            if i > 0:
                prev_seg_end = backtrack[i, curr_seg_end]
            else:
                prev_seg_end = 0
            x1, y1 = x_points[prev_seg_end], y_points[prev_seg_end]
            x2, y2 = x_points[curr_seg_end], y_points[curr_seg_end]
            slope, intercept = fit_line_through_points(x1, y1, x2, y2)
            segments.insert(0, (x1, x2, slope, intercept))
            curr_seg_end = prev_seg_end

        results[seg_count] = (segments, final_error)

    return results


def calculate_lipschitz_from_data(x_np, y_np):
    # bug fix 6: estimated over the unit's full sampled domain, no grid-window mask
    dx = np.diff(x_np)
    dy = np.diff(y_np)
    nonzero_dx = dx != 0
    slopes = np.zeros_like(dx)
    slopes[nonzero_dx] = dy[nonzero_dx] / dx[nonzero_dx]
    if len(slopes) == 0: return 0.0
    return float(np.max(np.abs(slopes)))


def validate_segments_error(segments, x_high, y_high):
    max_error = 0.0
    for (sx1, sx2, slope, intercept) in segments:
        mask = (x_high >= sx1) & (x_high <= sx2)
        if not np.any(mask):
            continue
        y_real = y_high[mask]
        y_pred = slope * x_high[mask] + intercept
        current_max = np.max(np.abs(y_real - y_pred))
        if current_max > max_error:
            max_error = current_max
    return max_error


def build_unit_grid(layer, lo, hi):
    """Sampling grid over the unit's actual input domain [lo, hi] (bug fix 1):
    dense inside the RBF active window, sparser over the flat Gaussian tails."""
    h = layer.rbf.denominator
    w_lo = layer.rbf.grid_min - 3.0 * h
    w_hi = layer.rbf.grid_max + 3.0 * h
    if lo >= w_lo and hi <= w_hi:
        return np.linspace(lo, hi, N_GRID_UNIFORM)
    pts = [np.linspace(lo, hi, N_GRID_TAIL)]
    d_lo, d_hi = max(lo, w_lo), min(hi, w_hi)
    if d_lo < d_hi:
        pts.append(np.linspace(d_lo, d_hi, N_GRID_DENSE))
    return np.unique(np.concatenate(pts))


def process_single_spline(key, x_grid, y_vals, max_segments):
    dp_resolution = max(50, max_segments * 2)
    idx = np.unique(np.linspace(0, len(x_grid) - 1, dp_resolution, dtype=int))
    x_dp, y_dp = x_grid[idx], y_vals[idx]
    results_low_res = find_bspline_segments_from_data(x_dp, y_dp, max_segments)
    results_validated = {}
    for k, (segs, low_res_err) in results_low_res.items():
        if low_res_err == np.inf or len(segs) == 0:
            results_validated[k] = ([], np.inf)
            continue
        true_error = validate_segments_error(segs, x_grid, y_vals)
        results_validated[k] = (segs, true_error)
    lip = calculate_lipschitz_from_data(x_grid, y_vals)
    return key, results_validated, lip


def compute_dp_tables_lipschitz(kan_model, max_segments, input_lb, input_ub):
    """Bug fix 1: fit every unit over its actual reachable domain, obtained by
    interval analysis. Hidden-unit domains are padded by the exactly computed
    1-piece chord error e1, which upper-bounds any allocation's error, so every
    PWA is defined (and its error bound valid) wherever the MILP can evaluate it.
    Also returns the sampled curves (shared by the uniform baseline) and the
    per-layer domains."""
    error_tables, segments_tables, lipschitz_constants = {}, {}, {}
    curves = {}
    domains = [(np.array(input_lb, dtype=float), np.array(input_ub, dtype=float))]

    cur_lo, cur_hi = domains[0]
    for layer_idx, layer in enumerate(kan_model.layers):
        tasks = []
        for input_idx in range(layer.input_dim):
            x_grid = build_unit_grid(layer, cur_lo[input_idx], cur_hi[input_idx])
            y_all = layer.eval_curves(input_idx, x_grid)
            for output_idx in range(layer.output_dim):
                key = (layer_idx, input_idx, output_idx)
                curves[key] = (x_grid, y_all[:, output_idx].astype(float))
                tasks.append((key, x_grid, y_all[:, output_idx].astype(float)))

        results_list = Parallel(n_jobs=-1)(
            delayed(process_single_spline)(key, xg, yv, max_segments) for key, xg, yv in tasks
        )

        next_lo = np.zeros(layer.output_dim)
        next_hi = np.zeros(layer.output_dim)
        for key, all_results, lip_const in results_list:
            _, _, output_idx = key
            error_table, segments_table = {}, {}
            for k, (segs, err) in all_results.items():
                segments_table[k] = segs
                error_table[k] = err
            error_tables[key] = error_table
            segments_tables[key] = segments_table
            lipschitz_constants[key] = lip_const
            x_grid, y_vals = curves[key]
            e1 = error_table.get(1, np.inf)
            if not np.isfinite(e1):
                e1 = float(np.max(y_vals) - np.min(y_vals))
            next_lo[output_idx] += float(np.min(y_vals)) - e1
            next_hi[output_idx] += float(np.max(y_vals)) + e1
        domains.append((next_lo, next_hi))
        cur_lo, cur_hi = next_lo, next_hi

    return error_tables, segments_tables, lipschitz_constants, curves, domains


def weight_dp_tables_lipschitz(kan_model, error_tables, segments_tables, lipschitz_constants):
    node_sensitivities = {}
    num_layers = len(kan_model.layers)

    final_layer = kan_model.layers[num_layers - 1]
    for out_idx in range(final_layer.output_dim):
        node_sensitivities[(num_layers, out_idx)] = 1.0

    layer_idx = num_layers - 1
    while layer_idx >= 0:
        current_layer = kan_model.layers[layer_idx]
        for input_idx in range(current_layer.input_dim):
            sensitivity_sum = 0.0
            for output_idx in range(current_layer.output_dim):
                lipschitz_constant = lipschitz_constants.get((layer_idx, input_idx, output_idx), 0.0)
                child_node_sensitivity = node_sensitivities[(layer_idx + 1, output_idx)]
                sensitivity_sum += lipschitz_constant * child_node_sensitivity
            node_sensitivities[(layer_idx, input_idx)] = sensitivity_sum
        layer_idx -= 1

    weighted_error_tables = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                current_bspline_key = (layer_idx, input_idx, output_idx)
                sensitivity_of_current_output_node = node_sensitivities[(layer_idx + 1, output_idx)]
                weighted_table_for_bspline = {}
                for num_segments, error in error_tables[current_bspline_key].items():
                    weighted_table_for_bspline[num_segments] = error * sensitivity_of_current_output_node
                weighted_error_tables[current_bspline_key] = weighted_table_for_bspline
    return weighted_error_tables


def solve_best_segment_allocation(weighted_error_tables, target_max_error):
    model = gp.Model()
    model.setParam("Method", 6)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", TIME_LIMIT)
    x = {}
    binary_vars_for_splines = {}
    for spline_key, num_segment_options in weighted_error_tables.items():
        binary_vars_for_splines[spline_key] = []
        for k_segments, error_val in num_segment_options.items():
            if np.isinf(error_val):
                continue
            x[(spline_key, k_segments)] = model.addVar(vtype=GRB.BINARY, obj=k_segments)
            binary_vars_for_splines[spline_key].append(x[(spline_key, k_segments)])
    model.update()

    for spline_key, variables in binary_vars_for_splines.items():
        model.addConstr(gp.quicksum(variables) == 1)

    total_error_expr = gp.quicksum(weighted_error_tables[s_key][k] * x[(s_key, k)] for s_key, k in x.keys())
    model.addConstr(total_error_expr <= target_max_error)

    total_segments_expr = gp.quicksum(k * x[(s_key, k)] for s_key, k in x.keys())
    model.setObjective(total_segments_expr, GRB.MINIMIZE)

    model.optimize()
    optimal_allocation = {}
    if model.status == GRB.OPTIMAL:
        for (spline_key, k_segments), variable in x.items():
            if variable.X > 0.5:
                optimal_allocation[spline_key] = k_segments
        return optimal_allocation, model.objVal
    elif model.status == GRB.INFEASIBLE:
        return None, float('inf')
    else:
        return None, float('inf')


def get_spline_bounds(segments, x_min, x_max):
    y_min, y_max = np.inf, -np.inf
    for (sx1, sx2, slope, intercept) in segments:
        overlap_start = max(sx1, x_min)
        overlap_end = min(sx2, x_max)
        if overlap_start <= overlap_end:
            val_start = slope * overlap_start + intercept
            val_end = slope * overlap_end + intercept
            y_min = min(y_min, val_start, val_end)
            y_max = max(y_max, val_start, val_end)
    return y_min, y_max


def propagate_kan_intervals(kan_shape, segments_tables, error_tables, optimal_allocation, input_lb, input_ub):
    layer_bounds = []
    current_lb = input_lb
    current_ub = input_ub
    layer_bounds.append((current_lb, current_ub))
    num_transitions = len(kan_shape) - 1

    for layer_idx in range(num_transitions):
        in_dim = kan_shape[layer_idx]
        out_dim = kan_shape[layer_idx + 1]
        next_lb = np.zeros(out_dim)
        next_ub = np.zeros(out_dim)
        for dst in range(out_dim):
            total_min = 0.0
            total_max = 0.0
            for src in range(in_dim):
                x_min = current_lb[src]
                x_max = current_ub[src]
                num_segs = optimal_allocation[(layer_idx, src, dst)]
                segs = segments_tables[(layer_idx, src, dst)][num_segs]
                approx_error = error_tables[(layer_idx, src, dst)][num_segs]
                s_min, s_max = get_spline_bounds(segs, x_min, x_max)
                total_min += s_min - approx_error
                total_max += s_max + approx_error
            next_lb[dst] = total_min
            next_ub[dst] = total_max
        layer_bounds.append((next_lb, next_ub))
        current_lb, current_ub = next_lb, next_ub
    return layer_bounds


def build_kan_milp_model(kan_shape, segments_tables, error_tables, optimal_allocation, x_min_vec, x_max_vec):
    model = gp.Model()
    model.setParam("Method", 6)
    model.setParam("TimeLimit", TIME_LIMIT)
    input_dim = kan_shape[0]
    all_layer_variables = []

    current_layer_range_variables = []
    for i in range(input_dim):
        input_range_variable = model.addVar(lb=x_min_vec[i], ub=x_max_vec[i])
        current_layer_range_variables.append(input_range_variable)
    all_layer_variables.append(current_layer_range_variables)

    num_transitions = len(kan_shape) - 1
    for layer_idx in range(num_transitions):
        layer_input_dimension = kan_shape[layer_idx]
        layer_output_dimension = kan_shape[layer_idx + 1]

        next_layer_input_ranges = []
        for i in range(layer_output_dimension):
            next_layer_input_ranges.append(gp.LinExpr())

        for src_idx in range(layer_input_dimension):
            for dst_idx in range(layer_output_dimension):
                num_segs = optimal_allocation[(layer_idx, src_idx, dst_idx)]
                seg_data = segments_tables[(layer_idx, src_idx, dst_idx)][num_segs]
                x_pts, y_pts = [], []
                for idx, (x1, x2, slope, intercept) in enumerate(seg_data):
                    x_pts.append(x1)
                    y_pts.append(slope * x1 + intercept)
                    if idx == len(seg_data) - 1:
                        x_pts.append(x2)
                        y_pts.append(slope * x2 + intercept)

                approx_error = error_tables[(layer_idx, src_idx, dst_idx)][num_segs]
                error_var = model.addVar(lb=-approx_error, ub=approx_error)
                src_var = current_layer_range_variables[src_idx]
                result_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrPWL(src_var, result_var, x_pts, y_pts)
                next_layer_input_ranges[dst_idx] += result_var + error_var

        next_layer_range_variables = []
        for j in range(layer_output_dimension):
            next_layer_range_variable = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            model.addConstr(next_layer_range_variable == next_layer_input_ranges[j])
            next_layer_range_variables.append(next_layer_range_variable)
        current_layer_range_variables = next_layer_range_variables

        all_layer_variables.append(current_layer_range_variables)

    model.update()
    return model, all_layer_variables


def solve_kan_interval_milp(kan_shape, segments_tables, error_tables, optimal_allocation, output_layer_mip_gap, x_min_vec, x_max_vec, time_limit):
    precomputed_bounds = propagate_kan_intervals(kan_shape, segments_tables, error_tables, optimal_allocation, x_min_vec, x_max_vec)
    model, all_layer_variables = build_kan_milp_model(kan_shape, segments_tables, error_tables, optimal_allocation, x_min_vec, x_max_vec)
    model.setParam("PreSolve", -1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    for layer_idx, vars_in_layer in enumerate(all_layer_variables):
        lbs, ubs = precomputed_bounds[layer_idx]
        for neuron_idx, var in enumerate(vars_in_layer):
            var.lb = lbs[neuron_idx]
            var.ub = ubs[neuron_idx]
    model.update()

    final_layer_vars = all_layer_variables[-1]
    target_indices = [v.index for v in final_layer_vars]

    def solve_single_neuron(var_index):
        local_model = model.copy()
        local_model.setParam("MIPGap", output_layer_mip_gap)
        local_model.setParam("Threads", 1)
        local_model.setParam("OutputFlag", 0)
        local_model.setParam("TimeLimit", time_limit)
        target_var = local_model.getVars()[var_index]

        # Bug fix 3: report the solver's proven bound (ObjBound), which is a sound
        # outer bound at any MIP gap; the incumbent ObjVal can sit up to the gap
        # inside the true range. Bug fix 9: print abnormal statuses.
        local_model.setObjective(target_var, GRB.MINIMIZE)
        local_model.optimize()
        if local_model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            min_val = local_model.ObjBound
        else:
            print(f"    [MILP] min solve returned status {local_model.status}; reporting -inf")
            min_val = -float("inf")

        local_model.setObjective(target_var, GRB.MAXIMIZE)
        local_model.optimize()
        if local_model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            max_val = local_model.ObjBound
        else:
            print(f"    [MILP] max solve returned status {local_model.status}; reporting +inf")
            max_val = float("inf")
        return min_val, max_val

    if len(target_indices) >= MAX_NEURON_TO_SOLVE:
        target_indices = target_indices[:MAX_NEURON_TO_SOLVE]
    results = Parallel(n_jobs=-1, backend="threading")(delayed(solve_single_neuron)(idx) for idx in target_indices)
    min_bounds = np.array([r[0] for r in results])
    max_bounds = np.array([r[1] for r in results])
    return min_bounds, max_bounds


# def verify_mlp_gurobi_lib(model, input_lb, input_ub, output_layer_mip_gap=0.05, timeout=300):
#     from gurobi_ml import add_predictor_constr
#     start_time = time.time()
#     m = gp.Model("mlp_verification_lib")
#     m.setParam("Method", 6)
#     m.setParam("OutputFlag", 0)
#     input_dim = len(input_lb)
#     input_vars = m.addMVar((1, input_dim), lb=input_lb, ub=input_ub, name="input")
#     pred_constr = add_predictor_constr(m, model.network, input_vars)
#     output_mvar = pred_constr.output
#     m.update()
#     output_vars_list = output_mvar.tolist()[0]
#     target_indices = [v.index for v in output_vars_list]

#     def solve_single_neuron(var_index):
#         local_model = m.copy()
#         local_model.setParam("MIPGap", output_layer_mip_gap)
#         local_model.setParam("TimeLimit", timeout)
#         local_model.setParam("Threads", 1)
#         local_model.setParam("OutputFlag", 0)
#         target_var = local_model.getVars()[var_index]

#         local_model.setObjective(target_var, GRB.MINIMIZE)
#         local_model.optimize()
#         if local_model.status == GRB.OPTIMAL:
#             min_val = local_model.ObjVal
#         elif local_model.status == GRB.TIME_LIMIT:
#             min_val = local_model.ObjBound
#         else:
#             min_val = -float("inf")

#         local_model.setObjective(target_var, GRB.MAXIMIZE)
#         local_model.optimize()
#         if local_model.status == GRB.OPTIMAL:
#             max_val = local_model.ObjVal
#         elif local_model.status == GRB.TIME_LIMIT:
#             max_val = local_model.ObjBound
#         else:
#             max_val = float("inf")
#         return min_val, max_val

#     if len(target_indices) >= MAX_NEURON_TO_SOLVE:
#         target_indices = target_indices[:MAX_NEURON_TO_SOLVE]
#     results = Parallel(n_jobs=-1, backend="threading")(delayed(solve_single_neuron)(idx) for idx in target_indices)
#     min_outputs = np.array([r[0] for r in results])
#     max_outputs = np.array([r[1] for r in results])
#     total_time = time.time() - start_time
#     return min_outputs, max_outputs, total_time

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log.fileno())

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _normalize_bounds(input_lb, input_ub, input_dim):
    if np.isscalar(input_lb):
        input_lb = np.full(input_dim, input_lb, dtype=float)
    else:
        input_lb = np.array(input_lb, dtype=float)

    if np.isscalar(input_ub):
        input_ub = np.full(input_dim, input_ub, dtype=float)
    else:
        input_ub = np.array(input_ub, dtype=float)

    return input_lb, input_ub


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[16], output_dim=10):
        super(MLP, self).__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def clean_and_load_mlp(model, state_dict):
    keys = list(state_dict.keys())

    def extract_first_num(text):
        nums = [int(s) for s in text.replace('.', ' ').split() if s.isdigit()]
        return nums[0] if nums else 999

    weight_keys = sorted([k for k in keys if 'weight' in k], key=extract_first_num)
    new_state_dict = {}
    valid_weight_keys = [k for k in weight_keys if len(state_dict[k].shape) == 2]
    for i, w_key in enumerate(valid_weight_keys):
        if i * 2 >= len(model.network):
            break
        target_idx = i * 2
        new_state_dict[f"network.{target_idx}.weight"] = state_dict[w_key]
        b_key = w_key.replace('weight', 'bias')
        if b_key in state_dict:
            new_state_dict[f"network.{target_idx}.bias"] = state_dict[b_key]
    model.load_state_dict(new_state_dict, strict=False)
    return model


def infer_mlp_config(state_dict):
    weight_keys = [k for k in state_dict.keys() if 'weight' in k]

    def extract_first_num(text):
        nums = [int(s) for s in text.replace('.', ' ').split() if s.isdigit()]
        return nums[0] if nums else 999

    weight_keys.sort(key=extract_first_num)
    hidden_dims = []
    input_dim = 0
    output_dim = 0
    linear_weight_keys = []
    for key in weight_keys:
        if len(state_dict[key].shape) == 2:
            linear_weight_keys.append(key)
    if not linear_weight_keys:
        raise ValueError("No valid 2D linear weights found in state_dict.")
    for i, key in enumerate(linear_weight_keys):
        out_f, in_f = state_dict[key].shape
        if i == 0:
            input_dim = in_f
        if i < len(linear_weight_keys) - 1:
            hidden_dims.append(out_f)
        else:
            output_dim = out_f
    return input_dim, hidden_dims, output_dim


sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout
print(f"Logging started. Writing to {log_filename}...")

files = sorted(os.listdir(MODEL_DIR))
files = [f for f in files if (not FILENAME_FILTER or any(fn in f for fn in FILENAME_FILTER)) and "_kan_model.pkl" in f and not any(ex in f for ex in EXCLUDE)]
pairs = {}

for f in files:
    if "_kan_model.pkl" in f:
        prefix = f.replace("_kan_model.pkl", "")
        if prefix not in pairs: pairs[prefix] = {}
        pairs[prefix]["kan"] = f


def width(min_o, max_o):
    if min_o is None or max_o is None: return np.nan
    return float(np.mean(max_o - min_o))


def allocate_segments_under_budget(weighted_error_tables, total_budget):
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", TIME_LIMIT)
    model.setParam("Method", 6)
    x = {}
    binary_vars_for_splines = {}
    for spline_key, options in weighted_error_tables.items():
        binary_vars_for_splines[spline_key] = []
        for k_segs, err_val in options.items():
            if np.isinf(err_val):
                continue
            x[(spline_key, k_segs)] = model.addVar(vtype=GRB.BINARY)
            binary_vars_for_splines[spline_key].append(x[(spline_key, k_segs)])
    model.update()

    for spline_key, variables in binary_vars_for_splines.items():
        if not variables:
            return None, float('inf'), 0
        model.addConstr(gp.quicksum(variables) == 1)

    model.addConstr(gp.quicksum(k * x[(sk, k)] for sk, k in x.keys()) <= total_budget)
    model.setObjective(gp.quicksum(weighted_error_tables[sk][k] * x[(sk, k)] for sk, k in x.keys()), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        alloc = {sk: k for (sk, k), v in x.items() if v.X > 0.5}
        used = sum(alloc.values())
        return alloc, model.ObjVal, used
    return None, float('inf'), 0

def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]
    return obj

graph_data = {}
curves_by_prefix, shape_by_prefix, tables_by_prefix = {}, {}, {}

for prefix, pair in sorted(pairs.items(), key=lambda kv: os.path.getsize(os.path.join(MODEL_DIR, kv[1]["kan"]))):
    if "kan" not in pair: continue
    print(f"\n=== {prefix} ===")

    k_ckpt = torch.load(os.path.join(MODEL_DIR, pair["kan"]), map_location='cpu', weights_only=False)
    clean_cfg = {k: v for k, v in k_ckpt['config'].items() if k not in ['target', 'target_norm', 'input_dim', 'optimizer']}
    # Bug fix 8: reconstruct RBF geometry from the checkpointed grid tensor; some
    # checkpoints (functionexp100, pm25) store a stale config grid, which loads the
    # trained centers with the wrong bandwidth.
    grids = [v for k, v in k_ckpt['model_state_dict'].items() if k.endswith('rbf.grid')]
    if grids:
        g0 = grids[0]
        clean_cfg['grid_min'] = float(g0.min())
        clean_cfg['grid_max'] = float(g0.max())
        clean_cfg['num_grids'] = int(g0.numel())
    try:
        kan = FastKAN(**clean_cfg)
        kan.load_state_dict(k_ckpt['model_state_dict'])
    except RuntimeError:
        continue

    in_dim = kan.layers[0].input_dim
    n_params = sum(p.numel() for p in kan.parameters() if p.requires_grad)
    n_splines = sum(l.input_dim * l.output_dim for l in kan.layers)

    # Bug fixes 1/4: fit the one-time abstraction over the widest experiment
    # domain (total width, so bounds are +-w/2) so it is sound for every query.
    max_w = max(FIXED_IW, max(INPUT_WIDTHS))
    dom_lb, dom_ub = _normalize_bounds(-max_w / 2, max_w / 2, in_dim)
    t_dp0 = time.time()
    err_t, seg_t, lip, curves, domains = compute_dp_tables_lipschitz(kan, MAX_SEGMENTS, dom_lb, dom_ub)
    weighted = weight_dp_tables_lipschitz(kan, err_t, seg_t, lip)
    dp_time = time.time() - t_dp0
    theo_min = sum(min(e for e in tbl.values() if np.isfinite(e)) for tbl in weighted.values())
    print(f"  theoretical min error: {theo_min:.4f}, n_splines: {n_splines}, dp_time: {dp_time:.3f}s")

    kan_shape = [in_dim] + [l.output_dim for l in kan.layers]
    curves_by_prefix[prefix] = curves
    shape_by_prefix[prefix] = kan_shape
    tables_by_prefix[prefix] = (err_t, seg_t)

    lb, ub = _normalize_bounds(-FIXED_IW / 2, FIXED_IW / 2, in_dim)
    opt_pareto = []
    van_pareto = []
    ilp_times = []
    milp_times = []
    van_times = []

    for k in SEGMENT_COUNTS:
        total_budget = k * n_splines
        print(f"  Testing total segment budget: {total_budget} (k={k} per spline for Van)")

        t_ilp0 = time.time()
        alloc, _, _ = allocate_segments_under_budget(weighted, total_budget)
        ilp_t = time.time() - t_ilp0

        if alloc is None:
            w, milp_t, mn, mx = np.nan, 0.0, None, None
        else:
            t_milp0 = time.time()
            mn, mx = solve_kan_interval_milp(kan_shape, seg_t, err_t, alloc, MIP_GAP, lb, ub, TIME_LIMIT)
            milp_t = time.time() - t_milp0
            w = width(mn, mx)

        print(f"    Opt k={k} (budget={total_budget}): width={w:.4f}, ilp={ilp_t:.3f}s, milp={milp_t:.3f}s")
        opt_pareto.append((ilp_t + milp_t, w))
        ilp_times.append(ilp_t)
        milp_times.append(milp_t)

        # Bug fixes 2/7: Vanilla reads row k of the SAME shared trade-off tables;
        # the timed portion is the MILP solve only (previously the timer also
        # covered a full Bellman re-fit, which was then compared against
        # Optimized's MILP-only time).
        van_alloc = {key: k for key in seg_t.keys()}
        t_v0 = time.time()
        mn, mx = solve_kan_interval_milp(kan_shape, seg_t, err_t, van_alloc, MIP_GAP, lb, ub, TIME_LIMIT)
        t = time.time() - t_v0
        van_times.append(t)
        w = width(mn, mx)
        print(f"    Van k={k}: width={w:.4f}, milp={t:.3f}s")
        van_pareto.append((t, w))

    # alloc_iw, _ = solve_best_segment_allocation(weighted, theo_min * 1.5)
    alloc_iw, _, _ = allocate_segments_under_budget(weighted, K_MID * n_splines)
    n_delta = sum(alloc_iw.values()) if alloc_iw is not None else 0    
    # Bug fix 5: the delta allocation uses ~some_x the Vanilla k=K_MID budget; state the
    # budgets explicitly and also report an equal-budget optimized allocation.
    alloc_eq, _, n_eq = allocate_segments_under_budget(weighted, K_MID * n_splines)

    print(f"  Input sweep allocations: delta-based uses {n_delta} segments "
          f"({n_delta / max(n_splines, 1):.1f}/spline); equal-budget uses {n_eq} "
          f"(Van k={K_MID} uses {K_MID * n_splines})")
    opt_input_sweep = []
    opt_input_sweep_eq = []
    van_input_sweep = []
    for iw in INPUT_WIDTHS:
        lb_w, ub_w = _normalize_bounds(-iw / 2, iw / 2, in_dim)
        print(f"  Testing input width: {iw} (bounds=({lb_w[0]:.3f}, {ub_w[0]:.3f}))")

        if alloc_iw is not None:
            t0 = time.time()
            mn, mx = solve_kan_interval_milp(kan_shape, seg_t, err_t, alloc_iw, MIP_GAP, lb_w, ub_w, TIME_LIMIT)
            opt_t = time.time() - t0
        else:
            mn, mx, opt_t = None, None, 0.0
        opt_input_sweep.append((iw, width(mn, mx), opt_t))
        print(f"    Opt (delta, {n_delta} segs): width={width(mn, mx):.4f}, milp={opt_t:.3f}s")

        if alloc_eq is not None:
            t0 = time.time()
            mn, mx = solve_kan_interval_milp(kan_shape, seg_t, err_t, alloc_eq, MIP_GAP, lb_w, ub_w, TIME_LIMIT)
            opt_eq_t = time.time() - t0
        else:
            mn, mx, opt_eq_t = None, None, 0.0
        opt_input_sweep_eq.append((iw, width(mn, mx), opt_eq_t))
        print(f"    Opt (eq-budget, {n_eq} segs): width={width(mn, mx):.4f}, milp={opt_eq_t:.3f}s")

        van_alloc = {key: K_MID for key in seg_t.keys()}
        t0 = time.time()
        mn, mx = solve_kan_interval_milp(kan_shape, seg_t, err_t, van_alloc, MIP_GAP, lb_w, ub_w, TIME_LIMIT)
        t = time.time() - t0
        van_input_sweep.append((iw, width(mn, mx), t))
        print(f"    Van k={K_MID}: width={width(mn, mx):.4f}, milp={t:.3f}s")

    graph_data[prefix] = {
        'n_params': n_params,
        'opt_pareto': opt_pareto,
        'van_pareto': van_pareto,
        'opt_input_sweep': opt_input_sweep,
        'opt_input_sweep_eqbudget': opt_input_sweep_eq,
        'van_input_sweep': van_input_sweep,
        'delta_alloc_segments': n_delta,
        'eq_alloc_segments': n_eq,
        'mid_alloc_segments': K_MID * n_splines,    
        'van_sweep_segments': K_MID * n_splines,
        'dp_time': dp_time,
        'ilp_time': ilp_times,
        'milp_time': milp_times,
        'van_time': van_times,
    }

    save_payload = {
        "filename": prefix,
        "SEGMENT_COUNTS": SEGMENT_COUNTS,
        "INPUT_WIDTHS": INPUT_WIDTHS,
        "graph_data": {prefix: _to_python(graph_data[prefix])},
    }
    save_path = f"logs/graph_data_{prefix}.json"
    with open(save_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    print(f"Saved graph data to {save_path}")

prefixes = list(graph_data.keys())

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)

# def _to_python(obj):
#     if isinstance(obj, dict):
#         return {k: _to_python(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_python(v) for v in obj]
#     if isinstance(obj, float) and np.isnan(obj):
#         return None
#     if isinstance(obj, (np.floating, np.integer)):
#         return obj.item()
#     if isinstance(obj, np.ndarray):
#         return [_to_python(v) for v in obj.tolist()]
#     return obj


# for p in prefixes:
#     save_payload = {
#         "filename": p,
#         "SEGMENT_COUNTS": SEGMENT_COUNTS,
#         "INPUT_WIDTHS": INPUT_WIDTHS,
#         "graph_data": {p: _to_python(graph_data[p])},
#     }
#     save_path = f"logs/graph_data_{p}.json"
#     with open(save_path, "w") as f:
#         json.dump(save_payload, f, indent=2)
#     print(f"Saved graph data to {save_path}")

for p in sorted(prefixes, key=lambda p: graph_data[p]["n_params"]):
    d = graph_data[p]
    print(f"\n{p}  ({d['n_params']:,} params)")
    print(f"  DP abstraction (one-time): {d['dp_time']:.3f}s")
    print(f"  {'k':>4}  {'ILP (s)':>10}  {'MILP (s)':>10}  {'Van (s)':>10}")
    for k, ilp_t, milp_t, van_t in zip(SEGMENT_COUNTS, d['ilp_time'], d['milp_time'], d['van_time']):
        print(f"  {k:>4}  {ilp_t:>10.3f}  {milp_t:>10.3f}  {van_t:>10.3f}")
print("=" * 60)

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
plt.rcParams['font.size'] = 9

sorted_p = sorted(prefixes, key=lambda p: graph_data[p]["n_params"])
colors = ["black", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

for i, p in enumerate(sorted_p):
    d = graph_data[p]
    c = colors[i % len(colors)]
    label = f"{p} ({d['n_params']:,}p)"
    ax1.axhline(d['dp_time'], c=c, linestyle='-.', alpha=0.6, label=f"{label} Abstraction (one-time)")
    ax1.plot(SEGMENT_COUNTS, d['ilp_time'], label=f"{label} ILP", c=c, marker='o', linestyle='-')
    ax1.plot(SEGMENT_COUNTS, d['milp_time'], label=f"{label} MILP", c=c, marker='s', linestyle=':')

ax1.set(xlabel="Segments per Spline (k)", ylabel="Time (s)", title="Graph 1: Segments vs Time")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=6, loc='upper left')

for i, p in enumerate(sorted_p):
    d = graph_data[p]
    c = colors[i % len(colors)]
    label = f"{p} ({d['n_params']:,}p)"
    obw_opt = [x[1] for x in d['opt_pareto']]
    van_opt = [x[1] for x in d['van_pareto']]
    ax2.plot(SEGMENT_COUNTS, obw_opt, label=f"{label} Opt", c=c, marker='o')
    ax2.plot(SEGMENT_COUNTS, van_opt, label=f"{label} Van", c=c, alpha=0.6, linestyle='--', marker='>')

ax2.set(xlabel="Segments per Spline (k)", ylabel="Output Width", title="Graph 2: Segments vs Output Width")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7)

for i, p in enumerate(sorted_p):
    d = graph_data[p]
    c = colors[i % len(colors)]
    label = f"{p} ({d['n_params']:,}p)"
    iw_opt, ow_opt, _ = zip(*d["opt_input_sweep"])
    iw_van, ow_van, _ = zip(*d["van_input_sweep"])
    ax3.plot(iw_opt, ow_opt, "o-", label=f"{label} Opt", color=c, lw=2)
    ax3.plot(iw_van, ow_van, "x:", color=c, alpha=0.9, label=f"{label} Van", lw=2, markersize=8)

ax3.set(xlabel="Input Width (log scale)", ylabel="Output Width", title="Graph 3: Input Width vs Output Width", xscale='log')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=7)

fig.suptitle("KAN Verification across model sizes", fontsize=13)
plt.tight_layout()
plt.savefig("plots/verification_multi_kan.png", bbox_inches="tight", dpi=300)
plt.close(fig)


def build_uniform_segments(x_high, y_high, k):
    n = len(x_high)
    sample_indices = np.linspace(0, n - 1, k + 1, dtype=int)
    segments = []
    for i_start, i_end in zip(sample_indices[:-1], sample_indices[1:]):
        x1, y1 = x_high[i_start], y_high[i_start]
        x2, y2 = x_high[i_end], y_high[i_end]
        slope, intercept = fit_line_through_points(x1, y1, x2, y2)
        segments.append((x1, x2, slope, intercept))
    return segments


def compute_true_vanilla_tables(curves, k):
    # Bug fixes 1/10: the uniform-breakpoint baseline is built from the same
    # domain-aware curves as the other methods (previously it re-sampled the
    # fixed window).
    segments_tables = {}
    error_tables = {}
    for key, (x_high, y_high) in curves.items():
        segs = build_uniform_segments(x_high, y_high, k)
        err = validate_segments_error(segs, x_high, y_high)
        segments_tables[key] = {k: segs}
        error_tables[key] = {k: err}
    return error_tables, segments_tables


true_van_results = {}

for prefix in graph_data.keys():
    print(f"\n=== True Vanilla: {prefix} ===")
    curves = curves_by_prefix[prefix]
    kan_shape = shape_by_prefix[prefix]
    err_t, seg_t = tables_by_prefix[prefix]
    in_dim = kan_shape[0]
    lb, ub = _normalize_bounds(-FIXED_IW / 2, FIXED_IW / 2, in_dim)
    pareto = []
    for k in SEGMENT_COUNTS:
        tv_err_t, tv_seg_t = compute_true_vanilla_tables(curves, k)
        tv_alloc = {key: k for key in tv_seg_t.keys()}
        t0 = time.time()
        mn, mx = solve_kan_interval_milp(kan_shape, tv_seg_t, tv_err_t, tv_alloc, MIP_GAP, lb, ub, TIME_LIMIT)
        t = time.time() - t0
        w = width(mn, mx)
        print(f"    True Van k={k}: width={w:.4f}, milp={t:.3f}s")
        pareto.append((t, w))
    true_van_results[prefix] = pareto

sorted_p = sorted(prefixes, key=lambda p: graph_data[p]["n_params"])
colors = ["black", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, p in enumerate(sorted_p):
    d = graph_data[p]
    c = colors[i % len(colors)]
    label = f"{p} ({d['n_params']:,}p)"
    opt_total = [ilp + mt for ilp, mt in zip(d['ilp_time'], d['milp_time'])]
    ax1.plot(SEGMENT_COUNTS, opt_total, c=c, marker='o', linestyle='-', label=f"{label} Opt")
    ax1.plot(SEGMENT_COUNTS, d['van_time'], c=c, marker='s', linestyle='--', label=f"{label} Van (DP)")
    tv_times = [t for (t, _) in true_van_results[p]]
    ax1.plot(SEGMENT_COUNTS, tv_times, c=c, alpha=0.5, marker='^', linestyle=':', label=f"{label} True Van (uniform)")

ax1.set(xlabel="Segments per Spline (k)", ylabel="Time (s)", title="k vs Total Time")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=6, loc='best')

for i, p in enumerate(sorted_p):
    d = graph_data[p]
    c = colors[i % len(colors)]
    label = f"{p} ({d['n_params']:,}p)"
    opt_widths = [w for (_, w) in d['opt_pareto']]
    van_widths = [w for (_, w) in d['van_pareto']]
    tv_widths  = [w for (_, w) in true_van_results[p]]
    ax2.plot(SEGMENT_COUNTS, opt_widths, c=c, marker='o', linestyle='-', label=f"{label} Opt")
    ax2.plot(SEGMENT_COUNTS, van_widths, c=c, marker='s', linestyle='--', label=f"{label} Van (DP)")
    ax2.plot(SEGMENT_COUNTS, tv_widths, c=c, marker='^', linestyle=':', label=f"{label} True Van (uniform)")

ax2.set(xlabel="Segments per Spline (k)", ylabel="Output Bound Width", title="k vs Output Width")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=6, loc='best')

fig.suptitle("True Vanilla (uniform breakpoints) vs Vanilla (DP) vs Optimized", fontsize=13)
plt.tight_layout()
plt.savefig("plots/true_vanilla_comparison.png", bbox_inches="tight", dpi=300)
plt.close(fig)