# ============================================================================
# RUNNING EXPERIMENTS
# ============================================================================

from src.kan_verification_grapher import *
from src.kan_verification_trainer import *
from src.kan_verification_fitter import *
import src.kan_verification_milp as kan_milp
import datetime
import numpy as np
import pandas as pd
import src.kan_verification_grapher as kan_verification_grapher
import time
import torch
from typing import Dict, Any
import os
import sys

def run_kan_input_range_experiment(
    kan_model,
    starting_range=0.05,
    range_increment=0.05,
    max_range=1.0,
    timeout_seconds=600,
    segments_per_curve=5,
    output_root_dir=None,
    validate_bounds=True,
    num_validation_samples=5000,
    min_x=-5.0,
    max_x=5.0,
    seed=42
):
    input_dim = kan_model.layers[0].input_dim
    hidden_dim = kan_model.layers[0].output_dim  
    output_dim = kan_model.layers[-1].output_dim
    
    if output_root_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_dir = os.path.join("output", f"{model_size}_range_experiment_{timestamp}")
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    results_data = []
    
    current_range = starting_range
    iteration = 1
    timed_out = False
    constant = 2.5  
    
    lipschitz_max_segments = int(segments_per_curve * 3)  
    
    while current_range <= max_range and not timed_out:
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}: Testing with input range [-{current_range}, {current_range}]")
        print(f"{'=' * 80}")
        
        input_bounds = [(-current_range, current_range)] * input_dim
        
        range_str = f"{current_range:.2f}".replace('.', 'p')  
        iter_output_dir = os.path.join(output_root_dir, f"range_{range_str}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        iter_start_time = time.time()
        
        try:
            fitting_results = run_kan_fitting_only(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,  
                max_segments_lipschitz=lipschitz_max_segments,
                min_x=min_x,
                max_x=max_x,
                seed=seed
            )
            
            verification_results, tightness_metrics = run_kan_verification_with_fitting(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                input_bounds=input_bounds,
                fitting_results=fitting_results,
                validate_bounds=validate_bounds,
                num_validation_samples=num_validation_samples,
                seed=seed,
                output_dir=iter_output_dir
            )
            
            total_time = time.time() - iter_start_time
            
            if total_time > timeout_seconds:
                print(f"\nTIMEOUT: Iteration took {total_time:.2f} seconds, exceeding limit of {timeout_seconds} seconds")
                timed_out = True
            
            vanilla_fit_time = fitting_results["vanilla"]["fit_time"]
            vanilla_verify_time = verification_results["verification"].get("vanilla_total_time", 0)
            vanilla_total_time = vanilla_fit_time + vanilla_verify_time
            
            lipschitz_fit_time = fitting_results["lipschitz"].get("total_fit_time", 0)
            lipschitz_verify_time = verification_results["verification"].get("lipschitz_total_time", 0)
            lipschitz_total_time = lipschitz_fit_time + lipschitz_verify_time
            
            max_tightness = tightness_metrics.get("max_tightness", 0)
            vanilla_max_tightness = tightness_metrics.get("vanilla_max_tightness", 0)
            lipschitz_max_tightness = tightness_metrics.get("lipschitz_max_tightness", 0)
            
            bound_width = 2 * current_range  
            
            result_entry = {
                "bound_width": bound_width,
                "input_range": current_range,
                "vanilla_fit_time": vanilla_fit_time,
                "vanilla_verify_time": vanilla_verify_time,
                "vanilla_total_time": vanilla_total_time,
                "vanilla_total_segments": fitting_results["vanilla"]["total_segments"],
                "vanilla_max_tightness": vanilla_max_tightness,
                "lipschitz_fit_time": lipschitz_fit_time,
                "lipschitz_verify_time": lipschitz_verify_time,
                "lipschitz_total_time": lipschitz_total_time,
                "lipschitz_total_segments": fitting_results["lipschitz"].get("total_segments", 0),
                "lipschitz_max_tightness": lipschitz_max_tightness,
                "max_tightness": max_tightness,
                "total_time": total_time,
                "timed_out": timed_out,
                "segments_per_curve": segments_per_curve  
            }
            
            results_data.append(result_entry)
            
            print(f"\nIteration {iteration} complete: {total_time:.2f} seconds")
            print(f"  - Input Range: [-{current_range}, {current_range}], Width: {bound_width}")
            print(f"  - Vanilla total time: {vanilla_total_time:.2f} seconds")
            print(f"  - Lipschitz total time: {lipschitz_total_time:.2f} seconds")
            print(f"  - Maximum tightness: {max_tightness:.2f}%")
            
        except Exception as e:
            print(f"ERROR in iteration {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result_entry = {
                "bound_width": 2 * current_range,
                "input_range": current_range,
                "error": str(e),
                "timed_out": False,
                "segments_per_curve": segments_per_curve
            }
            results_data.append(result_entry)
        
        current_range += range_increment
        iteration += 1
    
    results_df = pd.DataFrame(results_data)
    
    csv_path = os.path.join(output_root_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    kan_verification_grapher.create_input_range_summary_plot(results_df, output_root_dir, segments_per_curve)
    
    return results_df, output_root_dir

def run_kan_fitting_only(
    kan_model, 
    input_dim: int,
    hidden_dim: int, 
    output_dim: int,
    vanilla_segments_per_curve: int = 5,
    max_segments_lipschitz: int = 15,
    min_x: float = -5.0,
    max_x: float = 5.0,
    seed: int = 42,
    memory_efficient: bool = False,
) -> Dict[str, Any]:
    
    print("=" * 80)
    print("KAN SEGMENT FITTING SETTINGS")
    print("=" * 80)
    print(f"Input Dimensions: {input_dim}")
    print(f"Hidden Dimensions: {hidden_dim}")
    print(f"Output Dimensions: {output_dim}")
    print(f"Vanilla Segments Per Curve: {vanilla_segments_per_curve}")
    print(f"Max Segments Lipschitz: {max_segments_lipschitz}")
    print(f"Min X: {min_x}")
    print(f"Max X: {max_x}")
    print(f"Random Seed: {seed}")
    
    print("=" * 80)
    print("RUNNING KAN SEGMENT FITTING")
    print("=" * 80)
    
    results = {
        "vanilla": {},
        "lipschitz": {}
    }
    
    
    print("\n" + "-" * 50)
    print("METHOD 1: VANILLA FITTING (EQUAL SEGMENTS)")
    print("-" * 50)
    
    
    vanilla_start = time.time()
    vanilla_segments = fit_kan_vanilla(kan_model, segments_per_curve=vanilla_segments_per_curve, min_x=min_x, max_x=max_x)
    total_propagated_error, _, _ = kan_milp.compute_propagated_error(kan_model, vanilla_segments)
    max_error_threshold = total_propagated_error / 2
    print("Max Error Threshold Changed to: ", total_propagated_error / 2)
    vanilla_fit_time = time.time() - vanilla_start
    
    
    total_vanilla_segments = sum(len(segment_data['segments']) for segment_data in vanilla_segments)
    max_vanilla_error = max(segment_data['max_error'] for segment_data in vanilla_segments 
                          if np.isfinite(segment_data['max_error']))
    inf_error_count_vanilla = sum(1 for segment_data in vanilla_segments 
                                if not np.isfinite(segment_data['max_error']))
    
    
    results["vanilla"]["fit_time"] = vanilla_fit_time
    results["vanilla"]["total_segments"] = total_vanilla_segments
    results["vanilla"]["max_error"] = max_vanilla_error
    results["vanilla"]["inf_error_count"] = inf_error_count_vanilla
    results["vanilla"]["segments_data"] = vanilla_segments
    
    print(f"Vanilla Fitting Time: {vanilla_fit_time:.4f} seconds")
    print(f"Total Segments: {total_vanilla_segments}")
    print(f"Max Error: {max_vanilla_error:.6f}")
    print(f"Connections with Infinite Error: {inf_error_count_vanilla}")
    
    
    print("\n" + "-" * 50)
    print("METHOD 2: LIPSCHITZ-WEIGHTED FITTING")
    print("-" * 50)
    
    problem = False
    lipschitz_segments = None
    
    
    lipschitz_dp_start = time.time()
    try:
        
        error_tables, segments_tables, lipschitz_constants = compute_dp_tables_lipschitz(
            kan_model, min_x, max_x, max_segments_lipschitz
        )
        lipschitz_dp_time = time.time() - lipschitz_dp_start
        
        
        weighting_start = time.time()
        weighted_error_tables, _, _ = weight_dp_tables_lipschitz(
            kan_model, min_x, max_x, max_segments_lipschitz
        )
        weighting_time = time.time() - weighting_start
        
        
        allocation_start = time.time()
        if memory_efficient:
            optimal_allocation, actual_segments, total_lipschitz_segments, achieved_error, _, used_backup_allocation = fit_kan_optimally_lipschitz_memory_efficient(
                weighted_error_tables, 
                segments_tables,
                max_error_threshold,
                max_segments=max_segments_lipschitz
            )
        else:
            optimal_allocation, actual_segments, total_lipschitz_segments, achieved_error, _, used_backup_allocation = fit_kan_optimally_lipschitz(
                weighted_error_tables, 
                segments_tables,
                max_error_threshold,
                max_segments=max_segments_lipschitz
            )
        allocation_time = time.time() - allocation_start
        
        
        lipschitz_segments = []
        for (layer_idx, input_idx, output_idx), num_segments in optimal_allocation.items():
            segments = actual_segments.get((layer_idx, input_idx, output_idx), [])
            max_error = error_tables.get((layer_idx, input_idx, output_idx), {}).get(num_segments, np.inf)
            lipschitz_segments.append({
                'layer_idx': layer_idx,
                'input_idx': input_idx,
                'output_idx': output_idx,
                'segments': segments,
                'max_error': max_error
            })
        
        
        lipschitz_fit_time = lipschitz_dp_time + weighting_time + allocation_time
        inf_error_count_lipschitz = sum(1 for segment_data in lipschitz_segments 
                                      if not np.isfinite(segment_data['max_error']))
        
        
        print("LIPSCHITZ ERROR: " + str(achieved_error if achieved_error is not None else "N/A"))
        results["lipschitz"]["dp_time"] = lipschitz_dp_time
        results["lipschitz"]["weighting_time"] = weighting_time
        results["lipschitz"]["allocation_time"] = allocation_time
        results["lipschitz"]["total_fit_time"] = lipschitz_fit_time
        results["lipschitz"]["total_segments"] = total_lipschitz_segments
        results["lipschitz"]["achieved_error"] = achieved_error
        results["lipschitz"]["inf_error_count"] = inf_error_count_lipschitz
        results["lipschitz"]["segments_data"] = lipschitz_segments
        
        print(f"Lipschitz DP Computation Time: {lipschitz_dp_time:.4f} seconds")
        print(f"Weighting Time: {weighting_time:.4f} seconds")
        print(f"Allocation Time: {allocation_time:.4f} seconds")
        print(f"Total Lipschitz Fitting Time: {lipschitz_fit_time:.4f} seconds")
        print(f"Total Segments: {total_lipschitz_segments}")
        print(f"Achieved Error: {achieved_error:.6f}")
        print(f"Connections with Infinite Error: {inf_error_count_lipschitz}")
    except Exception as e:
        print(f"ERROR: Lipschitz method failed: {str(e)}")
        results["lipschitz"]["error"] = str(e)
        
        lipschitz_segments = vanilla_segments  
    
    if problem:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING: LIPSCHITZ error too small, used best possible error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("\nLIPSCHITZ error was valid")
    
    return results

def calculate_tightness_metrics(results):
    metrics = {}
    
    if "validation" not in results or "sampled_min" not in results["validation"]:
        return metrics
    
    sampled_min = results["validation"]["sampled_min"]
    sampled_max = results["validation"]["sampled_max"]
    
    if "vanilla_results" in results["verification"]:
        vanilla_results = results["verification"]["vanilla_results"]
        vanilla_tightness = []
        
        for i in range(len(vanilla_results)):
            mip_min, mip_max = vanilla_results[i]
            samp_min, samp_max = sampled_min[i], sampled_max[i]
            
            mip_width = mip_max - mip_min
            sampled_width = samp_max - samp_min
            
            tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
            vanilla_tightness.append(tightness)
        
        metrics["vanilla_tightness"] = vanilla_tightness
        metrics["vanilla_max_tightness"] = max(vanilla_tightness) if vanilla_tightness else 0
        metrics["vanilla_avg_tightness"] = sum(vanilla_tightness) / len(vanilla_tightness) if vanilla_tightness else 0
    
    if "lipschitz_results" in results["verification"]:
        lipschitz_results = results["verification"]["lipschitz_results"]
        lipschitz_tightness = []
        
        for i in range(len(lipschitz_results)):
            mip_min, mip_max = lipschitz_results[i]
            samp_min, samp_max = sampled_min[i], sampled_max[i]
            
            mip_width = mip_max - mip_min
            sampled_width = samp_max - samp_min
            
            tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
            lipschitz_tightness.append(tightness)
        
        metrics["lipschitz_tightness"] = lipschitz_tightness
        metrics["lipschitz_max_tightness"] = max(lipschitz_tightness) if lipschitz_tightness else 0
        metrics["lipschitz_avg_tightness"] = sum(lipschitz_tightness) / len(lipschitz_tightness) if lipschitz_tightness else 0
    
    max_tightness = max(
        metrics.get("vanilla_max_tightness", 0),
        metrics.get("lipschitz_max_tightness", 0)
    )
    metrics["max_tightness"] = max_tightness
    
    return metrics

def run_kan_verification_with_fitting(
    kan_model,
    input_dim,
    hidden_dim,
    output_dim,
    input_bounds,
    fitting_results,
    time_limit_seconds=6000,
    validate_bounds=True,
    num_validation_samples=5000,
    abs_tol=1e-3,
    rel_tol=1e-2,
    seed=42,
    output_dir=None
):
    
    if output_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        if input_bounds:
            input_range = f"{input_bounds[0][0]}_{input_bounds[0][1]}"
            input_range = input_range.replace('-', 'neg')  
            input_range = input_range.replace('.', 'p')    
        else:
            input_range = "unknown"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"{model_size}_{input_range}_{timestamp}")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_stdout = sys.stdout
    output_md_path = os.path.join(output_dir, "output.md")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    
    x_min_vec = [bound[0] for bound in input_bounds]
    x_max_vec = [bound[1] for bound in input_bounds]
    
    
    vanilla_segments = fitting_results["vanilla"]["segments_data"]
    if "error" not in fitting_results["lipschitz"]:
        lipschitz_segments = fitting_results["lipschitz"]["segments_data"]
        lipschitz_successful = True
    else:
        lipschitz_segments = vanilla_segments  
        lipschitz_successful = False
    
    
    results = {
        "vanilla": fitting_results["vanilla"],
        "lipschitz": fitting_results["lipschitz"],
        "verification": {},
        "validation": {}
    }
    
    with open(output_md_path, 'w') as f:
        
        sys.stdout = Tee(original_stdout, f)
        
        print("=" * 80)
        print("VERIFICATION SETTINGS")
        print("=" * 80)
        print(f"Input Bounds: {input_bounds}")
        print(f"Time Limit (seconds): {time_limit_seconds}")
        print(f"Validate Bounds: {validate_bounds}")
        print(f"Number of Validation Samples: {num_validation_samples}")
        print(f"Absolute Tolerance: {abs_tol}")
        print(f"Relative Tolerance: {rel_tol}")
        
        
        print("\n" + "-" * 50)
        print("VERIFICATION COMPARISON")
        print("-" * 50)
        
        lipschitz_all_bounds_valid = True
        vanilla_all_bounds_valid = True
        
        
        if lipschitz_successful:
            
            lipschitz_start_verify = time.time()
            print("\nConverting Lipschitz segments to verification format...")
            layer0_lipschitz, layer1_lipschitz = kan_milp.prep_segments_data(lipschitz_segments, input_dim, hidden_dim, output_dim)
            
            
            print("\nRunning MIP verification for Lipschitz segments...")
            lipschitz_verify_start = time.time()
            lipschitz_results = kan_milp.MIP_interval_analysis(layer0_lipschitz, layer1_lipschitz, x_min_vec, x_max_vec, M=1e4)
            lipschitz_verify_time = time.time() - lipschitz_verify_start
            total_lipschitz_time = time.time() - lipschitz_start_verify
            
            
            results["verification"]["lipschitz_conversion_time"] = lipschitz_verify_start - lipschitz_start_verify
            results["verification"]["lipschitz_mip_time"] = lipschitz_verify_time
            results["verification"]["lipschitz_total_time"] = total_lipschitz_time
            results["verification"]["lipschitz_results"] = lipschitz_results
            
            print(f"Lipschitz Verification Time: {lipschitz_verify_time:.4f} seconds")
            print(f"Lipschitz Total Time (conversion + verification): {total_lipschitz_time:.4f} seconds")
        
        
        vanilla_start_verify = time.time()
        print("\nConverting vanilla segments to verification format...")
        layer0_vanilla, layer1_vanilla = kan_milp.prep_segments_data(vanilla_segments, input_dim, hidden_dim, output_dim)
        
        
        print("\nRunning MIP verification for vanilla segments...")
        vanilla_verify_start = time.time()
        vanilla_results = kan_milp.MIP_interval_analysis(layer0_vanilla, layer1_vanilla, x_min_vec, x_max_vec, M=1e4)
        vanilla_verify_time = time.time() - vanilla_verify_start
        total_vanilla_time = time.time() - vanilla_start_verify
        
        
        results["verification"]["vanilla_conversion_time"] = vanilla_verify_start - vanilla_start_verify
        results["verification"]["vanilla_mip_time"] = vanilla_verify_time
        results["verification"]["vanilla_total_time"] = total_vanilla_time
        results["verification"]["vanilla_results"] = vanilla_results
        
        print(f"Vanilla Verification Time: {vanilla_verify_time:.4f} seconds")
        print(f"Vanilla Total Time (conversion + verification): {total_vanilla_time:.4f} seconds")
        
        
        if validate_bounds:
            print("\n" + "-" * 50)
            print("VALIDATING MIP BOUNDS")
            print("-" * 50)
            
            
            print(f"\nGenerating {num_validation_samples} validation samples...")
            validation_start = time.time()
            
            
            np.random.seed(seed)
            x_test_np = np.random.uniform(
                low=x_min_vec,
                high=x_max_vec,
                size=(num_validation_samples, input_dim)
            )
            x_test = torch.tensor(x_test_np, dtype=torch.float32)
            
            
            print("Running KAN model on validation samples...")
            with torch.no_grad():
                kan_model.eval()
                y_test = kan_model(x_test).detach().cpu().numpy()
            
            
            sampled_min = np.min(y_test, axis=0)
            sampled_max = np.max(y_test, axis=0)
            
            
            results["validation"]["x_samples"] = x_test_np
            results["validation"]["y_samples"] = y_test
            results["validation"]["sampled_min"] = sampled_min
            results["validation"]["sampled_max"] = sampled_max
            
            
            print("\nValidating vanilla MIP bounds...")
            vanilla_bound_checks = []
            vanilla_all_bounds_valid = True
            
            for o in range(output_dim):
                mip_min, mip_max = vanilla_results[o]
                samp_min, samp_max = sampled_min[o], sampled_max[o]
                
                print(f"Output {o}:")
                print(f"  MIP bounds:    [{mip_min:.6f}, {mip_max:.6f}]")
                print(f"  Sampled bounds: [{samp_min:.6f}, {samp_max:.6f}]")
                
                lower_ok, upper_ok, message = kan_milp.check_bounds(
                    mip_min, mip_max, samp_min, samp_max, abs_tol, rel_tol
                )
                
                bound_valid = lower_ok and upper_ok
                vanilla_all_bounds_valid = vanilla_all_bounds_valid and bound_valid
                vanilla_bound_checks.append({
                    "output_dim": o,
                    "mip_bounds": (mip_min, mip_max),
                    "sampled_bounds": (samp_min, samp_max),
                    "lower_valid": lower_ok,
                    "upper_valid": upper_ok,
                    "message": message,
                    "valid": bound_valid
                })
                
                if bound_valid:
                    print(f"  ✓ Bounds valid")
                else:
                    print(f"  ✗ Bounds invalid: {message}")
            
            results["validation"]["vanilla_bound_checks"] = vanilla_bound_checks
            results["validation"]["vanilla_all_bounds_valid"] = vanilla_all_bounds_valid
            
            if vanilla_all_bounds_valid:
                print("\n✓ All vanilla MIP bounds are valid")
            else:
                print("\n✗ Some vanilla MIP bounds are invalid")
            
            
            if lipschitz_successful:
                print("\nValidating Lipschitz MIP bounds...")
                lipschitz_bound_checks = []
                lipschitz_all_bounds_valid = True
                
                for o in range(output_dim):
                    mip_min, mip_max = lipschitz_results[o]
                    samp_min, samp_max = sampled_min[o], sampled_max[o]
                    
                    print(f"Output {o}:")
                    print(f"  MIP bounds:    [{mip_min:.6f}, {mip_max:.6f}]")
                    print(f"  Sampled bounds: [{samp_min:.6f}, {samp_max:.6f}]")
                    
                    lower_ok, upper_ok, message = kan_milp.check_bounds(
                        mip_min, mip_max, samp_min, samp_max, abs_tol, rel_tol
                    )
                    
                    bound_valid = lower_ok and upper_ok
                    lipschitz_all_bounds_valid = lipschitz_all_bounds_valid and bound_valid
                    lipschitz_bound_checks.append({
                        "output_dim": o,
                        "mip_bounds": (mip_min, mip_max),
                        "sampled_bounds": (samp_min, samp_max),
                        "lower_valid": lower_ok,
                        "upper_valid": upper_ok,
                        "message": message,
                        "valid": bound_valid
                    })
                    
                    if bound_valid:
                        print(f"  ✓ Bounds valid")
                    else:
                        print(f"  ✗ Bounds invalid: {message}")
                
                results["validation"]["lipschitz_bound_checks"] = lipschitz_bound_checks
                results["validation"]["lipschitz_all_bounds_valid"] = lipschitz_all_bounds_valid
                
                if lipschitz_all_bounds_valid:
                    print("\n✓ All Lipschitz MIP bounds are valid")
                else:
                    print("\n✗ Some Lipschitz MIP bounds are invalid")
        
        if not (lipschitz_all_bounds_valid and vanilla_all_bounds_valid):
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING: VERIFICATION FAILED FOR ONE OR BOTH METHODS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:  
            print("\nVERIFICATION PASSED FOR BOTH METHODS")
    
    
    sys.stdout = original_stdout
    
    
    kan_verification_grapher.visualize_comparison_results(results, output_dir)
    kan_verification_grapher.detailed_metrics_visualization(results, output_dir)
    if validate_bounds and "create_relative_width_ratio_graph" in globals():
        kan_verification_grapher.create_relative_width_ratio_graph(results, output_dir)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Console output: {output_md_path}")
    print(f"  - Comparison chart: {os.path.join(output_dir, 'output.png')}")
    print(f"  - Detailed metrics: {os.path.join(output_dir, 'detailed_metrics.png')}")
    
    
    tightness_metrics = calculate_tightness_metrics(results) if validate_bounds else {}
    
    return results, tightness_metrics

def run_kan_bound_tightness_experiment(
    kan_model,
    input_bounds,
    timeout_seconds,
    starting_segments=3,
    segment_increment=5,
    max_segments=10,
    output_root_dir=None,
    validate_bounds=True,
    num_validation_samples=5000,
    min_x=-5.0,
    max_x=5.0,
    seed=42
):
    input_dim = kan_model.layers[0].input_dim
    hidden_dim = kan_model.layers[0].output_dim  
    output_dim = kan_model.layers[-1].output_dim
    
    if output_root_dir is None:
        model_size = f"{input_dim}_{hidden_dim}_{output_dim}"
        if input_bounds:
            input_range = f"{input_bounds[0][0]}_{input_bounds[0][1]}"
            input_range = input_range.replace('-', 'neg')  
            input_range = input_range.replace('.', 'p')    
        else:
            input_range = "unknown"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_dir = os.path.join("output", f"{model_size}_{input_range}_{timestamp}")
    
    os.makedirs(output_root_dir, exist_ok=True)
    
    results_data = []
    
    segments = starting_segments
    iteration = 3
    timed_out = False
    constant = 2.5
    while segments <= max_segments and not timed_out:
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}: Testing with {segments} segments")
        print(f"{'=' * 80}")
        
        iter_output_dir = os.path.join(output_root_dir, f"segments_{segments}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        iter_start_time = time.time()
        
        try:
            lipschitz_max_segments = int(segments * 3 * (iteration + 0.4)**0.185)
            constant = constant * 2 * (iteration + 0.4)**0.075
            
            fitting_results = run_kan_fitting_only(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments,
                max_segments_lipschitz=lipschitz_max_segments,
                min_x=min_x,
                max_x=max_x,
                seed=seed,
            )
            
            verification_results, tightness_metrics = run_kan_verification_with_fitting(
                kan_model=kan_model,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                input_bounds=input_bounds,
                fitting_results=fitting_results,
                validate_bounds=validate_bounds,
                num_validation_samples=num_validation_samples,
                seed=seed,
                output_dir=iter_output_dir
            )
            
            total_time = time.time() - iter_start_time
            
            if total_time > timeout_seconds:
                print(f"\nTIMEOUT: Iteration took {total_time:.2f} seconds, exceeding limit of {timeout_seconds} seconds")
                timed_out = True
            
            vanilla_fit_time = fitting_results["vanilla"]["fit_time"]
            vanilla_verify_time = verification_results["verification"].get("vanilla_total_time", 0)
            vanilla_total_time = vanilla_fit_time + vanilla_verify_time
            
            lipschitz_fit_time = fitting_results["lipschitz"].get("total_fit_time", 0)
            lipschitz_verify_time = verification_results["verification"].get("lipschitz_total_time", 0)
            lipschitz_total_time = lipschitz_fit_time + lipschitz_verify_time
            
            max_tightness = tightness_metrics.get("max_tightness", 0)
            vanilla_max_tightness = tightness_metrics.get("vanilla_max_tightness", 0)
            lipschitz_max_tightness = tightness_metrics.get("lipschitz_max_tightness", 0)
            
            result_entry = {
                "segments": segments,
                "vanilla_fit_time": vanilla_fit_time,
                "vanilla_verify_time": vanilla_verify_time,
                "vanilla_total_time": vanilla_total_time,
                "vanilla_total_segments": fitting_results["vanilla"]["total_segments"],
                "vanilla_max_tightness": vanilla_max_tightness,
                "lipschitz_fit_time": lipschitz_fit_time,
                "lipschitz_verify_time": lipschitz_verify_time,
                "lipschitz_total_time": lipschitz_total_time,
                "lipschitz_total_segments": fitting_results["lipschitz"].get("total_segments", 0),
                "lipschitz_max_tightness": lipschitz_max_tightness,
                "max_tightness": max_tightness,
                "total_time": total_time,
                "timed_out": timed_out
            }
            
            results_data.append(result_entry)
            
            print(f"\nIteration {iteration} complete: {total_time:.2f} seconds")
            print(f"  - Vanilla total time: {vanilla_total_time:.2f} seconds")
            print(f"  - Lipschitz total time: {lipschitz_total_time:.2f} seconds")
            print(f"  - Maximum tightness: {max_tightness:.2f}%")
            
        except Exception as e:
            print(f"ERROR in iteration {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result_entry = {
                "segments": segments,
                "error": str(e),
                "timed_out": False
            }
            results_data.append(result_entry)
        
        segments += segment_increment
        iteration += 1
    
    results_df = pd.DataFrame(results_data)
    
    csv_path = os.path.join(output_root_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    kan_verification_grapher.create_experiment_summary_plot(results_df, output_root_dir)
    
    return results_df, output_root_dir