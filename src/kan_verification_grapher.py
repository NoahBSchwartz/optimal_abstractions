import matplotlib.pyplot as plt
import scipy as sc
import math
import numpy as np
import os
import tikzplotlib
from typing import Dict, Any
import pandas as pd
from src.custom_fastkan import FastKAN, FastKANLayer

# ODE Visualization 
def system_eq_dis(cond_input, t_eval, time):
    def system_cont(t, x, d):
        dxdt = [x[1], x[0] - x[0]**3 - d*x[1]]
        return dxdt
    x0, y0, d = cond_input
    sol = sc.integrate.solve_ivp(system_cont, [0, time], (x0, y0), args=(d,), t_eval=t_eval)
    return sol.y.T, sol.t.T

def graph_ode_results(dataset, model, steps, t_eval, duration):
    for k in range(0, 1):
        plt.figure(figsize=[5, 4])
        sol, t_h = system_eq_dis(dataset[int(steps)*k, 1:4].detach().cpu(), t_eval.numpy(), duration)
        plt.plot(sol[:, 0], sol[:, 1], c='b', marker='x', label='gt')
        y_pred = model(dataset[int(steps)*k:int(steps)*(k+1)]).detach().cpu().numpy()
        plt.plot(y_pred[:, 0], y_pred[:, 1], c='r', marker='o', label='pred')
        plt.grid()
        x0, y0, d = dataset[int(steps)*k, 1:4].detach().numpy()
        plt.title(f"Test set predictions $x_0={x0:.3f}, y_0={y0:.3f}, d={d:.3f}$")
        plt.xlabel(r"$x_t$")
        plt.ylabel(r"$y_t$")
        plt.legend()
        plt.show() 


# Vanilla Visualization
def plot_kan(
    kan_model,
    all_segments,
    figsize=(15, 10),
    min_x=-5.0,
    max_x=5.0,
    num_pts=500,
):
    total_curves = len(all_segments)
    cols = math.ceil(math.sqrt(total_curves))
    rows = math.ceil(total_curves / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle('FastKAN Spline Curves Visualization', fontsize=16)
    axes = axes.flatten()
    for idx, segment_data in enumerate(all_segments):
        layer_idx = segment_data['layer_idx']
        input_index = segment_data['input_idx']
        output_index = segment_data['output_idx']
        segments = segment_data['segments']
        max_error = segment_data['max_error']
        ax = axes[idx]
        layer = kan_model.layers[layer_idx]
        # Plot original curve
        x_tensor, y_tensor = layer.plot_curve(
            input_index, output_index, 
            num_pts=num_pts
        )
        x_orig = x_tensor.detach().cpu().numpy()
        y_orig = y_tensor.detach().cpu().numpy()
        mask = (x_orig >= min_x) & (x_orig <= max_x)
        x_orig = x_orig[mask]
        y_orig = y_orig[mask]
        ax.plot(x_orig, y_orig, 'b-', linewidth=1.5, alpha=0.7, label='Original')
        # Plot segments if they exist
        if segments and max_error != np.inf:
            for x1, x2, slope, intercept in segments:
                x_seg = np.linspace(x1, x2, 50)
                y_seg = slope * x_seg + intercept
                ax.plot(x_seg, y_seg, 'r-', linewidth=2, alpha=0.7)
                ax.plot([x1], [slope * x1 + intercept], 'ro', markersize=3)
                ax.plot([x2], [slope * x2 + intercept], 'ro', markersize=3)
        
        error_text = f'Max err: {max_error:.3f}' if max_error != np.inf else 'Fitting failed'
        ax.text(0.05, 0.05, error_text, transform=ax.transAxes, 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        ax.set_title(f'Layer {layer_idx}, φ_{{{input_index},{output_index}}}', fontsize=10)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel(f'φ', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.3)
    for i in range(total_curves, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# Lipschitz Visualization
def print_dp_table(dp_table, num_splines, max_total_segments, step=20):
    print("\nDynamic Programming Table (Error values):")
    print("-" * 80)
    header = "Spline \\ Segments"
    for j in range(0, max_total_segments + 1, step):
        header += f" | {j:5d}"
    print(header)
    print("-" * len(header))
    for i in range(num_splines + 1):
        row = f"{i:3d}"
        for j in range(0, max_total_segments + 1, step):
            if np.isinf(dp_table[i, j]):
                row += " | inf  "
            else:
                row += f" | {dp_table[i, j]:.3f}"
        print(row)
    print("-" * 80)
    print("Note: 'inf' indicates that the error is infinite (impossible allocation)")
    print("Only the first 100 columns are shown (if applicable)")

def visualize_all_splines(kan_model, optimal_allocation, actual_segments):
    splines_by_layer = {}
    for spline_key in optimal_allocation.keys():
        layer_idx, input_idx, output_idx = spline_key
        if layer_idx not in splines_by_layer:
            splines_by_layer[layer_idx] = []
        splines_by_layer[layer_idx].append((input_idx, output_idx))
    figures = []
    # Create a separate figure for each layer
    for layer_idx, splines in splines_by_layer.items():
        layer = kan_model.layers[layer_idx]
        num_splines = len(splines)
        grid_size = math.ceil(math.sqrt(num_splines))
        rows = math.ceil(num_splines / grid_size)
        cols = min(grid_size, num_splines)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        fig.suptitle(f"Layer {layer_idx}: All Splines with Optimal Segment Allocation", fontsize=16)
        if num_splines > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        for i, (input_idx, output_idx) in enumerate(splines):
            if i >= len(axes_flat):
                break
            ax = axes_flat[i]
            spline_key = (layer_idx, input_idx, output_idx)
            x_tensor, y_tensor = layer.plot_curve(input_idx, output_idx, num_pts=1000)
            x_original = x_tensor.detach().cpu().numpy()
            y_original = y_tensor.detach().cpu().numpy()
            segments = actual_segments[spline_key]
            x_simplified = []
            y_simplified = []
            for x1, x2, slope, intercept in segments:
                segment_x = np.linspace(x1, x2, 50)
                segment_y = slope * segment_x + intercept
                x_simplified.extend(segment_x)
                y_simplified.extend(segment_y)
            ax.plot(x_original, y_original, 'b-', label='Original')
            ax.plot(x_simplified, y_simplified, 'r--', label=f'{optimal_allocation[spline_key]} segments')
            for x1, x2, _, _ in segments:
                ax.axvline(x=x1, color='g', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Input {input_idx} → Output {output_idx}')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, alpha=0.3)
        for j in range(num_splines, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)
    return figures


# MILP Visualization
def visualize_comparison_results(results: Dict[str, Any], output_dir: str):
    lipschitz_successful = "error" not in results["lipschitz"]
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    labels = ['Vanilla']
    vanilla_times = [results["vanilla"]["fit_time"]]
    if lipschitz_successful:
        labels.append('Lipschitz')
        lipschitz_times = [
            results["lipschitz"]["dp_time"],
            results["lipschitz"]["weighting_time"],
            results["lipschitz"]["allocation_time"]
        ]
        lipschitz_total = sum(lipschitz_times)
        vanilla_times.append(lipschitz_total)
        plt.bar(1, lipschitz_times[0], color='#ff8c00', label='DP Computation')
        plt.bar(1, lipschitz_times[1], bottom=lipschitz_times[0], color='#e74c3c', label='Error Weighting')
        plt.bar(1, lipschitz_times[2], bottom=sum(lipschitz_times[:2]), color='#9b59b6', label='Allocation')
    plt.bar(0, vanilla_times[0], color='#3498db', label='Vanilla Fitting')
    plt.title('Segment Fitting Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels)), labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 2)
    segments = [results["vanilla"]["total_segments"]]
    if lipschitz_successful:
        segments.append(results["lipschitz"]["total_segments"])
    plt.bar(range(len(segments)), segments, color=['#3498db', '#e74c3c'][:len(segments)])
    plt.title('Total Number of Segments')
    plt.ylabel('Number of Segments')
    plt.xticks(range(len(segments)), labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 3)
    verification_labels = ['Vanilla']
    verification_times = [results["verification"]["vanilla_mip_time"]]
    if lipschitz_successful and "lipschitz_mip_time" in results["verification"]:
        verification_labels.append('Lipschitz')
        verification_times.append(results["verification"]["lipschitz_mip_time"])
    plt.bar(range(len(verification_times)), verification_times, color=['#3498db', '#e74c3c'][:len(verification_times)])
    plt.title('MIP Verification Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(verification_times)), verification_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show figure before saving
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'output.png'), dpi=300)
    plt.close()

def detailed_metrics_visualization(results: Dict[str, Any], output_dir: str):
    lipschitz_successful = "error" not in results["lipschitz"]
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    vanilla_fitting_time = results["vanilla"]["fit_time"]
    vanilla_mip_time = results["verification"]["vanilla_mip_time"]
    if "vanilla_conversion_time" in results["verification"] and results["verification"]["vanilla_conversion_time"] > 0:
        vanilla_conversion_time = results["verification"]["vanilla_conversion_time"]
    else:
        vanilla_total = results["verification"]["vanilla_total_time"]
        vanilla_conversion_time = vanilla_total - vanilla_mip_time
    vanilla_times = [
        vanilla_fitting_time,
        vanilla_conversion_time,
        vanilla_mip_time
    ]
    labels = ['Vanilla']
    plt.bar(0, vanilla_times[0], color='#3498db', label='Fitting')
    plt.bar(0, vanilla_times[1], bottom=vanilla_times[0], color='#2ecc71', label='Conversion')
    plt.bar(0, vanilla_times[2], bottom=sum(vanilla_times[:2]), color='#f39c12', label='MIP')
    if lipschitz_successful and "lipschitz_mip_time" in results["verification"]:
        labels.append('Lipschitz')
        lipschitz_fitting_time = results["lipschitz"]["total_fit_time"]
        lipschitz_mip_time = results["verification"]["lipschitz_mip_time"]
        if "lipschitz_conversion_time" in results["verification"] and results["verification"]["lipschitz_conversion_time"] > 0:
            lipschitz_conversion_time = results["verification"]["lipschitz_conversion_time"]
        else:
            lipschitz_total = results["verification"]["lipschitz_total_time"]
            lipschitz_conversion_time = lipschitz_total - lipschitz_mip_time
        lipschitz_times = [
            lipschitz_fitting_time,
            lipschitz_conversion_time,
            lipschitz_mip_time
        ]
        plt.bar(1, lipschitz_times[0], color='#3498db')
        plt.bar(1, lipschitz_times[1], bottom=lipschitz_times[0], color='#2ecc71')
        plt.bar(1, lipschitz_times[2], bottom=sum(lipschitz_times[:2]), color='#f39c12')
    plt.title('End-to-End Time Breakdown')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels)), labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 2)
    if lipschitz_successful:
        lipschitz_times = [
            results["lipschitz"]["dp_time"],
            results["lipschitz"]["weighting_time"],
            results["lipschitz"]["allocation_time"]
        ]
        labels = ['DP Comp.', 'Error Weight.', 'Allocation']
        plt.bar(range(len(lipschitz_times)), lipschitz_times, color=['#ff8c00', '#e74c3c', '#9b59b6'])
        plt.title('Lipschitz Method Time Breakdown')
    else:
        plt.title('Lipschitz Method Not Available')
        plt.text(0.5, 0.5, 'Lipschitz method failed', 
                 horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels) if lipschitz_successful else 0), labels if lipschitz_successful else [])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 3)
    vanilla_results = results["verification"]["vanilla_results"]
    output_dim = len(vanilla_results)
    if lipschitz_successful and "lipschitz_results" in results["verification"]:
        lipschitz_results = results["verification"]["lipschitz_results"]
        x = np.arange(output_dim)
        width = 0.35
        vanilla_mins = [r[0] if r[0] is not None else 0 for r in vanilla_results]
        vanilla_maxs = [r[1] if r[1] is not None else 0 for r in vanilla_results]
        vanilla_ranges = [max_val - min_val for min_val, max_val in zip(vanilla_mins, vanilla_maxs)]
        lipschitz_mins = [r[0] if r[0] is not None else 0 for r in lipschitz_results]
        lipschitz_maxs = [r[1] if r[1] is not None else 0 for r in lipschitz_results]
        lipschitz_ranges = [max_val - min_val for min_val, max_val in zip(lipschitz_mins, lipschitz_maxs)]
        plt.bar(x - width/2, vanilla_ranges, width, label='Vanilla', color='#3498db')
        plt.bar(x + width/2, lipschitz_ranges, width, label='Lipschitz', color='#e74c3c')
    else:
        vanilla_mins = [r[0] if r[0] is not None else 0 for r in vanilla_results]
        vanilla_maxs = [r[1] if r[1] is not None else 0 for r in vanilla_results]
        vanilla_ranges = [max_val - min_val for min_val, max_val in zip(vanilla_mins, vanilla_maxs)]
        plt.bar(range(output_dim), vanilla_ranges, color='#3498db', label='Vanilla')
    plt.title('Output Range Width Comparison')
    plt.ylabel('Range Width')
    plt.xlabel('Output Dimension')
    plt.xticks(range(output_dim))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show figure before saving
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.png'), dpi=300)
    plt.close()

# Experiments Shown in the Paper Visualization
def create_input_range_summary_plot(results_df, output_dir, segments_per_curve):
    if results_df.empty:
        print("No results to plot.")
        return
    
    has_tightness_metrics = "max_tightness" in results_df.columns and not all(pd.isna(results_df["max_tightness"]))
    
    plt.figure(figsize=(12, 8))
    
    if "vanilla_total_time" in results_df.columns:
        plt.plot(results_df["bound_width"], results_df["vanilla_total_time"], 
                 'o', label="Vanilla Total Time", color='#3498db', markersize=10)
        
        if "vanilla_verify_time" in results_df.columns:
            plt.plot(results_df["bound_width"], results_df["vanilla_verify_time"], 
                    's', label="Vanilla Verification Time", color='#3498db', alpha=0.6, markersize=8)
    
    if "lipschitz_total_time" in results_df.columns:
        valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_total_time"])]
        if not valid_lipschitz.empty:
            plt.plot(valid_lipschitz["bound_width"], valid_lipschitz["lipschitz_total_time"], 
                     'o', label="Lipschitz Total Time", color='#e74c3c', markersize=10)
            
            if "lipschitz_verify_time" in valid_lipschitz.columns:
                plt.plot(valid_lipschitz["bound_width"], valid_lipschitz["lipschitz_verify_time"], 
                        's', label="Lipschitz Verification Time", color='#e74c3c', alpha=0.6, markersize=8)
    
    plt.title(f'Time vs Input Bound Width (Fixed {segments_per_curve} Segments Per Curve)', fontsize=16)
    plt.xlabel('Input Bound Width', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    if any(results_df["timed_out"]):
        timeout_width = results_df[results_df["timed_out"]]["bound_width"].min()
        plt.axvline(x=timeout_width, color='red', linestyle='--')
        plt.text(timeout_width, plt.ylim()[1]*0.9, "TIMEOUT", 
                 rotation=90, verticalalignment='top', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bound_width_vs_time.png'), dpi=300)
    
    plt.title('')  
    plt.grid(False)  
    tikzplotlib.save(os.path.join(output_dir, 'bound_width_vs_time.tex'))
    
    plt.close()
    
    if has_tightness_metrics:
        plt.figure(figsize=(12, 8))
        
        if "vanilla_max_tightness" in results_df.columns:
            plt.plot(results_df["bound_width"], results_df["vanilla_max_tightness"], 
                     'o', label="Vanilla Tightness", color='#3498db', markersize=10)
        
        if "lipschitz_max_tightness" in results_df.columns:
            valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_max_tightness"])]
            if not valid_lipschitz.empty:
                plt.plot(valid_lipschitz["bound_width"], valid_lipschitz["lipschitz_max_tightness"], 
                        's', label="Lipschitz Tightness", color='#e74c3c', markersize=10)
        
        plt.title(f'Tightness vs Input Bound Width (Fixed {segments_per_curve} Segments Per Curve)', fontsize=16)
        plt.xlabel('Input Bound Width', fontsize=14)
        plt.ylabel('Maximum Tightness (%)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bound_width_vs_tightness.png'), dpi=300)
        
        plt.title('')  
        plt.grid(False)  
        tikzplotlib.save(os.path.join(output_dir, 'bound_width_vs_tightness.tex'))
        
        plt.close()
    
    plt.figure(figsize=(12, 8))
    
    if "vanilla_fit_time" in results_df.columns and "vanilla_verify_time" in results_df.columns:
        bar_width = 0.35
        index = range(len(results_df))
        
        plt.bar(np.array(index) - bar_width/2, results_df["vanilla_fit_time"], 
                width=bar_width, label="Vanilla Fitting", color='#3498db', alpha=0.7)
        
        plt.bar(np.array(index) - bar_width/2, results_df["vanilla_verify_time"], 
                width=bar_width, bottom=results_df["vanilla_fit_time"], 
                label="Vanilla Verification", color='#9b59b6', alpha=0.7)
        
        if "lipschitz_fit_time" in results_df.columns and "lipschitz_verify_time" in results_df.columns:
            valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_fit_time"])]
            if not valid_lipschitz.empty:
                lipschitz_indices = []
                for bound_width in valid_lipschitz["bound_width"]:
                    lipschitz_indices.append(results_df[results_df["bound_width"] == bound_width].index[0])
                
                plt.bar(np.array(lipschitz_indices) + bar_width/2, valid_lipschitz["lipschitz_fit_time"], 
                        width=bar_width, label="Lipschitz Fitting", color='#e74c3c', alpha=0.7)
                plt.bar(np.array(lipschitz_indices) + bar_width/2, valid_lipschitz["lipschitz_verify_time"], 
                        width=bar_width, bottom=valid_lipschitz["lipschitz_fit_time"], 
                        label="Lipschitz Verification", color='#f39c12', alpha=0.7)
        
        plt.xticks(index, [f"{w:.2f}" for w in results_df["bound_width"]])
        
        plt.title(f'Time Breakdown by Input Bound Width (Fixed {segments_per_curve} Segments Per Curve)', fontsize=16)
        plt.xlabel('Input Bound Width', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_breakdown.png'), dpi=300)
        
        plt.title('')  
        plt.grid(False)  
        tikzplotlib.save(os.path.join(output_dir, 'time_breakdown.tex'))
        
        plt.close()
    
    plt.figure(figsize=(14, len(results_df) * 0.5 + 2))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    table_data = []
    columns = ['Bound Width', 'Vanilla Time (s)', 'Vanilla Tightness (%)', 
              'Lipschitz Time (s)', 'Lipschitz Tightness (%)']
    
    for i, row in results_df.iterrows():
        bound_width = row.get('bound_width', '-')
        van_time = row.get('vanilla_total_time', '-')
        van_tight = row.get('vanilla_max_tightness', '-')
        lip_time = row.get('lipschitz_total_time', '-')
        lip_tight = row.get('lipschitz_max_tightness', '-')
        
        bound_width = f"{bound_width:.2f}" if isinstance(bound_width, (int, float)) else bound_width
        van_time = f"{van_time:.2f}" if isinstance(van_time, (int, float)) else van_time
        van_tight = f"{van_tight:.2f}" if isinstance(van_tight, (int, float)) else van_tight
        lip_time = f"{lip_time:.2f}" if isinstance(lip_time, (int, float)) else lip_time
        lip_tight = f"{lip_tight:.2f}" if isinstance(lip_tight, (int, float)) else lip_tight
        
        table_data.append([bound_width, van_time, van_tight, lip_time, lip_tight])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i, row in enumerate(results_df["timed_out"].values):
        if row:
            for j in range(len(columns)):
                cell = table[(i+1, j)]
                cell.set_facecolor('#ffcccc')
    
    plt.title(f'Input Range Experiment Summary (Fixed {segments_per_curve} Segments Per Curve)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(results_df) >= 3 and "vanilla_verify_time" in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        valid_data = results_df[~results_df["timed_out"]]
        
        if len(valid_data) >= 3:
            x = valid_data["bound_width"]
            y_vanilla = valid_data["vanilla_verify_time"]
            
            plt.scatter(x, y_vanilla, color='#3498db', s=100, label="Vanilla Verification Time")
            
            try:
                coeffs = np.polyfit(x, y_vanilla, 2)
                p = np.poly1d(coeffs)
                
                x_smooth = np.linspace(min(x), max(x), 100)
                y_smooth = p(x_smooth)
                
                plt.plot(x_smooth, y_smooth, '--', color='#3498db', 
                         label=f"Quadratic Fit: {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}")
                
                y_mean = np.mean(y_vanilla)
                ss_tot = np.sum((y_vanilla - y_mean) ** 2)
                ss_res = np.sum((y_vanilla - p(x)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                plt.text(0.05, 0.95, f"R² = {r_squared:.4f}", transform=plt.gca().transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                print("Could not fit quadratic curve to vanilla verification time")
            
            if "lipschitz_verify_time" in valid_data.columns:
                valid_lipschitz = valid_data[~pd.isna(valid_data["lipschitz_verify_time"])]
                
                if len(valid_lipschitz) >= 3:
                    x_lip = valid_lipschitz["bound_width"]
                    y_lip = valid_lipschitz["lipschitz_verify_time"]
                    
                    plt.scatter(x_lip, y_lip, color='#e74c3c', s=100, label="Lipschitz Verification Time")
                    
                    try:
                        coeffs_lip = np.polyfit(x_lip, y_lip, 2)
                        p_lip = np.poly1d(coeffs_lip)
                        
                        x_smooth_lip = np.linspace(min(x_lip), max(x_lip), 100)
                        y_smooth_lip = p_lip(x_smooth_lip)
                        
                        plt.plot(x_smooth_lip, y_smooth_lip, '--', color='#e74c3c',
                                label=f"Quadratic Fit: {coeffs_lip[0]:.4f}x² + {coeffs_lip[1]:.4f}x + {coeffs_lip[2]:.4f}")
                        
                        y_lip_mean = np.mean(y_lip)
                        ss_tot_lip = np.sum((y_lip - y_lip_mean) ** 2)
                        ss_res_lip = np.sum((y_lip - p_lip(x_lip)) ** 2)
                        r_squared_lip = 1 - (ss_res_lip / ss_tot_lip)
                        
                        plt.text(0.05, 0.85, f"Lipschitz R² = {r_squared_lip:.4f}", transform=plt.gca().transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        print("Could not fit quadratic curve to lipschitz verification time")
            
            plt.title(f'Quadratic Growth of Verification Time with Input Bound Width', fontsize=16)
            plt.xlabel('Input Bound Width', fontsize=14)
            plt.ylabel('Verification Time (seconds)', fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quadratic_time_relationship.png'), dpi=300)
            
            plt.title('')  
            plt.grid(False)  
            for text in plt.gca().texts:
                text.set_visible(False)
            plt.legend(['Vanilla Data', 'Vanilla Fit', 'Lipschitz Data', 'Lipschitz Fit'])
            tikzplotlib.save(os.path.join(output_dir, 'quadratic_time_relationship.tex'))
            
            plt.close()
            
    if "vanilla_fit_time" in results_df.columns and "vanilla_verify_time" in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        x = results_df["bound_width"]
        y1 = results_df["vanilla_fit_time"]  
        y2 = results_df["vanilla_verify_time"]  
        
        plt.fill_between(x, 0, y1, label='Fitting Time', color='#3498db', alpha=0.7)
        plt.fill_between(x, y1, y1 + y2, label='MILP Time', color='#9b59b6', alpha=0.7)
        
        plt.plot(x, y1 + y2, 'k--', label='Total Time', linewidth=2)
        
        if any(results_df["timed_out"]):
            timeout_width = results_df[results_df["timed_out"]]["bound_width"].min()
            plt.axvline(x=timeout_width, color='red', linestyle='--')
            plt.text(timeout_width, plt.ylim()[1]*0.9, "TIMEOUT", 
                    rotation=90, verticalalignment='top', color='red', fontweight='bold')
        
        plt.title(f'Dynamic Programming Approach: Time Breakdown', fontsize=16)
        plt.xlabel('Input Bound Width', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        valid_data = results_df[~results_df["timed_out"]]
        if len(valid_data) > 0:
            annotation_indices = [0]  
            if len(valid_data) > 2:
                annotation_indices.append(len(valid_data) // 2)  
            annotation_indices.append(len(valid_data) - 1)  
            
            annotation_indices = list(set(annotation_indices))
            
            for idx in annotation_indices:
                if idx < len(valid_data):
                    row = valid_data.iloc[idx]
                    total_time = row["vanilla_fit_time"] + row["vanilla_verify_time"]
                    fit_percentage = row["vanilla_fit_time"] / total_time * 100 if total_time > 0 else 0
                    milp_percentage = row["vanilla_verify_time"] / total_time * 100 if total_time > 0 else 0
                    
                    midpoint_fit = row["vanilla_fit_time"] / 2
                    midpoint_milp = row["vanilla_fit_time"] + row["vanilla_verify_time"] / 2
                    
                    plt.text(row["bound_width"], midpoint_fit, f"{fit_percentage:.1f}%", 
                             ha='center', va='center', fontweight='bold', color='white')
                    plt.text(row["bound_width"], midpoint_milp, f"{milp_percentage:.1f}%", 
                             ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dp_time_breakdown_area.png'), dpi=300)
        
        plt.title('')  
        plt.grid(False)  
        for text in plt.gca().texts:
            text.set_visible(False)  
        tikzplotlib.save(os.path.join(output_dir, 'dp_time_breakdown_area.tex'))
        
        plt.close()

def create_relative_width_ratio_graph(results: Dict[str, Any], output_dir: str):
    
    if "validation" not in results or "sampled_min" not in results["validation"]:
        print("Cannot create relative width ratio graph: validation data not available")
        return
    
    sampled_min = results["validation"]["sampled_min"]
    sampled_max = results["validation"]["sampled_max"]
    
    vanilla_results = results["verification"]["vanilla_results"]
    output_dim = len(vanilla_results)
    
    lipschitz_successful = ("error" not in results["lipschitz"] and 
                           "lipschitz_results" in results["verification"])
    
    columns = ['Output', 'Method', 'MILP Min', 'MILP Max', 'MILP Width', 
               'Sampled Min', 'Sampled Max', 'Sampled Width', 'Tightness (%)']
    df_data = []
    
    vanilla_tightness = []
    for i in range(output_dim):
        mip_min, mip_max = vanilla_results[i]
        samp_min, samp_max = sampled_min[i], sampled_max[i]
        
        mip_width = mip_max - mip_min
        sampled_width = samp_max - samp_min
        
        tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
        vanilla_tightness.append(tightness)
        
        df_data.append([
            i, 'Vanilla', mip_min, mip_max, mip_width,
            samp_min, samp_max, sampled_width, tightness
        ])
    
    lipschitz_tightness = []
    if lipschitz_successful:
        lipschitz_results = results["verification"]["lipschitz_results"]
        
        for i in range(output_dim):
            mip_min, mip_max = lipschitz_results[i]
            samp_min, samp_max = sampled_min[i], sampled_max[i]
            
            mip_width = mip_max - mip_min
            sampled_width = samp_max - samp_min
            
            tightness = (sampled_width / mip_width * 100) if mip_width > 0 else 100.0
            lipschitz_tightness.append(tightness)
            
            df_data.append([
                i, 'Lipschitz', mip_min, mip_max, mip_width,
                samp_min, samp_max, sampled_width, tightness
            ])
    
    df = pd.DataFrame(df_data, columns=columns)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(output_dim)
    width = 0.35
    
    plt.bar(x - width/2, vanilla_tightness, width, label='Vanilla', color='#3498db')
    
    if lipschitz_successful:
        plt.bar(x + width/2, lipschitz_tightness, width, label='Lipschitz', color='#e74c3c')
    
    avg_vanilla = sum(vanilla_tightness) / len(vanilla_tightness)
    plt.axhline(y=avg_vanilla, color='#3498db', linestyle='--', 
               label=f'Avg Vanilla: {avg_vanilla:.2f}%')
    
    if lipschitz_successful:
        avg_lipschitz = sum(lipschitz_tightness) / len(lipschitz_tightness)
        plt.axhline(y=avg_lipschitz, color='#e74c3c', linestyle='--',
                   label=f'Avg Lipschitz: {avg_lipschitz:.2f}%')
    
    plt.title('MILP Bound Tightness (Higher % = Tighter Bounds)')
    plt.ylabel('Tightness (%)')
    plt.xlabel('Output Dimension')
    plt.xticks(x)
    plt.ylim(0, 110)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'relative_width_ratio.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.figure(figsize=(14, output_dim * 0.5 + 2)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    if lipschitz_successful:
        table_data = []
        for i in range(output_dim):
            table_data.append([
                i,
                f"{vanilla_tightness[i]:.2f}%",
                f"{lipschitz_tightness[i]:.2f}%"
            ])
        table_data.append([
            "Average",
            f"{avg_vanilla:.2f}%",
            f"{avg_lipschitz:.2f}%"
        ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Output Dimension', 'Vanilla Tightness', 'Lipschitz Tightness'],
            loc='center',
            cellLoc='center'
        )
    else:
        table_data = []
        for i in range(output_dim):
            table_data.append([i, f"{vanilla_tightness[i]:.2f}%"])
        table_data.append(["Average", f"{avg_vanilla:.2f}%"])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Output Dimension', 'Vanilla Tightness'],
            loc='center',
            cellLoc='center'
        )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for j in range(len(table_data[0])):
        cell = table[(len(table_data), j)]
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(weight='bold')
    
    plt.title('MILP Bound Tightness - Detailed Comparison')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'relative_width_ratio_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_experiment_summary_plot(results_df, output_dir):
    if results_df.empty:
        print("No results to plot.")
        return
    
    has_tightness_metrics = "max_tightness" in results_df.columns and not all(pd.isna(results_df["max_tightness"]))
    
    if has_tightness_metrics:
        plt.figure(figsize=(12, 8))
        
        if "vanilla_total_time" in results_df.columns and "vanilla_max_tightness" in results_df.columns:
            plt.scatter(results_df["vanilla_max_tightness"], results_df["vanilla_total_time"], 
                    label="Vanilla", marker='o', s=100, color='#3498db')
            
            for i, row in results_df.iterrows():
                plt.annotate(f"{int(row['segments'])}", 
                            (row["vanilla_max_tightness"], row["vanilla_total_time"]),
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        if "lipschitz_total_time" in results_df.columns and "lipschitz_max_tightness" in results_df.columns:
            valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_max_tightness"])]
            if not valid_lipschitz.empty:
                plt.scatter(valid_lipschitz["lipschitz_max_tightness"], valid_lipschitz["lipschitz_total_time"], 
                        label="Lipschitz", marker='s', s=100, color='#e74c3c')
                
                for i, row in valid_lipschitz.iterrows():
                    plt.annotate(f"{int(row['segments'])}", 
                                (row["lipschitz_max_tightness"], row["lipschitz_total_time"]),
                                textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title('Time vs Tightness', fontsize=14)
        plt.xlabel('Maximum Tightness (%)', fontsize=12)
        plt.ylabel('Time (s)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_vs_tightness.png'), dpi=300)
        tikzplotlib.save(os.path.join(output_dir, 'time_vs_tightness.tex'))
        plt.close()
    
    plt.figure(figsize=(12, 8))
    
    if "vanilla_fit_time" in results_df.columns and "vanilla_verify_time" in results_df.columns:
        plt.bar(results_df["segments"] - 0.2, results_df["vanilla_fit_time"], 
                width=0.4, label="Vanilla Fitting", color='#3498db', alpha=0.7)
        plt.bar(results_df["segments"] - 0.2, results_df["vanilla_verify_time"], 
                width=0.4, bottom=results_df["vanilla_fit_time"], 
                label="Vanilla Verification", color='#9b59b6', alpha=0.7)
    
    if "lipschitz_fit_time" in results_df.columns and "lipschitz_verify_time" in results_df.columns:
        valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_fit_time"])]
        if not valid_lipschitz.empty:
            plt.bar(valid_lipschitz["segments"] + 0.2, valid_lipschitz["lipschitz_fit_time"], 
                    width=0.4, label="Lipschitz Fitting", color='#e74c3c', alpha=0.7)
            plt.bar(valid_lipschitz["segments"] + 0.2, valid_lipschitz["lipschitz_verify_time"], 
                    width=0.4, bottom=valid_lipschitz["lipschitz_fit_time"], 
                    label="Lipschitz Verification", color='#f39c12', alpha=0.7)
    
    plt.title('Time by Segment Count', fontsize=14)
    plt.xlabel('Segments', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=10)
    
    if any(results_df["timed_out"]):
        timeout_segments = results_df[results_df["timed_out"]]["segments"].min()
        plt.axvline(x=timeout_segments, color='red', linestyle='--', label=f"Timeout at {timeout_segments}")
        plt.text(timeout_segments, plt.ylim()[1]*0.9, "TIMEOUT", 
                 rotation=90, verticalalignment='top', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segments_vs_time.png'), dpi=300)
    tikzplotlib.save(os.path.join(output_dir, 'segments_vs_time.tex'))
    plt.close()
    
    if has_tightness_metrics:
        plt.figure(figsize=(12, 8))
        
        if "vanilla_max_tightness" in results_df.columns:
            plt.scatter(results_df["segments"], results_df["vanilla_max_tightness"], 
                     marker='o', label="Vanilla", color='#3498db', s=100)
        
        if "lipschitz_max_tightness" in results_df.columns:
            valid_lipschitz = results_df[~pd.isna(results_df["lipschitz_max_tightness"])]
            if not valid_lipschitz.empty:
                plt.scatter(valid_lipschitz["segments"], valid_lipschitz["lipschitz_max_tightness"], 
                        marker='s', label="Lipschitz", color='#e74c3c', s=100)
        
        plt.title('Tightness by Segment Count', fontsize=14)
        plt.xlabel('Segments', fontsize=12)
        plt.ylabel('Tightness (%)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'segments_vs_tightness.png'), dpi=300)
        tikzplotlib.save(os.path.join(output_dir, 'segments_vs_tightness.tex'))
        plt.close()
    
    plt.figure(figsize=(14, len(results_df) * 0.5 + 2))
    ax = plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    table_data = []
    columns = ['Segments', 'Vanilla Time', 'Vanilla Tightness', 
              'Lipschitz Time', 'Lipschitz Tightness']
    
    for i, row in results_df.iterrows():
        van_time = row.get('vanilla_total_time', '-')
        van_tight = row.get('vanilla_max_tightness', '-')
        lip_time = row.get('lipschitz_total_time', '-')
        lip_tight = row.get('lipschitz_max_tightness', '-')
        
        van_time = f"{van_time:.2f}" if isinstance(van_time, (int, float)) else van_time
        van_tight = f"{van_tight:.2f}" if isinstance(van_tight, (int, float)) else van_tight
        lip_time = f"{lip_time:.2f}" if isinstance(lip_time, (int, float)) else lip_time
        lip_tight = f"{lip_tight:.2f}" if isinstance(lip_tight, (int, float)) else lip_tight
        
        table_data.append([int(row['segments']), van_time, van_tight, lip_time, lip_tight])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i, row in enumerate(results_df["timed_out"].values):
        if row:
            for j in range(len(columns)):
                cell = table[(i+1, j)]
                cell.set_facecolor('#ffcccc')
    
    plt.title('Experiment Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    
    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write('\\begin{tabular}{ccccc}\n')
        f.write('\\hline\n')
        f.write(' & '.join(columns) + ' \\\\\n')
        f.write('\\hline\n')
        
        for row in table_data:
            f.write(' & '.join(map(str, row)) + ' \\\\\n')
        
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
    
    plt.close()