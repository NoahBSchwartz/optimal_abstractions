"""Regenerate all paper figures + timing table from logs/graph_data_*.json.

Run from the notebooks/ directory (or anywhere; paths are resolved relative to
this file). Produces, in ../plots/:
  plot_input_vs_output_width.pdf         (main benchmarks, Fig. "iw_width")
  plot_segments_vs_output_width.pdf      (main benchmarks, appendix Fig. "seg_width")
  plot_timing_summary.pdf                (main benchmarks, Fig. "timing")
  plot_input_vs_output_width_small.pdf   (appendix benchmarks)
  plot_segments_vs_output_width_small.pdf
  plot_timing_summary_small.pdf
  timing_table.tex

Differences from the original notebook (matching the fable_fixes main.py):
  - van_time is now the vanilla MILP time only (FIX 2), so the timing panels and
    the annotated multipliers are true MILP-stage comparisons.
  - The input-width figure shows BOTH the delta-based Optimized allocation (as
    described in the paper) and the new equal-budget Optimized allocation that
    matches Vanilla k=10's total segment budget (FIX 5).
"""

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")
PLOTS = os.path.join(HERE, "..", "plots")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 9
if shutil.which("latex") is not None:
    # usetex needs a full TeX install (type1cm etc.); verify it can actually
    # render before enabling, otherwise fall back to mathtext.
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
        _probe = plt.figure(figsize=(1, 1))
        _probe.text(0.5, 0.5, r'$k$')
        _probe.canvas.draw()
        plt.close(_probe)
    except Exception:
        plt.rcParams['text.usetex'] = False
        plt.close('all')

OPT_COLOR = 'green'
EQ_COLOR  = 'darkorange'
VAN_COLOR = 'blue'
DP_COLOR  = 'gray'

MAIN_MODELS = {
    "functionxy": "graph_data_functionxy.json",
    "functionexp4": "graph_data_functionexp4.json",
    "heatPDE": "graph_data_pinnheat.json",
    "functionexp100": "graph_data_functionexp100.json",
}
MAIN_TAGS = ["Small", "Medium", "Medium-2", "Large"]

SMALL_MODELS = {
    "functionbessel": "graph_data_functionbessel.json",
    "funcellipeinc": "graph_data_funcellipeinc.json",
    "funcellipkinc": "graph_data_funcellipkinc.json",
    "functionlegendre": "graph_data_functionlegendre.json",
    "funcsphharm": "graph_data_funcsphharm.json",
}


def load(model_files):
    data, seg_counts, input_widths = {}, None, None
    for name, fname in model_files.items():
        path = os.path.join(LOGS, fname)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path) as f:
            payload = json.load(f)
        seg_counts = payload["SEGMENT_COUNTS"]
        input_widths = payload["INPUT_WIDTHS"]
        key = list(payload["graph_data"].keys())[0]
        data[name] = payload["graph_data"][key]
    return data, seg_counts, input_widths


def fig_input_width(data, models, tags, outname):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(15 / 4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, m) in enumerate(zip(axes, models)):
        d = data[m]
        iw_opt, ow_opt, _ = zip(*d["opt_input_sweep"])
        iw_van, ow_van, _ = zip(*d["van_input_sweep"])
        nd = d.get("delta_alloc_segments")
        ne = d.get("eq_alloc_segments")
        nv = d.get("van_sweep_segments")
        lbl_opt = f"Optimized $\\delta$ ({nd} segs)" if nd else "Optimized ($\\delta$)"
        lbl_van = f"Vanilla k=10 ({nv} segs)" if nv else "Vanilla (DP)"
        ax.plot(iw_opt, ow_opt, "d-", color=OPT_COLOR, lw=2.5, label=lbl_opt)
        if "opt_input_sweep_eqbudget" in d:
            iw_eq, ow_eq, _ = zip(*d["opt_input_sweep_eqbudget"])
            lbl_eq = f"Optimized eq-budget ({ne} segs)" if ne else "Optimized (eq budget)"
            ax.plot(iw_eq, ow_eq, "s-.", color=EQ_COLOR, lw=2, alpha=0.9, label=lbl_eq)
        ax.plot(iw_van, ow_van, "x--", color=VAN_COLOR, lw=2, alpha=0.7, markersize=8, label=lbl_van)
        for x, y in zip(iw_opt, ow_opt):
            if y is not None:
                ax.annotate(f"{x:g}", (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='black')
        ax.set(xlabel="Input Bound Width (log scale)", title=f"{tags[i]} -- {m} ({d['n_params']:,} params)", xscale='log')
        ax.set_ylabel("Output Bound Width" if i == 0 else "")
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS, outname)
    plt.savefig(path, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_segments(data, models, tags, seg_counts, outname):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(15 / 4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, m) in enumerate(zip(axes, models)):
        d = data[m]
        opt_widths = [w for (_, w) in d["opt_pareto"]]
        van_widths = [w for (_, w) in d["van_pareto"]]
        ax.plot(seg_counts, opt_widths, "d-", color=OPT_COLOR, lw=2.5, label="Optimized")
        ax.plot(seg_counts, van_widths, "x--", color=VAN_COLOR, lw=2, alpha=0.7, markersize=8, label="Vanilla (DP)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set(xlabel="Segments per Spline (k)", title=f"{tags[i]} -- {m} ({d['n_params']:,} params)")
        ax.set_ylabel("Output Bound Width" if i == 0 else "")
    plt.tight_layout()
    path = os.path.join(PLOTS, outname)
    plt.savefig(path, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_timing(data, models, tags, seg_counts, outname):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(15 / 4 * n, 4), sharex=True)
    if n == 1:
        axes = [axes]
    for col, m in enumerate(models):
        d = data[m]
        opt_total = [ilp + milp for ilp, milp in zip(d['ilp_time'], d['milp_time'])]
        van_milp = d['van_time']  # MILP-only after FIX 2
        # MILP-stage ratio: vanilla MILP / optimized MILP (the multiplier the
        # paper's caption describes).
        milp_ratio = [vt / ot if ot > 0 else float('nan') for vt, ot in zip(van_milp, d['milp_time'])]

        ax = axes[col]
        ax.fill_between(seg_counts, opt_total, alpha=0.25, color=OPT_COLOR, zorder=1)
        ax.fill_between(seg_counts, van_milp, alpha=0.18, color=VAN_COLOR, zorder=1)
        ax.plot(seg_counts, opt_total, 'o-', color=OPT_COLOR, lw=2, label='Optimized (Alloc+MILP)', zorder=3)
        ax.plot(seg_counts, van_milp, 's--', color=VAN_COLOR, lw=1.5, label='Vanilla (MILP)', alpha=0.9, zorder=3)
        ax.axhline(d['dp_time'], color=DP_COLOR, linestyle=':', lw=1.5, label=f'DP abstraction ({d["dp_time"]:.2f}s, one-time)')

        for k, ot, s in zip(seg_counts, opt_total, milp_ratio):
            if s == s:
                ax.annotate(f'{s:.1f}×', xy=(k, ot), xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, color=OPT_COLOR, fontweight='bold')

        ax.set(xlabel="Segments per Spline ($k$)", title=f'{tags[col]} -- {m} ({d["n_params"]:,} params)')
        ax.set_ylabel("Time (s)" if col == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    path = os.path.join(PLOTS, outname)
    plt.savefig(path, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f"  wrote {path}")


def gen_latex_table(data, models, tags, seg_counts):
    n_k = len(seg_counts)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Verification timing breakdown across model sizes. "
                 r"The DP abstraction is computed once per model and shared by both methods; "
                 r"allocation (Alloc) and MILP verification are per query. "
                 r"Van MILP is the vanilla baseline's MILP solve time (excluding the shared abstraction). "
                 r"Speedup is Van MILP / Opt MILP.}")
    lines.append(r"\label{tab:timing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Size & $k$ & Alloc (s) & Opt MILP (s) & Opt Total (s) & Van MILP (s) & Speedup \\")
    lines.append(r"\midrule")

    for idx, m in enumerate(models):
        d = data[m]
        npar = d['n_params']
        dp_t = d['dp_time']
        model_cell = rf"\texttt{{{m}}} ({npar:,}p) [DP={dp_t:.2f}s]"

        for row_idx, (k, ilp_t, milp_t, van_t) in enumerate(zip(
                seg_counts, d['ilp_time'], d['milp_time'], d['van_time'])):
            opt_total = ilp_t + milp_t
            speedup = van_t / milp_t if milp_t > 1e-6 else float('nan')
            mc = rf"\multirow{{{n_k}}}{{*}}{{{model_cell}}}" if row_idx == 0 else ""
            sz = rf"\multirow{{{n_k}}}{{*}}{{{tags[idx]}}}" if row_idx == 0 else ""
            lines.append(
                f"  {mc} & {sz} & {k} "
                f"& {ilp_t:.3f} & {milp_t:.3f} "
                f"& {opt_total:.3f} & {van_t:.3f} "
                f"& {speedup:.2f}x \\\\"
            )
        if idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    os.makedirs(PLOTS, exist_ok=True)

    print("Main benchmarks:")
    data, seg_counts, input_widths = load(MAIN_MODELS)
    if data:
        models = sorted(data.keys(), key=lambda m: data[m]["n_params"])
        tags = MAIN_TAGS[:len(models)]
        fig_input_width(data, models, tags, "plot_input_vs_output_width.pdf")
        fig_segments(data, models, tags, seg_counts, "plot_segments_vs_output_width.pdf")
        fig_timing(data, models, tags, seg_counts, "plot_timing_summary.pdf")
        table = gen_latex_table(data, models, tags, seg_counts)
        with open(os.path.join(PLOTS, "timing_table.tex"), "w") as f:
            f.write(table)
        print(f"  wrote {os.path.join(PLOTS, 'timing_table.tex')}")

    print("Appendix benchmarks:")
    data_s, seg_counts_s, _ = load(SMALL_MODELS)
    if data_s:
        models_s = sorted(data_s.keys(), key=lambda m: data_s[m]["n_params"])
        tags_s = [f"Appendix-{i+1}" for i in range(len(models_s))]
        fig_input_width(data_s, models_s, tags_s, "plot_input_vs_output_width_small.pdf")
        fig_segments(data_s, models_s, tags_s, seg_counts_s, "plot_segments_vs_output_width_small.pdf")
        fig_timing(data_s, models_s, tags_s, seg_counts_s, "plot_timing_summary_small.pdf")


if __name__ == "__main__":
    main()
