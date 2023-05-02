"""
Functions to generate plots for the paper.
"""
import os
import re
import math
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 50)

from graph_states import Graph
from run_gs_bmc import search_depth


xlabels = {'nqubits' : 'qubits',
           'edge_prob' : '$p$'}
leg_names = {'z3' : 'z3',
             'glucose4' : 'glu4'}


parser = argparse.ArgumentParser()
parser.add_argument('folder')

formats = ['png', 'pdf'] # file formats for plots
timeout_time = 1800 # visualize values which timed-out as this time


def check_timeout(bench_prefix: str, nsteps: int, nqubits: int, reachable: str):
    """
    Returns True if binary serach timed-out before reaching max_depth.
    """
    if reachable == True:
        return False
    source = Graph.from_cnf(bench_prefix + "_s.cnf")
    target = Graph.from_cnf(bench_prefix + "_t.cnf")
    max_depth = search_depth(source, target)
    is_power_of_2 = (nsteps & (nsteps - 1) == 0)
    return nsteps < max_depth and is_power_of_2


def get_last_runs(df: pd.DataFrame):
    """
    Get first satisfying run for each benchmark if it exists. Otherwise get
    last unsat unsat run.
    """
    # 1. split into reachable / unreachable
    reach_true  = df.loc[df['reachable'] == True]
    reach_false = df.loc[df['reachable'] == False]

    # 2. get longest solve time from unreachable
    idx = reach_false.groupby(['name', 'solver'])['solve_time'].transform(max) == reach_false['solve_time']
    reach_false = reach_false[idx]

    # 3. get shortest solve time from reachable
    idx = reach_true.groupby(['name', 'solver'])['solve_time'].transform(min) == reach_true['solve_time']
    reach_true = reach_true[idx]

    # 4. join reachable / unreachable
    merged = pd.merge(reach_true, reach_false, 
                      how='outer',
                      on=['name', 'solver', 'nqubits'],
                      suffixes=('_t', '_f')).astype({'reachable_t' : bool, 
                                                     'reachable_f' : bool})
    merged['reachable'] = ~merged['solve_time_t'].isnull()
    merged['solve_time'] = merged[['solve_time_t', 'solve_time_f']].max(axis=1)
    merged['enc_time'] = merged[['enc_time_t', 'enc_time_f']].max(axis=1)
    merged['nsteps'] = merged[['nsteps_t', 'nsteps_f']].min(axis=1).astype(int)

    merged = merged[['name','nqubits','solver','reachable','solve_time','nsteps']]

    return merged


def process_bmc_data(folder: str):
    """
    Get the BMC data from the given folder.
    """
    df = pd.read_csv(os.path.join(folder, 'bmc_results.csv'), skipinitialspace=True)
    df = df.rename(columns=lambda x: x.strip())

    # get first sat or last unsat run for each benchmark
    df = get_last_runs(df)

    df['edge_prob'] = 0 # add edge probs to info (parsed from )

    # Get meta info + Mark instances which timed-out
    for idx, row in df.iterrows():
        with open(row['name'][:-6] + '_info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)
            timed_out = check_timeout(row['name'][:-6], row['nsteps'], info['nqubits'], row['reachable'])
            if timed_out:
                df.at[idx, 'solve_time'] = timeout_time
            # parse edge prob if possible
            if info['source'].startswith('ER'):
                split = re.split(';|\(|\)', info['source']) # good ol' regular expressions
                df.at[idx, 'edge_prob'] = float(split[2])
            elif info['source'].startswith('RABBIE_RAND'):
                split = re.split('\(|\)', info['source'])
                df.at[idx, 'edge_prob'] = float(split[1])

    return df


def _plot_diagonal_lines(ax, min_val, max_val, at=[0.1, 10]):
    """
    Add diagonal lines to ax
    """
    
    # bit of margin for vizualization
    #ax.set_xlim([min_val-0.15*min_val, max_val+0.15*max_val])
    #ax.set_ylim([min_val-0.15*min_val, max_val+0.15*max_val])

    # diagonal lines
    ax.plot([min_val, max_val], [min_val, max_val], ls="--", c="gray")
    for k in at:
        ax.plot([min_val, max_val], [min_val*k, max_val*k], ls=":", c="#767676")

    return ax


def plot_bmc_scatter(df: pd.DataFrame, xval, args):
    """
    Plot solve time against # qubits for given data.
    """
    coziness = 2.5 # lower is more cozy
    relwidth = 1.4 # relative width
    fig, ax = plt.subplots(figsize=(coziness*relwidth, coziness))

    # Plot sat and unsat separately
    for sat, marker in zip([True, False], ['*', 'o']):
        sub1 = df.loc[df['reachable'] == sat]

        # Plot solvers separately
        for solver, col in zip(['z3', 'glucose4'], ['royalblue', 'darkorange']):
            sub2 = sub1.loc[sub1['solver'] == solver]

            xs = np.array(sub2[xval])
            ys = np.array(sub2['solve_time'])

            if len(xs) == 0:
                continue

            # NOTE: col must be at least four characters because of np use
            fc_cols = np.array([col for _ in range(len(xs))])
            fc_cols[ys == timeout_time] = 'none'
            unreach_lab = 'timeout' if ys[0] == timeout_time else 'unsat'
            label = f"{'sat' if sat else unreach_lab}, {leg_names[solver]}"
            ax.scatter(xs, ys, facecolors=fc_cols, edgecolors=col, marker=marker, label=label)

    # Axis labels, etc.
    y_ticks = [300*t for t in range(7)]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabels[xval])
    ax.set_ylim(-timeout_time*0.05, timeout_time*1.05)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([])
    if xval == 'edge_prob':
        ax.set_xticks([.5, .6, .7, .8, .9])

    # Save bare axes
    Path(args.folder).mkdir(parents=True, exist_ok=True)
    for _format in formats:
        fig.savefig(os.path.join(args.folder, f'bmc_scatter_{xval}_bare.{_format}'),
                    bbox_inches='tight')

    ax.set_yticklabels(y_ticks)
    ax.set_ylabel('time (s)')

    # Save figure w/ axis labels but no legend
    Path(args.folder).mkdir(parents=True, exist_ok=True)
    for _format in formats:
        fig.savefig(os.path.join(args.folder, f'bmc_scatter_{xval}_no_leg.{_format}'),
                    bbox_inches='tight')

    ax.legend()

    # Save figure
    Path(args.folder).mkdir(parents=True, exist_ok=True)
    for _format in formats:
        fig.savefig(os.path.join(args.folder, f'bmc_scatter_{xval}.{_format}'),
                    bbox_inches='tight')


def plot_bmc_solver_vs(solver1, solver2, df: pd.DataFrame, args):
    """
    Plot solver1 vs solver2 on bmc data.
    """
    coziness = 3.4 # lower is more cozy
    relwidth = 1.4 # relative width
    fig, ax = plt.subplots(figsize=(coziness*relwidth, coziness))

    s1_data = df.loc[df['solver'] == solver1].set_index('name')
    s2_data = df.loc[df['solver'] == solver2].set_index('name')
    
    joined = s1_data.join(s2_data, lsuffix='_1', rsuffix='_2')

    # Plot sat and unsat separately
    for sat, marker in zip([True, False], ['*', 'o']):
        if sat:
            sub = joined.loc[(joined['reachable_1'] == True) | (joined['reachable_2'] == True)]
        else:
            sub = joined.loc[(joined['reachable_1'] == False) & (joined['reachable_2'] == False)]

        sub = sub.sort_values('nqubits_1')
        xs = sub['solve_time_1']
        ys = sub['solve_time_2']

        if len(xs) == 0:
                continue

        col = 'royalblue' # NOTE: must be at least four chars because of np use
        fc_cols = np.array([col for _ in range(len(xs))])
        fc_cols[(ys == timeout_time) | (xs == timeout_time)] = 'none'
        label = 'sat' if sat else 'unsat'
        ax.scatter(xs, ys, facecolors=fc_cols, edgecolors=col, marker=marker, label=label)

    # Axis labels, etc.
    ax = _plot_diagonal_lines(ax, 0, timeout_time, at=[])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([300*t for t in range(7)])
    ax.set_yticks([300*t for t in range(7)])
    ax.set_xlabel(f"{solver1} time (s)")
    ax.set_ylabel(f"{solver2} time (s)")
    
    ax.legend()
    plt.tight_layout()

    # Save figure
    Path(args.folder).mkdir(parents=True, exist_ok=True)
    for _format in formats:
        fig.savefig(os.path.join(args.folder, f'bmc_solvers.{_format}'))


def main():
    """
    Generate plots from given folder.
    """
    args = parser.parse_args()
    df = process_bmc_data(args.folder)
    plot_bmc_scatter(df, 'nqubits', args)
    plot_bmc_solver_vs('z3', 'glucose4', df, args)
    plot_bmc_scatter(df, 'edge_prob', args)


if __name__ == '__main__':
    main()
