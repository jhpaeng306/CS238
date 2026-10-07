import sys
import numpy as np
from math import lgamma
import networkx
import matplotlib.pyplot as plt
import time
import pandas as pd

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def plot_graph(dag, idx2names, filename):
    networkx.draw(dag, labels=idx2names, with_labels=True)
    plt.savefig(filename)
    plt.close()

def bayesian_score(data, parents):
    values = (data.to_numpy()).astype(np.int64)
    cardinality = np.max(values, axis=0)
    column_idx = {name: idx for idx, name in enumerate(data.columns)}

    score = 0.0

    for i, name in enumerate(data.columns):
        r_i = int(cardinality[i])
        parent_idx = [column_idx[p] for p in parents.get(name, [])]

        if len(parent_idx) == 0:
            groups = np.zeros(len(values), dtype=np.int64)
        else:
            _, groups = np.unique(values[:, parent_idx], axis=0, return_inverse=True)

        counts = np.zeros((int(np.max(groups) + 1), r_i), dtype=np.int64)
        np.add.at(counts, (groups, values[:, i] - 1), 1) # counts[groups, values[:,i] - 1] += 1

        for row in counts:
            total = int(np.sum(row))
            score += lgamma(r_i) - lgamma(r_i + total)
            score += sum(lgamma(int(count) + 1) for count in row)
    return score

def compute(infile, outfile):
    data = pd.read_csv(infile)
    max_parents = 3

    names = list(data.columns)
    idx2names = dict(enumerate(names))
    name2idx = {name: idx for idx, name in enumerate(names)}

    dag = networkx.DiGraph()
    dag.add_nodes_from(range(len(names)))
    parents = {name: [] for name in names}

    t0 = time.perf_counter()
    curr_score = bayesian_score(data, parents)

    for i, child in enumerate(names):
        while len(parents[child]) < max_parents:
            best_perent = None
            best_score = curr_score

            for p in names[:i]:
                if p in parents[child]:
                    continue
                parents[child].append(p)
                trial_score = bayesian_score(data, parents)
                parents[child].pop()

                if trial_score > best_score + 1e-8:
                    best_perent = p
                    best_score = trial_score

            if best_perent is None:
                break

            parents[child].append(best_perent)
            dag.add_edge(name2idx[best_perent], name2idx[child])
            curr_score = best_score

    elapsed = time.perf_counter() - t0
    write_gph(dag, idx2names, outfile)
    plot_graph(dag, idx2names, outfile.replace(".gph", ".png"))
    print(f"score = {curr_score:.2f}, time = {elapsed:.2f} s")

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()
