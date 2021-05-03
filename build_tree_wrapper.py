# stdlib
import glob
import itertools
import pickle
import multiprocessing
import sys

# 3rd party
import numpy as np
import gudhi
import gudhi.wasserstein

# ours
import build_tree



def bottle_dist(idx):
    '''
    '''
    i, j, homol = idx
    return (i, j, gudhi.bottleneck_distance(dgrms[i][homol], dgrms[j][homol]))

def wass_dist(idx):
    '''
    '''
    i, j, homol = idx
    return (i, j, gudhi.wasserstein.wasserstein_distance(dgrms[i][homol], dgrms[j][homol]))


def main():
    '''
    '''

    assert len(sys.argv) == 3, "Run as, e.g. : python3 build_tree_wrapper.py bottleneck single"
    # params
    if sys.argv[1] == "bottleneck":
        dist_func = bottle_dist
    elif sys.argv[1] == "wasserstein":
        dist_func = wass_dist
    else:
        print("Specify bottleneck or wasserstein as distance metric.")
        return

    if sys.argv[2] == "single":
        linkage = "single"
    elif sys.argv[2] == "complete":
        linkage = "complete"
    else:
        print("Specify linkage, single or complete.")
        return

    ncores = 16
    print("Using {} cores for computing distance metric.".format(ncores))

    global dgrms
    dgrms = []
    langs = []
    for f in sorted(glob.glob("data/wiki/indian/*reservoir*.saved.dat")):
        print(f)
        with open(f, "rb") as IN:
            rips = pickle.load(IN)
            lang = f.split("/")[-1].split(".")[1]
            dgrms.append(rips["dgms"])
            langs.append(lang.capitalize().replace("punjabi", " Punjabi"))


    to_print = list(langs)
    to_print.append('  ')

    ## H_1
    homol = 1
    pairwise_indices = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            pairwise_indices.append((i, j, homol))

    p = multiprocessing.Pool(ncores)
    pairwise_dist = np.zeros((len(langs), len(langs)), dtype = np.float)
    for res in p.imap(dist_func, pairwise_indices):
        i, j, dist = res
        pairwise_dist[i, j] = dist
        pairwise_dist[j, i] = dist
    p.close()
    p.join()

    q1 = build_tree.hclustering_pass_dmatrix(dgrms, pairwise_dist, linkage)
    print("H_1: ")
    print(q1.display(to_print))
    print("\n\n")
    build_tree.make_latex_tree(q1, to_print)

    print("\n\n---------------------------------------------------------------------------------------------------\n\n")

    ## H_0
    print("H_0: ")
    homol = 0
    pairwise_indices = []
    for i, lang1 in enumerate(langs):
        for j in range(i + 1, len(langs)):
            pairwise_indices.append((i, j, homol))

    p = multiprocessing.Pool(14)
    pairwise_dist = np.zeros((len(langs), len(langs)), dtype = np.float)
    for res in p.imap(dist_func, pairwise_indices):
        i, j, dist = res
        pairwise_dist[i, j] = dist
        pairwise_dist[j, i] = dist
    p.close()
    p.join()
    q2 = build_tree.hclustering_pass_dmatrix(dgrms, pairwise_dist, linkage)
    print(q2.display(to_print))
    print("\n\n")
    build_tree.make_latex_tree(q2, to_print)


if __name__ == '__main__':
    main()

