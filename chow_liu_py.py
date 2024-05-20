import collections
import itertools

import numpy as np


__all__ = ["chow_liu"]


def chow_liu(X, root=None):

     # Compute the mutual information between each pair of variables
    marginals = {v: X[v].value_counts(normalize=True) for v in X.columns} #prawdopodobienstwa brzegowe
    edge = collections.namedtuple("edge", ["u", "v", "mi"])
    '''
    mis = (
        edge(u, v, mutual_info(puv=X.groupby([u, v]).size() / len(X), pu=marginals[u], pv=marginals[v]))
        for u, v in itertools.combinations(sorted(X.columns), 2)
    )
    mis_list = list(mis)
    edges = ((e.u, e.v) for e in sorted(mis_list, key=lambda e: e.mi, reverse=True))
    '''
    mutual_info_values = {}
    for u, v in itertools.combinations(sorted(X.columns), 2):
        puv = X.groupby([u, v]).size() / len(X) #prawdopodobienstwo wspolne
        pu = marginals[u]
        pv = marginals[v]
        '''
        a = puv.index.levels
        b = pu.index
        c = pv.index
        '''
        mutual_info_values[(u, v)] = mutual_info(puv, pu, pv)

    #tworzenie listy krotek (u, v, mi)
    mis = []
    for key, value in mutual_info_values.items():
        u, v = key
        mis.append(edge(u, v, value))

    edges = ((e.u, e.v) for e in sorted(mis, key=lambda e: e.mi, reverse=True))
    e_see = list(edges)

    # Extract the maximum spanning tree
    neighbors = kruskal(vertices=X.columns, edges=e_see)

    if root is None:
        root = X.columns[0]

    return list(orient_tree(neighbors, root, visited=set()))


def mutual_info(puv, pu, pv):
    mi = 0.0
    for (u_val, v_val), p_uv in puv.items():
        p_u = pu[u_val]
        p_v = pv[v_val]
        mi += p_uv * np.log(p_uv / (p_u * p_v))
    return mi

class DisjointSet:

    def __init__(self, *values):
        self.parents = {x: x for x in values}
        self.sizes = {x: 1 for x in values}

    def find(self, x):
        while self.parents[x] != x:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]
        return x

    def union(self, x, y):
        if self.sizes[x] < self.sizes[y]:
            x, y = y, x
        self.parents[y] = x
        self.sizes[x] += self.sizes[y]


def kruskal(vertices, edges):
    ds = DisjointSet(*vertices)
    neighbors = collections.defaultdict(set)

    for u, v in edges:
        if ds.find(u) != ds.find(v):
            neighbors[u].add(v)
            neighbors[v].add(u)
            ds.union(ds.find(u), ds.find(v))

        if len(neighbors) == len(vertices):
            break

    return neighbors


def orient_tree(neighbors, root, visited):
    #Zwraca pary zależności na podstawie roota oraz sąsiadow z algorytmu Kruskala
    for neighbor in neighbors[root] - visited:
        yield root, neighbor
        yield from orient_tree(neighbors, root=neighbor, visited={root})