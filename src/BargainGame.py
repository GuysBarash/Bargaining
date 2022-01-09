import random
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from copy import copy
import itertools
from collections import deque

import pandas as pd
from tqdm import tqdm

IN_COLAB = 'google.colab' in sys.modules


class BargainGame:
    def __init__(self,
                 coalitions_value=dict(),
                 bargain_step=50,
                 default_coalition_value=0,
                 ):
        self.coalitions_value = coalitions_value
        self.bargain_step = bargain_step
        self.default_coalition_value = default_coalition_value
        self.players_count = None
        self.players = list()
        self.info = None
        self._update_players_count()

        self.metacoalitions = None
        self.roots = None
        self.coalitions_map = None

        self.negotiation_graph = None
        self.negotiation_graph_nontransient = None
        self.cycles = None
        self.end_points = None
        self.components = None
        self.adjacency_matrix = None

        self.compiled = False

    def get_info(self):
        if self.info is None:
            self.info = dict()
            self.info['metacoalitions'] = self.metacoalitions
            self.info['players_count'] = self.players_count
            self.info['bargain_step'] = self.bargain_step
            self.info['default_coalition_value'] = self.default_coalition_value
        return self.info

    def add_coalitions_from_dict(self, coalitions_value):
        self.coalitions_value.update(coalitions_value)
        self._update_players_count()

    def clear_all_coalitions(self):
        self.coalitions_value = dict()
        self._update_players_count()

    def get_coalitions(self):
        return self.coalitions_value

    def _update_players_count(self):
        self.players_count = len(
            list(set(itertools.chain.from_iterable([str_to_coalition_keys(t) for t in self.coalitions_value.keys()]))))
        if self.players_count == 0:
            self.players = list()
        else:
            self.players = list(range(1, self.players_count + 1))

    def compile(self):
        metacoalitions = [MetaCoalition(k, v, number_of_players=self.players_count) for k, v in
                          coalitions_value.items()]

        self.metacoalitions = {mc.sig: mc for mc in metacoalitions}
        self.calculate_roots()

        # BFS
        g = dict()
        q = []
        visited = dict()

        for root in self.roots:
            g[root.sig] = root
            q.append(root.sig)

        while len(q) > 0:
            node_sig = q.pop(0)
            node = g[node_sig]
            if node_sig in visited:
                continue
            else:
                visited[node_sig] = True

                n_nodes = node.expand()
                n_nodes = Coalition.reduce_coalition(n_nodes)
                node.set_edges([t._sig() for t in n_nodes])
                for n_node in n_nodes:
                    if n_node is None:
                        continue
                    elif n_node.sig in visited:
                        continue
                    else:
                        g[n_node.sig] = n_node
                        q.append(n_node.sig)

        self.coalition_map = g

        # Build graph
        nodes = list(g.keys())
        edges = list()
        for n_node_sig in nodes:
            edges += g[n_node_sig].get_as_edge()

        self.negotiation_graph = nx.DiGraph()
        for n_node in nodes:
            self.negotiation_graph.add_node(n_node, name=n_node)
        self.negotiation_graph.add_edges_from(edges)

        # Find end points
        self.end_points = list()
        for node_sig, node in g.items():
            e_t = node.get_as_edge()
            if e_t is None or len(e_t) == 0:
                self.end_points.append(node_sig)

        # Find cycles
        self.cycles = list(nx.algorithms.simple_cycles(self.negotiation_graph))

        # Build non transient graph
        # Plot just loops
        nodes_sig = list()
        if len(self.cycles) > 0:
            nodes_sig = list(set.union(*[set(s) for s in self.cycles]))
        edges = list()
        for n_node_sig in nodes_sig:
            e = g[n_node_sig].get_as_edge()
            e = [et for et in e if et[1] in nodes_sig]
            edges += e

        self.negotiation_graph_nontransient = nx.DiGraph()
        for n_node in nodes_sig + self.end_points:
            self.negotiation_graph_nontransient.add_node(n_node, name=n_node)
        self.negotiation_graph_nontransient.add_edges_from(edges)

        self.components = [list(qt) for qt in nx.strongly_connected_components(self.negotiation_graph_nontransient)]

        # adjacency matrix calculations
        self.adjacency_matrix = nx.adjacency_matrix(self.negotiation_graph_nontransient).todense()
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(self.adjacency_matrix, self.adjacency_matrix.sum(axis=1))
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
            self.adjacency_matrix = np.array(c)

        def steady_state_prop(p):
            dim = p.shape[0]
            q = (p - np.eye(dim))
            ones = np.ones(dim)
            q = np.c_[q, ones]
            QTQ = np.dot(q, q.T)
            bQT = np.ones(dim)
            return np.linalg.solve(QTQ, bQT)

        steady_state_matrix = steady_state_prop(self.adjacency_matrix)
        steady_state_matrix = steady_state_matrix.round(4)
        eps = 0.00001
        self.ssdf = pd.Series(index=self.negotiation_graph_nontransient.nodes, data=steady_state_matrix, dtype=float)
        self.ssdf = self.ssdf[self.ssdf > eps]

        self.compiled = True

    def calculate_roots(self):
        roots = list()
        for leader_p in self.players:
            known_coalitions_with_leader = [c for c in self.metacoalitions.values() if leader_p in c.coalition_members]
            for coalition_c in known_coalitions_with_leader:
                coalition_c_sig = coalition_c.sig
                coalition_c_value = coalition_c.value
                root = [0] * self.players_count

                # Set value of all non-leaders to the bargaining step
                for player_j in coalition_c.coalition_members:
                    if player_j != leader_p:
                        root[player_j - 1] = bargain_step

                leader_value = coalition_c_value - sum(root)
                root[leader_p - 1] = leader_value
                if leader_value > 0:
                    coalition = Coalition(root, [coalition_c], leader_p, info=self.get_info())
                    roots.append(coalition)
        self.roots = roots

    def get_nontransient_points(self):
        return self.cycles, self.end_points

    def get_negotiation_graph(self, nontransient=False):
        if nontransient:
            return self.negotiation_graph_nontransient
        else:
            return self.negotiation_graph

    def plot_bargaining_graph(self, colab_mode=False):
        if colab_mode:
            # COLORING
            color_map = []
            for node in self.negotiation_graph:
                color = 'ORANGE'
                for cycle in cycles:
                    if node in cycle:
                        color = 'RED'
                if node in end_points:
                    color = 'YELLOW'
                color_map += [color]

            # Plot all graph
            node_size = 500
            font_size = 10
            plt.figure(3, figsize=(12, 12))

            nx.draw(self.negotiation_graph, node_color=color_map, node_size=node_size, with_labels=True)
            plt.show()
        else:
            print("<< Not colab mode. Skipping print >>")

    def plot_bargaining_graph_non_transient(self, colab_mode=False):
        if colab_mode:
            # COLORING
            color_map = []
            for node in self.negotiation_graph_nontransient:
                color = 'ORANGE'
                for cycle in cycles:
                    if node in cycle:
                        color = 'RED'
                if node in end_points:
                    color = 'YELLOW'
                color_map += [color]

            # Plot all graph
            node_size = 500
            font_size = 10
            plt.figure(3, figsize=(12, 12))

            nx.draw(self.negotiation_graph_nontransient, node_color=color_map, node_size=node_size, with_labels=True)
            plt.show()
        else:
            print("<< Not colab mode. Skipping print >>")

    def random_walk_graph(self, normalize=False):
        assert self.compiled, Exception("Cannot random walk a graph before model is compiled.")

        histdf = pd.DataFrame(index=self.negotiation_graph_nontransient.nodes,
                              columns=range(1, len(self.components) + 1), data=0)
        for comp_idx, comp in enumerate(components):
            s_node = np.random.choice(comp)
            hist_comp = dict()
            hist_comp[s_node] = 1
            itrs = 10000
            for i in tqdm(range(itrs), desc=f'Random walk on component {comp_idx + 1}/{len(components)}'):
                options_edges = list(self.negotiation_graph_nontransient.out_edges(s_node))
                if len(options_edges) == 0:
                    time.sleep(0.1)
                    print(f"Component {comp_idx + 1} was broken")
                    break
                selected_edge = options_edges[np.random.randint(len(options_edges))]
                s_node = selected_edge[1]
                hist_comp[s_node] = hist_comp.get(s_node, 0) + 1
            sr = pd.Series(hist_comp)
            histdf.loc[sr.index, comp_idx + 1] = sr
        histdf = histdf[histdf.sum(axis=1) > 0]
        if normalize:
            histdf = histdf / histdf.sum(axis=0)
        return histdf

    def get_steady_state(self):
        j = 3


def coalition_keys_to_str(c):
    c_list = [c_t + 1 for c_t in c]
    r = '_'.join([str(c_int) for c_int in c_list])
    return r


def str_to_coalition_keys(s):
    r = sorted(list(set([int(t) - 1 for t in s.split('_')])))
    return r


class MetaCoalition:
    curr_idx = 0

    def __init__(self, key, value, number_of_players=3):
        self.idx = MetaCoalition.curr_idx + 1
        self.number_of_players = number_of_players
        MetaCoalition.curr_idx += 1

        key_int_l = [int(t) for t in key.split('_')]
        key_int_l.sort()
        self.coalition_members = key_int_l
        self.key = '_'.join([str(t) for t in key_int_l])
        self.sig = self.key

        self.coalition_vectors = [0] * self.number_of_players
        for p in key_int_l:
            self.coalition_vectors[p - 1] = 1

        self.possible_adversaries_vector = [0] * self.number_of_players
        self.possible_adversaries = list()
        for i in range(len(self.coalition_vectors)):
            if self.coalition_vectors[i] == 0:
                self.possible_adversaries += [i]
                self.possible_adversaries_vector[i] = 1

        self.value = value

    def __str__(self):
        return self.sig

    def __contains__(self, key):
        return key in self.coalition_members

    def check_overlap(self, mc):
        overlaps = list(set(self.coalition_members) & set(mc.coalition_members))
        return len(overlaps) > 0


class Coalition:

    @staticmethod
    def reduce_coalition(coalitions):
        if len(coalitions) == 0:
            return list()

        optimals = list()
        adversaries = list(set([c.last_adv for c in coalitions]))
        for adv_p in adversaries:
            coalitions_with_leader = [c for c in coalitions if c.last_adv == adv_p]
            optimal_coalition = max(coalitions_with_leader, key=lambda c: c.get_last_adv_value())

            # Option B: split in case of equilibrium
            optimal_value = optimal_coalition.get_last_adv_value()
            optimals_p = [c for c in coalitions_with_leader if c.get_last_adv_value() == optimal_value]

            optimals += optimals_p

        return optimals

    def __init__(self, payoffs, subcoalitions, last_adv, info):
        self.info = info
        self.last_adv = last_adv
        self.all_metacoalitions = info.get('metacoalitions', dict())
        self.subcoalitions = subcoalitions
        self.players_count = info.get('players_count', 50)
        self.bargain_step = info.get('bargain_step', 50)
        self.payoff = np.array(payoffs)
        self.players_vector = [int(t > 0) for t in self.payoff]
        self.players = [p + 1 for p in range(len(self.players_vector)) if self.players_vector[p] > 0]
        self.adversaries_vector = [int(t == 0) for t in self.payoff]
        self.adversaries = [p + 1 for p in range(len(self.adversaries_vector)) if self.adversaries_vector[p] > 0]
        self.next = list()
        self.sig = None
        self.sig = self._sig()

    def get_last_adv_value(self):
        return self.payoff[self.last_adv - 1]

    def _sig(self):
        if self.sig is None:
            self.sig = '_'.join([str(po) for po in self.payoff])
            self.sig = f'({self.last_adv})' + self.sig
        return self.sig

    def __str__(self):
        return self.sig

    def expand(self):
        self.next = list()
        returns = list()

        for adv_p in self.adversaries:
            known_metacoalitions_with_leader = [k for k, mc in self.all_metacoalitions.items() if
                                                adv_p in mc]
            for mc_key in known_metacoalitions_with_leader:
                mc = self.all_metacoalitions[mc_key]
                mc_value = mc.value
                payoffs_c = [0] * self.players_count

                # Set value of all non-leaders to the bargaining step
                for player_j in mc.coalition_members:
                    if player_j != adv_p:
                        payoffs_c[player_j - 1] = self.payoff[player_j - 1] + bargain_step

                adv_value = mc_value - sum(payoffs_c)
                payoffs_c[adv_p - 1] = adv_value
                if adv_value > 0:
                    current_mcs = self.subcoalitions
                    remain_mcs = [mc_t for mc_t in current_mcs if not mc.check_overlap(mc_t)]
                    for r_mc in remain_mcs:
                        for old_player in r_mc.coalition_members:
                            payoffs_c[old_player - 1] = self.payoff[old_player - 1]
                    n_coalition = Coalition(payoffs_c, [mc] + remain_mcs, last_adv=adv_p, info=self.info)
                    self.next.append(n_coalition.sig)
                    returns.append(n_coalition)
                else:
                    # leader <= 0, coalition not relevant
                    pass
        return returns

    def set_edges(self, edges):
        self.next = copy(edges)

    def get_as_edge(self):
        r = [(self.sig, t_next) for t_next in self.next]
        return r


if __name__ == '__main__':

    coalitions_value = dict()
    coalitions_value['1_5'] = 1000
    coalitions_value['1_2'] = 1000
    coalitions_value['2_3'] = 1000
    coalitions_value['3_4'] = 1000
    coalitions_value['4_5'] = 1000
    bargain_step = 10
    default_coalition_value = 0

    game = BargainGame(bargain_step=bargain_step)
    game.add_coalitions_from_dict(coalitions_value)
    print(f"Players in game: {game.players_count}")

    game.compile()
    print(f"Roots detected: {len(game.roots)}")

    cycles, end_points = game.get_nontransient_points()
    print(f"Cycles: {len(cycles)}")
    print(f"DeadEnds: {len(end_points)}")
    possible_endings = len(cycles) + len(end_points)
    print(f"Possible endings: {possible_endings}")

    # Plot entire barganing process
    game.plot_bargaining_graph()

    # Print components
    components = game.components
    for idx, comp in enumerate(components):
        print(f'[{idx + 1}/{len(components)}]\t')
        for c in components[0]:
            print(c)

    # Plot negotiation final form
    game.plot_bargaining_graph_non_transient()

    histdf = game.random_walk_graph()



