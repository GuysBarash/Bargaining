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
from tabulate import tabulate
from itertools import chain, combinations

IN_COLAB = 'google.colab' in sys.modules


class MetaCoalition:
    curr_idx = 0

    def __init__(self, key, value, info, number_of_players=3):
        self.info = info
        self.idx = MetaCoalition.curr_idx + 1
        self.number_of_players = number_of_players
        MetaCoalition.curr_idx += 1

        self.bargain_step = info.get('bargain_step', -1)
        self.default_coalition_value = info.get('default_coalition_value', 0)

        key_int_l = [int(t) for t in key.split('_')]
        key_int_l.sort()
        self.coalition_members = key_int_l
        self.number_of_members = len(self.coalition_members)
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

    def maximal_payoff(self):
        r = np.zeros((self.number_of_members, self.number_of_players))
        r = pd.DataFrame(index=self.coalition_members, columns=range(1, self.number_of_players + 1), data=r)
        r.loc[:, self.coalition_members] = self.bargain_step
        for roller in self.coalition_members:
            v = self.value - r.loc[roller].sum()
            r.loc[roller, roller] += v

        return r


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

    def __init__(self, payoffs, leader, info):
        self.info = info
        self.leader = leader
        self.all_metacoalitions = info.get('metacoalitions', dict())
        self.players_count = info.get('players_count', 50)
        self.bargain_step = info.get('bargain_step', 50)
        self.payoff = np.array(payoffs)
        self.players_vector = [int(t > 0) for t in self.payoff]
        self.players = [p + 1 for p in range(len(self.players_vector)) if self.players_vector[p] > 0]
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
        self.info = dict()
        self._update()

        self.metacoalitions = None
        self.powerIndex = None

        # Additional information
        self.two_step_bargaining = None

    def calculate_shapley(self):
        binary_coalitions = np.array(list(map(list, itertools.product([0, 1], repeat=len(self.players)))))

        coalitions = pd.DataFrame(index=range(binary_coalitions.shape[0]),
                                  columns=self.players,
                                  data=binary_coalitions,
                                  )
        coalitions['sig'] = ''
        for p in self.players:
            coalitions.loc[coalitions[p].eq(1), 'sig'] += f'_{p}'
        coalitions['sig'] = coalitions['sig'].str.replace('^_', '')
        coalitions = coalitions.set_index('sig')

        coalitions_value = coalitions.copy()
        coalitions_value['value'] = -1

        # MAX COALITION
        q = [t for t in self.coalitions_value.items()]
        q.sort(key=lambda t: t[1])
        for k, v in q:
            plyrs = [int(t) for t in k.split('_')]
            idx_1 = coalitions_value['value'] < v
            for plyr in plyrs:
                idx_1 &= coalitions_value[plyr].eq(1)
            coalitions_value.loc[idx_1, 'value'] = v

        coalitions_value.loc[coalitions_value['value'].eq(-1), 'value'] = self.default_coalition_value

        # Calculate shapley
        shapley_values = pd.Series(index=self.players)
        banzhf_values = pd.Series(index=self.players)
        for c_player in self.players:
            other_parties = [cp for cp in self.players if cp != c_player]
            withdf = coalitions_value[coalitions_value[c_player].eq(1)]
            withoutdf = coalitions_value[coalitions_value[c_player].eq(0)]
            reordered = pd.concat([withdf, withoutdf])
            reordered = reordered.sort_values(by=other_parties)
            reordered['value_shift'] = reordered['value'].shift(-1).fillna(0)

            reordered['gain'] = (reordered['value'] - reordered['value_shift']).fillna(0)

            reordered = reordered.loc[coalitions_value[c_player].eq(1)]

            n = len(self.players)
            bnzf_coef = 1.0 / (np.power(2, n - 1))  # 1/(2^(n-1))
            reordered['N'] = len(self.players)
            reordered['S'] = reordered[other_parties].sum(axis=1)

            n_fac = np.math.factorial(len(self.players))
            s_fac = reordered['S'].apply(np.math.factorial)
            comp_s_fac = (reordered['N'] - reordered['S'] - 1).apply(np.math.factorial)
            gains = reordered['gain']

            shap_gain = ((s_fac * comp_s_fac).astype(float) / float(n_fac)) * gains
            shap_val = shap_gain.sum()

            banzhf_gain = gains.sum()
            banzhf_val = bnzf_coef * banzhf_gain

            shapley_values[c_player] = shap_val
            banzhf_values[c_player] = banzhf_val

        self.powerIndex['Shapley'] = shapley_values
        self.powerIndex['Banzhf'] = banzhf_values

    def calculate_2_steps(self):

        # Get a list of all coalitions
        def _s_iterate_all_coalitions():
            res = list()
            for si in self.coalitions_value.keys():
                v_si = self.coalitions_value[si]

                sj_list = list(self.coalitions_value.keys())
                for sj in sj_list:
                    v_sj = self.coalitions_value[sj]
                    si_l = [int(t) for t in si.split('_')]
                    sj_l = [int(t) for t in sj.split('_')]
                    possible_j = sj_l
                    if len(possible_j) == 0:
                        continue
                    for i in si_l:
                        for j in possible_j:
                            if i != j:
                                res += [(i, si, j, sj)]
            return res

        # Evaluate a single coalition
        def _s_eval_tuple(t):
            i, si, j, sj = t

            # print(f'Case: ({i}, {si}, {j}, {sj})')
            if si == '2_4' and sj == '2_3_4':
                if i == 2 and j == 3:
                    # DEBUG
                    q = 1

            si_l = [int(t) for t in si.split('_')]
            sj_l = [int(t) for t in sj.split('_')]
            intersection = [tq for tq in si_l if tq in sj_l]
            intersection_no_i = [tq for tq in intersection if tq != i]
            intersection_no_j = [tq for tq in intersection if tq != j]
            intersection_no_i_no_j = [tq for tq in intersection if tq not in [i, j]]

            si_no_i = [tq for tq in si_l if tq not in [i]]
            sj_no_j = [tq for tq in sj_l if tq not in [j]]
            si_only = [tq for tq in si_l if tq not in intersection]
            sj_only = [tq for tq in sj_l if tq not in intersection]
            si_only_no_i = [tq for tq in si_only if tq != i]
            sj_only_no_j = [tq for tq in sj_only if tq != j]

            v_si = self.coalitions_value[si]
            v_sj = self.coalitions_value[sj]
            v_step = self.bargain_step

            min_vi = v_si % v_step
            if min_vi == 0:
                min_vi = v_step

            min_vj = v_sj % v_step
            if min_vj == 0:
                min_vj = v_step

            payoffs = {k: 0 for k in list(set(si_l + sj_l))}
            winner = list()
            winning_coalition = list()

            # Case 5.1.1
            if len(intersection) == 0:
                v_i = v_si - (v_step * (len(si_l) - 1))
                if v_i > 0:
                    winner += [i]
                    winning_coalition += [si]
                    for si_member in si_l:
                        payoffs[si_member] = v_step
                    payoffs[i] = v_i

                v_j = v_sj - (v_step * (len(sj_l) - 1))
                if v_j > 0:
                    winner += [j]
                    winning_coalition += [sj]
                    for sj_member in sj_l:
                        payoffs[sj_member] = v_step
                    payoffs[j] = v_j

            # Case 5.1.2
            elif (i not in intersection) and (j not in intersection):
                vj_sj_no_j_unit = v_step
                vj_sj_no_j = len(sj_only_no_j) * v_step
                vj_intersection = v_sj - vj_sj_no_j - min_vj

                vj_intersection_unit = vj_intersection / float(len(intersection))
                vj_intersection_unit = int(np.floor(vj_intersection_unit / v_step) * v_step)

                vi_si_no_i_unit = v_step
                vi_si_no_i = len(si_only_no_i) * v_step
                vi_intersection = v_si - vi_si_no_i - min_vi

                vi_intersection_unit = vi_intersection / float(len(intersection))
                vi_intersection_unit = int(np.floor(vi_intersection_unit / v_step) * v_step)

                if (vj_intersection_unit <= 0) and (vi_intersection_unit <= 0):
                    # No coalition is formed
                    pass
                elif vj_intersection_unit > vi_intersection_unit:
                    # J wins
                    v_intersection = max(vi_intersection_unit, 0) + v_step
                    v_outside = v_step

                    v_j = v_sj - (len(intersection) * v_intersection) - (len(sj_only_no_j) * v_outside)
                    if v_j > 0:
                        winner += [j]
                        winning_coalition += [sj]

                        for sj_member in intersection:
                            payoffs[sj_member] = v_intersection
                        for sj_member in sj_only_no_j:
                            payoffs[sj_member] = v_outside
                        payoffs[j] = v_j
                    else:
                        # J cannot build a coalition
                        pass
                else:
                    # I wins
                    v_intersection = max(vj_intersection_unit, v_step)
                    v_outside = v_step
                    v_i = v_si - (len(intersection) * v_intersection) - (len(si_only_no_i) * v_outside)
                    if v_i > 0:
                        winner += [i]
                        winning_coalition += [si]

                        for si_member in intersection:
                            payoffs[si_member] = v_intersection
                        for si_member in si_only_no_i:
                            payoffs[si_member] = v_outside
                        payoffs[i] = v_i

            elif (i not in intersection) and (j in intersection):
                vj_intersection_no_j = len(intersection_no_j) * v_step
                vj_outside = len(sj_only_no_j) * v_step
                vj_j = v_sj - vj_intersection_no_j - vj_outside
                if vj_j <= 0:
                    # No coalition can be formed by j
                    vi_non_i = v_step
                    vi_i = v_si - (vi_non_i * (len(si_l) - 1))

                    if vi_i > 0:
                        winner += [i]
                        winning_coalition += [si]

                        for si_member in [qt for qt in si_l if qt != i]:
                            payoffs[si_member] = vi_non_i
                        payoffs[i] = vi_i
                    else:
                        # No coalition formed
                        pass
                else:
                    vi_intersection_no_j = len(intersection_no_j) * v_step
                    vi_outside = len(si_only_no_i) * v_step
                    vi_j = vj_j  # Should he give j an additional step?
                    vi_i = v_si - vi_intersection_no_j - vi_outside - vi_j

                    if vi_i > 0:
                        # i builds the coalition
                        winner += [i]
                        winning_coalition += [si]

                        payoffs[i] = vi_i
                        payoffs[j] = vi_j
                        for si_member in intersection_no_j:
                            payoffs[si_member] = v_step
                        for si_member in si_only_no_i:
                            payoffs[si_member] = v_step
                    else:
                        # j build the coalition
                        winner += [j]
                        winning_coalition += [sj]

                        payoffs[j] = vj_j
                        for sj_member in intersection_no_j:
                            payoffs[sj_member] = v_step
                        for sj_member in sj_only_no_j:
                            payoffs[sj_member] = v_step


            # 5.1.3
            elif (i in intersection) and (j in intersection):
                vi_intersection_no_i_unit = v_step
                vi_si_no_i_unit = v_step
                vi_intersection_no_i = vi_intersection_no_i_unit * len(intersection_no_i)
                vi_si_no_i = vi_si_no_i_unit * len(si_only_no_i)
                vi_i = v_si - vi_intersection_no_i - vi_si_no_i
                vi_j = v_step

                if vi_i <= 0:
                    # i can't form any coalition
                    vj_intersection_no_j = v_step * len(intersection_no_j)
                    vj_sj_no_j = v_step * len(sj_only_no_j)
                    vj = v_sj - vj_sj_no_j - vj_intersection_no_j

                    if vj <= 0:
                        # j also can't form any coalition
                        pass
                    else:
                        # j can form a coalition, by i can't
                        winner += [j]
                        winning_coalition += [sj]

                        for sj_member in intersection_no_j:
                            payoffs[sj_member] = v_step
                        for sj_member in sj_only_no_j:
                            payoffs[sj_member] = v_step
                        payoffs[j] = vj
                else:
                    # i has a valid coalition
                    # Let's check if j can out-perform i
                    vj_sj_no_j_unit = v_step
                    vj_intersection_no_i_no_j_unit = 2 * v_step

                    vj_sj_no_j = vj_sj_no_j_unit * len(sj_only_no_j)
                    vj_intersection_no_i_no_j = vj_intersection_no_i_no_j_unit * len(intersection_no_i_no_j)

                    vj_i = vi_i + v_step
                    vj_j = v_sj - vj_sj_no_j - vj_intersection_no_i_no_j - vj_i
                    if vj_j > 0:
                        # j can outperform i
                        winner += [j]
                        winning_coalition += [sj]

                        for sj_member in intersection_no_i_no_j:
                            payoffs[sj_member] = vj_intersection_no_i_no_j_unit
                        for sj_member in sj_only_no_j:
                            payoffs[sj_member] = vj_sj_no_j_unit
                        payoffs[j] = vj_j
                        payoffs[i] = vj_i
                    else:
                        # j cannot outperform i
                        # i takes
                        winner += [i]
                        winning_coalition += [si]

                        for si_member in intersection_no_i_no_j:
                            payoffs[si_member] = vi_intersection_no_i_unit
                        for si_member in si_only_no_i:
                            payoffs[si_member] = vi_si_no_i_unit
                        payoffs[j] = vi_j
                        payoffs[i] = vi_i


            # Case 5.1.4
            elif (i in intersection) and (j not in intersection):
                # i offer
                v_non_i_intersection = v_step * (len(intersection) - 1)
                v_si_only = v_step * len(si_only)
                v_i = v_si - (v_non_i_intersection + v_si_only)

                vj_sj_non_j_unit = v_step
                vj_sj_non_j = vj_sj_non_j_unit * len(sj_only_no_j)
                v_j = v_sj
                v_j -= v_non_i_intersection + (v_step * (len(intersection) - 1))  # Pay non-i intersection
                v_j -= (max(v_i, v_step) + v_step)  # Pay i
                v_j -= vj_sj_non_j

                if v_j > 0 and v_i > 0:
                    # J wins
                    winner += [j]
                    winning_coalition += [sj]

                    payoffs[i] = v_i + v_step
                    for sj_member in [t for t in intersection if t not in [i, j]]:
                        payoffs[sj_member] = 2 * v_step
                    for sj_member in [t for t in sj_only if t not in [i, j]]:
                        payoffs[sj_member] = v_step
                    payoffs[j] = v_j

                    if sum(payoffs.values()) != v_sj:
                        raise Exception("Sanity! bad sum")

                elif (v_j <= 0) and (v_i > 0):
                    # i wins
                    winner += [i]
                    winning_coalition += [si]

                    tot = 0
                    for si_member in [t for t in si_l if t not in [i]]:
                        tot += v_step
                        payoffs[si_member] = v_step
                    payoffs[i] = v_si - tot

                elif v_i <= 0 and (v_j > 0):
                    # J wins
                    winner += [j]
                    winning_coalition += [sj]

                    tot = 0
                    v_non_j = v_step
                    v_j = v_sj - (v_step * (len(sj_l) - 1))
                    if v_j > 0:
                        for sj_member in [t for t in intersection if t not in [j]]:
                            payoffs[sj_member] = v_non_j

                        for sj_member in sj_only_no_j:
                            payoffs[sj_member] = vj_sj_non_j_unit

                        payoffs[j] = v_j
                    else:
                        # No coalition is formed.
                        pass

                else:
                    pass

            else:
                raise Exception(f"UNKNOWN STATE FOR ({i}, {si}, {j}, {sj})")

            info = dict()
            info['winner'] = ','.join([str(t) for t in winner])
            info['winning_coalition'] = ','.join([str(t) for t in winning_coalition])
            info['payoffs'] = payoffs
            info['i'] = i
            info['j'] = j
            info['si'] = si
            info['sj'] = sj
            return payoffs, info

        # Auxilaty
        def s_tuple_to_sig(t):
            s = f'({t[0]}){t[1]} --> ({t[2]}){t[3]}'
            return s

        vals = dict()
        info = dict()
        for t in _s_iterate_all_coalitions():
            res, info_t = _s_eval_tuple(t)
            key = s_tuple_to_sig(t)
            vals[key] = res
            info[key] = info_t
        resdf = pd.DataFrame(vals).T.fillna(0).astype(int)

        resdf['winner'] = ''
        resdf['Coalition'] = ''
        resdf['Coalition value'] = 0
        for k, info_t in info.items():
            winning_coal = info_t['winning_coalition']
            resdf.loc[k, 'winner'] = info_t['winner']
            resdf.loc[k, 'Coalition'] = winning_coal
            if len(winning_coal.split(',')) > 1:
                q = [self.coalitions_value.get(winning_coal_t, 0) for winning_coal_t in winning_coal.split(',')]
                resdf.loc[k, 'Coalition value'] = sum(q)
            else:
                resdf.loc[k, 'Coalition value'] = self.coalitions_value.get(winning_coal, 0)

            resdf.loc[k, 'Real value'] = resdf.loc[k, self.players].sum()
        resdf['valid sum'] = resdf['Coalition value'] == resdf['Real value']

        self.two_step_bargaining = resdf[self.players + ['winner', 'Coalition']]
        self.powerIndex['2-step'] = resdf[self.players].mean()
        q = 1

    def step(self):
        self.calculate_2_steps()
        self.calculate_shapley()

    def _update(self):
        self.players_count = len(
            list(set(itertools.chain.from_iterable([str_to_coalition_keys(t) for t in self.coalitions_value.keys()]))))
        if self.players_count == 0:
            self.players = list()
        else:
            self.players = list(range(1, self.players_count + 1))

        self.info['players_count'] = self.players_count
        self.info['players'] = self.players
        self.info['bargain_step'] = self.bargain_step

        self.info['coalitions_value'] = self.coalitions_value
        self.info['default_coalition_value'] = self.default_coalition_value

        self.powerIndex = pd.DataFrame(index=self.players)

    def add_coalitions_from_dict(self, coalitions_value):
        self.coalitions_value.update(coalitions_value)
        self._update()

        # Fill all empty combinations
        k = [sorted(t) for t in chain.from_iterable(
            combinations(self.players, r) for r in range(len(self.players) + 1)) if len(t) > 0]
        k = ['_'.join([str(ktt) for ktt in kt]) for kt in k]
        k = [kt for kt in k if kt not in self.coalitions_value.keys()]
        for kt in k:
            coalitions_value[kt] = self.default_coalition_value

        self.coalitions_value.update(coalitions_value)
        self._update()


def coalition_keys_to_str(c):
    c_list = [c_t + 1 for c_t in c]
    r = '_'.join([str(c_int) for c_int in c_list])
    return r


def str_to_coalition_keys(s):
    r = sorted(list(set([int(t) - 1 for t in s.split('_')])))
    return r


if __name__ == '__main__':
    coalitions_value = dict()

    coalitions_value['1_2'] = 1000
    coalitions_value['1_2_4'] = 1000
    coalitions_value['1_3'] = 1000
    coalitions_value['1_3_4'] = 1000

    bargain_step = 50
    default_coalition_value = 0

    game = BargainGame(bargain_step=bargain_step)
    game.add_coalitions_from_dict(coalitions_value)
    print(f"Players in game: {game.players_count}")

    game.step()
    print("")
    print(game.powerIndex.head(10))

    df = game.two_step_bargaining
    j = 3
