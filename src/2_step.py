import sys
import numpy as np

import itertools

import pandas as pd
from itertools import chain, combinations

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
        self.info = dict()
        self._update()

        self.metacoalitions = None
        self.powerIndex = None

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

    # ----------------- SPECIFIC -------------------
    def s_tuple_to_sig(self, t):
        s = f'({t[0]}){t[1]} --> ({t[2]}){t[3]}'
        return s

    def s_iterate_all_coalitions(self):
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

    def s_eval_tuple(self, t):
        i, si, j, sj = t

        print(f'Case: ({i}, {si}, {j}, {sj})')
        if si == '2_3' and sj == '1_2_3':
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
                v_intersection = vi_intersection_unit + v_step
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
                v_intersection = vj_intersection_unit
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
            v_j -= (v_i + v_step)  # Pay i
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

            elif v_i <= 0:
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


def coalition_keys_to_str(c):
    c_list = [c_t + 1 for c_t in c]
    r = '_'.join([str(c_int) for c_int in c_list])
    return r


def str_to_coalition_keys(s):
    r = sorted(list(set([int(t) - 1 for t in s.split('_')])))
    return r


if __name__ == '__main__':
    coalitions_value = dict()

    coalitions_value['1_2_3'] = 100
    coalitions_value['2_3'] = 50
    coalitions_value['1_2'] = 100
    coalitions_value['1_3'] = 100

    bargain_step = 10
    default_coalition_value = 0

    game = BargainGame(bargain_step=bargain_step)
    game.add_coalitions_from_dict(coalitions_value)
    print(f"Players in game: {game.players_count}")

    vals = dict()
    info = dict()
    for t in game.s_iterate_all_coalitions():
        res, info_t = game.s_eval_tuple(t)
        key = game.s_tuple_to_sig(t)
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
            q = [game.coalitions_value.get(winning_coal_t, 0) for winning_coal_t in winning_coal.split(',')]
            resdf.loc[k, 'Coalition value'] = sum(q)
        else:
            resdf.loc[k, 'Coalition value'] = game.coalitions_value.get(winning_coal, 0)

        resdf.loc[k, 'Real value'] = resdf.loc[k, game.players].sum()
    resdf['valid sum'] = resdf['Coalition value'] == resdf['Real value']

    print(f"Game played: {resdf.shape[0]}")
    print("Power:")
    print(resdf[game.players].mean())
    print(resdf[game.players].mean().sum())

    print("END OF CODE.")
