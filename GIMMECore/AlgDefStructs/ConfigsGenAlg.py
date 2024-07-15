import gc
import random
import copy
import re
# from abc import ABC, abstractmethod

import GIMMESolver as gs

from deap import base, creator, tools, algorithms

# from ..InteractionsProfile import InteractionsProfile
# from ..PlayerStructs import *
from ..AlgDefStructs.QualityEvalAlg import *


class ConfigsGenAlg(ABC):
    # protected members
    _joint_players_constraints = None
    _separated_players_constraints = None
    _all_constraints = None
    _min_num_players_per_group = None
    _max_num_players_per_group = None
    _player_model_bridge = None
    _task_model_bridge = None
    _interactions_profile_template = None
    _completion_percentage = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 task_model_bridge=None,
                 preferred_num_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):

        self._joint_players_constraints = []
        self._separated_players_constraints = []
        self._all_constraints = []

        min_num_players_per_group = 2 if min_num_players_per_group is None else min_num_players_per_group
        max_num_players_per_group = 5 if max_num_players_per_group is None else max_num_players_per_group

        if min_num_players_per_group > max_num_players_per_group:
            raise ValueError('The min number of players per group cannot be higher than the max!')

        if preferred_num_players_per_group is None:
            self._max_num_players_per_group = max_num_players_per_group
            self._min_num_players_per_group = min_num_players_per_group
        else:
            self._max_num_players_per_group = preferred_num_players_per_group
            self._min_num_players_per_group = preferred_num_players_per_group - 1

        self._player_model_bridge = player_model_bridge
        self._task_model_bridge = task_model_bridge
        self._interactions_profile_template = interactions_profile_template

        joint_player_constraints = self._from_string_constraint_to_list(joint_player_constraints)
        separated_player_constraints = self._from_string_constraint_to_list(separated_player_constraints)

        for i in range(len(joint_player_constraints)):
            if joint_player_constraints[i] == ['']:
                continue
            self.add_joint_players_constraints(joint_player_constraints[i])

        for i in range(len(separated_player_constraints)):
            if separated_player_constraints[i] == ['']:
                continue
            self.add_separated_players_constraints(separated_player_constraints[i])

        self._completion_percentage = 0.0

    def _random_config_generator(self, player_ids, min_num_groups, max_num_groups):

        returned_config = []
        player_joint_requirements = {}
        player_separated_requirements = {}
        if self._joint_players_constraints != []:
            for player_id in player_ids:
                player_joint_requirements[str(player_id)] = []

            for constraint in self._joint_players_constraints:
                for i in range(len(constraint)):

                    for j in range(len(constraint)):
                        if constraint[i] == constraint[j]:
                            continue

                        player_joint_requirements[constraint[i]].append(constraint[j])

            for player_id in player_ids:
                for restrictedId in player_joint_requirements[player_id]:
                    for restrictionOfRestrictedId in player_joint_requirements[restrictedId]:
                        if (restrictionOfRestrictedId not in player_joint_requirements[player_id] and
                                restrictionOfRestrictedId != restrictedId):
                            player_joint_requirements[player_id].append(restrictionOfRestrictedId)

        if self._separated_players_constraints != []:
            for player_id in player_ids:
                player_separated_requirements[str(player_id)] = []

            for constraint in self._separated_players_constraints:
                for i in range(len(constraint)):

                    for j in range(len(constraint)):
                        if constraint[i] == constraint[j]:
                            continue

                        player_separated_requirements[constraint[i]].append(constraint[j])

        if len(player_ids) < self._min_num_players_per_group:
            raise ValueError("number of players is lower than the minimum number of players per group!")

        # generate random config
        players_without_group = player_ids.copy()

        if min_num_groups < max_num_groups:
            num_groups = numpy.random.randint(min_num_groups, max_num_groups)
        else:  # players length is 1
            num_groups = max_num_groups

        # generate min num players for each group
        players_without_group_size = len(players_without_group)

        # playersWithoutGroupWithoutRestrictions = list(set(players_without_group) - set(listOfPlayersWithJointRequirements))
        for g in range(num_groups):
            curr_group = []

            if (players_without_group_size < 1):
                break

            # add min number of players to the group
            for p in range(self._min_num_players_per_group):
                curr_player_index = random.randint(0, len(players_without_group) - 1)

                curr_player_id = players_without_group[curr_player_index]
                curr_group.append(curr_player_id)
                del players_without_group[curr_player_index]

            if ((player_separated_requirements != {} or player_joint_requirements != {}) and
                    len(players_without_group) > 0):
                self._verify_coalition_validity(curr_group, player_joint_requirements, player_separated_requirements,
                                                players_without_group)
            returned_config.append(curr_group)

        # append the rest
        players_without_group_size = len(players_without_group)
        while players_without_group_size > 0:
            curr_player_index = 0
            if players_without_group_size > 1:
                curr_player_index = random.randint(0, players_without_group_size - 1)
            curr_player_id = players_without_group[curr_player_index]

            available_groups = returned_config.copy()
            curr_group = random.choice(available_groups)
            while len(curr_group) > (self._max_num_players_per_group - 1):
                if len(available_groups) < 1:
                    curr_group = random.choice(returned_config)
                    break
                curr_group = random.choice(available_groups)
                available_groups.remove(curr_group)

            curr_group.append(curr_player_id)

            del players_without_group[curr_player_index]
            players_without_group_size = len(players_without_group)

        return returned_config

    def _verify_coalition_validity(self, config, player_joint_requirements, player_separated_requirements,
                                   players_without_group):

        for i in range(len(config)):
            if player_joint_requirements[config[i]] != []:
                players_not_in_coalition = []
                for player in player_joint_requirements[config[i]]:
                    if player not in config:
                        players_not_in_coalition.append(player)

                if players_not_in_coalition != []:
                    for j in range(len(config)):
                        if i != j and players_not_in_coalition[0] in players_without_group and config[j] not in \
                                player_joint_requirements[config[i]]:
                            players_without_group.append(config[j])
                            config[j] = players_not_in_coalition[0]
                            players_without_group.remove(players_not_in_coalition[0])
                            del players_not_in_coalition[0]

                            if len(players_not_in_coalition) == 0:
                                break

            if player_separated_requirements[config[i]] != []:
                for player in player_separated_requirements[config[i]]:
                    if player in config:
                        curr_player_index = random.randint(0, len(players_without_group) - 1)
                        while players_without_group[curr_player_index] in player_separated_requirements[config[i]]:
                            curr_player_index = random.randint(0, len(players_without_group) - 1)

                        config.remove(player)
                        config.append(players_without_group[curr_player_index])
                        del players_without_group[curr_player_index]

                        players_without_group.append(player)

        return config

    def _from_string_constraint_to_list(self, constraints):
        constraints = constraints.split(';')
        for i in range(len(constraints)):
            constraints[i] = re.sub('[^A-Za-z0-9,_]+', '', constraints[i]).split(',')
        return constraints

    # the next methods can be used to operate coalitions with a bit representation
    def _convert_coalition_from_byte_to_bit_format(self, coalition_in_byte_format, coalition_size):
        coalition_in_bit_format = 0
        for i in range(coalition_size):
            coalition_in_bit_format += 1 << (coalition_in_byte_format[i] - 1)
        return coalition_in_bit_format

    # convert group in bit format to group with the player ids
    def _get_group_from_bit_format(self, coalition):
        group = []
        temp_coalition = coalition
        player_number = 0
        while temp_coalition != 0:
            if temp_coalition & 1:
                group.append(player_number + 1)

            player_number += 1
            temp_coalition >>= 1
        return group

    def _convert_from_byte_to_ids(self, coalition):
        player_ids = self._player_model_bridge.get_all_player_ids()
        group = []
        for agent in coalition:
            group.append(player_ids[agent - 1])
        return group

    def _convert_from_ids_to_bytes(self, coalition):
        player_ids = self._player_model_bridge.get_all_player_ids()
        group = []
        for agent in coalition:
            for i in range(len(player_ids)):
                if player_ids[i] == agent:
                    group.append(i + 1)
        return group

    def _convert_set_of_combinations_from_bit_format(self, set_of_combinations_in_bit_format):
        set_of_combinations_in_byte_format = numpy.empty(len(set_of_combinations_in_bit_format), dtype=list)
        for i in range(len(set_of_combinations_in_bit_format)):
            set_of_combinations_in_byte_format[i] = self._get_group_from_bit_format(
                set_of_combinations_in_bit_format[i])
        return set_of_combinations_in_byte_format

    def _compute_coalitions_restrictions(self):
        joint_players_constraint_in_bit_format = self._joint_players_constraints[:]
        separated_players_constraint_in_bit_format = self._separated_players_constraints[:]

        for i in range(len(joint_players_constraint_in_bit_format)):
            joint_players_constraint_in_bit_format[i] = (
                self._convert_from_ids_to_bytes(joint_players_constraint_in_bit_format[i]))
            joint_players_constraint_in_bit_format[i] = self._convert_coalition_from_byte_to_bit_format(
                joint_players_constraint_in_bit_format[i], len(joint_players_constraint_in_bit_format[i]))

        for i in range(len(separated_players_constraint_in_bit_format)):
            separated_players_constraint_in_bit_format[i] = self._convert_from_ids_to_bytes(
                separated_players_constraint_in_bit_format[i])
            separated_players_constraint_in_bit_format[i] = self._convert_coalition_from_byte_to_bit_format(
                separated_players_constraint_in_bit_format[i], len(separated_players_constraint_in_bit_format[i]))

        return joint_players_constraint_in_bit_format, separated_players_constraint_in_bit_format

    def _convert_cs_bytes_to_dict(self, cs_in_byte_format, coalitions_profiles, coalitions_avg_characteristics):
        best_groups = []
        best_groups_in_bit_format = []
        best_config_profiles = []
        avg_characteristics_array = []

        for coalition in cs_in_byte_format:
            best_groups.append(self._convert_from_byte_to_ids(coalition))
            best_groups_in_bit_format.append(self._convert_coalition_from_byte_to_bit_format(coalition, len(coalition)))

        for group in best_groups_in_bit_format:
            best_config_profiles.append(coalitions_profiles[group])
            avg_characteristics_array.append(coalitions_avg_characteristics[group])

        return {"groups": best_groups,
                "profiles": best_config_profiles,
                "avgCharacteristics": avg_characteristics_array}

    @abstractmethod
    def organize(self):
        pass

    def add_joint_players_constraints(self, players):
        self._joint_players_constraints.append(players)
        self._all_constraints.append({"players": players, "type": "JOIN"})

    def add_separated_players_constraints(self, players):
        self._separated_players_constraints.append(players)
        self._all_constraints.append({"players": players, "type": "SEPARATE"})

    def reset_players_constraints(self):
        self._joint_players_constraints = []
        self._separated_players_constraints = []
        self._all_constraints = []

    def get_player_constraints(self):
        return self._all_constraints

    def get_completion_percentage(self):
        return self._completion_percentage


class RandomConfigsGenAlg(ConfigsGenAlg):

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 preferred_num_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):
        super().__init__(
            player_model_bridge=player_model_bridge,
            interactions_profile_template=interactions_profile_template,
            preferred_num_players_per_group=preferred_num_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

    def organize(self):
        player_ids = self._player_model_bridge.get_all_player_ids()
        min_num_groups = math.ceil(len(player_ids) / self._max_num_players_per_group)
        max_num_groups = math.floor(len(player_ids) / self._min_num_players_per_group)

        new_config_profiles = []
        new_avg_characteristics = []
        new_groups = self._random_config_generator(player_ids, min_num_groups, max_num_groups)
        new_config_size = len(new_groups)
        # generate profiles
        for groupI in range(new_config_size):
            group = new_groups[groupI]
            group_size = len(group)

            profile = self._interactions_profile_template.generate_copy().randomize()
            new_config_profiles.append(profile)

            curr_avg_characteristics = PlayerCharacteristics().reset()
            for currPlayer in group:
                curr_state = self._player_model_bridge.get_player_curr_state(currPlayer)
                curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

            diversity_value_alg = DiversityQualityEvalAlg(self._player_model_bridge, 0)
            personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
            curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                personalities)

            new_avg_characteristics.append(curr_avg_characteristics)

            self.completion_percentage = groupI / new_config_size

        return {"groups": new_groups, "profiles": new_config_profiles, "avgCharacteristics": new_avg_characteristics}


class PureRandomSearchConfigsGenAlg(ConfigsGenAlg):
    # private members
    __quality_eval_alg = None
    __pref_est_alg = None
    __num_config_choices = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 pref_est_alg,
                 num_config_choices=None,
                 preferred_num_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):

        super().__init__(
            player_model_bridge=player_model_bridge,
            interactions_profile_template=interactions_profile_template,
            preferred_num_players_per_group=preferred_num_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

        self.__quality_eval_alg = quality_eval_alg
        self.__pref_est_alg = pref_est_alg
        self.__num_config_choices = 100 if num_config_choices is None else num_config_choices

    def organize(self):
        player_ids = self._player_model_bridge.get_all_player_ids()
        min_num_groups = math.ceil(len(player_ids) / self._max_num_players_per_group)
        max_num_groups = math.floor(len(player_ids) / self._min_num_players_per_group)

        curr_max_quality = -float("inf")
        best_groups = []
        best_config_profiles = []
        best_avg_characteristics = []

        # estimate preferences
        self.__pref_est_alg.update_estimates()

        # generate several random groups, calculate their fitness and select the best one
        for i in range(self.__num_config_choices):

            # generate several random groups
            new_groups = self._random_config_generator(player_ids, min_num_groups, max_num_groups)
            new_config_size = len(new_groups)
            curr_quality = 0.0
            new_config_profiles = []
            new_avg_characteristics = []

            # generate profiles
            for groupI in range(new_config_size):
                group = new_groups[groupI]
                group_size = len(group)

                # generate profile as average of the preferences estimates
                profile = self._interactions_profile_template.generate_copy().reset()

                for curr_player in group:
                    preferences = self._player_model_bridge.get_player_preferences_est(curr_player)
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / group_size)

                new_config_profiles.append(profile)

                # calculate quality and average state
                curr_avg_characteristics = PlayerCharacteristics()
                curr_avg_characteristics.reset()
                for i in range(len(group)):
                    curr_state = self._player_model_bridge.get_player_curr_state(group[i])
                    curr_state.profile = profile

                    curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                    curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                curr_quality += self.__quality_eval_alg.evaluate(profile, group)

                diversity_value_alg = DiversityQualityEvalAlg(self._player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
                curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                    personalities)

                new_avg_characteristics.append(curr_avg_characteristics)

            if curr_quality > curr_max_quality:
                best_groups = new_groups
                best_config_profiles = new_config_profiles
                best_avg_characteristics = new_avg_characteristics
                curr_max_quality = curr_quality

            self.completion_percentage = i / self.__num_config_choices

        return {"groups": best_groups, "profiles": best_config_profiles, "avgCharacteristics": best_avg_characteristics}


class EvolutionaryConfigsGenAlg(ConfigsGenAlg):
    # private members
    __quality_eval_alg = None
    __initial_population_size = None
    __num_evolutions_per_iteration = None

    __prob_cross = None
    __prob_mut = None
    __prob_mut_config = None
    __prob_mut_profiles = None

    __num_children_per_iteration = None
    __num_survivors = None

    __cx_op = None

    __search_id = None

    __fitness_func_id = None
    __individual_id = None

    __toolbox = None
    __pop = None
    __hof = None

    __player_ids = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 preferred_num_players_per_group=None,

                 initial_population_size=None,
                 num_evolutions_per_iteration=None,
                 prob_cross=None,

                 prob_mut=None,
                 prob_mut_config=None,
                 prob_mut_profiles=None,

                 num_children_per_iteration=None,
                 num_survivors=None,

                 cx_op=None,

                 joint_player_constraints="",
                 separated_player_constraints=""):

        super().__init__(
            player_model_bridge=player_model_bridge,
            interactions_profile_template=interactions_profile_template,
            preferred_num_players_per_group=preferred_num_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

        self.__quality_eval_alg = quality_eval_alg

        self.__initial_population_size = 100 if initial_population_size is None else initial_population_size

        self.__num_evolutions_per_iteration = 500 if num_evolutions_per_iteration is None \
            else num_evolutions_per_iteration

        self.__prob_mut = 0.2 if prob_mut is None else prob_mut
        self.__prob_cross = 0.7 if prob_cross is None else prob_cross

        self.__prob_mut_config = 0.2 if prob_mut_config is None else prob_mut_config
        self.__prob_mut_profiles = 0.2 if prob_mut_profiles is None else prob_mut_profiles

        self.__num_children_per_iteration = 5 if num_children_per_iteration is None else num_children_per_iteration
        self.__num_survivors = 5 if num_survivors is None else num_survivors

        self.__quality_eval_alg = KNNRegQualityEvalAlg(player_model_bridge=player_model_bridge, k=5) \
            if quality_eval_alg is None else quality_eval_alg

        self.__search_id = str(id(self))

        self.__fitness_func_id = "fitness_func_" + self.__search_id
        self.__individual_id = "individual_" + self.__search_id

        creator.create(self.__fitness_func_id, base.Fitness, weights=(1.0,))
        creator.create(self.__individual_id, list, fitness=getattr(creator, self.__fitness_func_id))

        # # conv test
        # creator.create(fitness_func_id, base.Fitness, weights=(-1.0,))
        # creator.create(individual_id, list, fitness=getattr(creator, fitness_func_id))

        self.__player_ids = self._player_model_bridge.get_all_player_ids()
        min_num_groups = math.ceil(len(self.__player_ids) / self._max_num_players_per_group)
        max_num_groups = math.floor(len(self.__player_ids) / self._min_num_players_per_group)

        self.__toolbox = base.Toolbox()

        # indices also left to reset because it has to be updated as needed
        self.__toolbox.register("indices", self.__random_individual_generator,
                                self.__player_ids, min_num_groups, max_num_groups)
        self.__toolbox.register("individual", tools.initIterate, getattr(creator, self.__individual_id),
                                self.__toolbox.indices)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        self.__cx_op = "order" if cx_op is None else cx_op
        if self.__cx_op == "order":
            self.__toolbox.register("mate", self.__cx_gimme_order)
        else:
            self.__toolbox.register("mate", self.__cx_gimme_simple)

        self.__toolbox.register("mutate", self.__mut_gimme, p_profiles=self.__prob_mut_profiles,
                                p_configs=self.__prob_mut_config)

        # self.toolbox.register("select", tools.selRoulette)
        # self.toolbox.register("select", tools.selBest, k=self.numFitSurvivors)
        self.__toolbox.register("select", self.__sel_gimme)

        # self.toolbox.register("evaluate", self.calcFitness_convergenceTest)
        self.__toolbox.register("evaluate", self.__calc_fitness)

        self.__pop = []
        self.__hof = []

    def __reset(self):
        # if isinstance(self.quality_eval_alg, TabQualityEvalAlg):
        #     self.playerPrefEstimates = self.pers_est_alg.update_estimates()

        new_player_ids = self._player_model_bridge.get_all_player_ids()
        if self.__player_ids != new_player_ids:
            self.__player_ids = new_player_ids
            min_num_groups = math.ceil(len(self.__player_ids) / self._max_num_players_per_group)
            max_num_groups = math.floor(len(self.__player_ids) / self._min_num_players_per_group)

            self.__toolbox.register("indices", self.__random_individual_generator,
                                    self.__player_ids, min_num_groups, max_num_groups)
            self.__toolbox.register("individual", tools.initIterate, getattr(creator, self.__individual_id),
                                    self.__toolbox.indices)
            self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        if hasattr(self, "pop"):
            del self.__pop
        if hasattr(self, "hof"):
            del self.__hof

        self.__pop = self.__toolbox.population(n=self.__initial_population_size)
        self.__hof = tools.HallOfFame(1)

    def __random_individual_generator(self, player_ids, min_num_groups, max_num_groups):
        groups = self._random_config_generator(player_ids, min_num_groups, max_num_groups)
        profs = [self._interactions_profile_template.randomized() for i in range(len(groups))]
        return [groups, profs]

    def __cx_gimme_order(self, ind1, ind2):

        # print("-----")
        # print("ind1:")
        # print(ind1)
        # print("ind2:")
        # print(ind2)
        # print("-----")

        # configs
        config1 = ind1[0]
        config2 = ind2[0]

        new_config1 = []
        new_config2 = []

        l1 = len(config1)
        l2 = len(config2)

        if l1 > l2:
            max_len_config = config1
            min_len_config = config2
            min_len = l2
        else:
            max_len_config = config2
            min_len_config = config1
            min_len = l1

        cxpoints = []

        clist1 = []
        clist2 = []

        remainder1 = []
        remainder2 = []
        for i in range(min_len):
            parent1 = min_len_config[i]
            parent2 = max_len_config[i]

            cxpoint = random.randint(0, len(min_len_config[i]))
            cxpoints.append(cxpoint)

            clist1.extend(parent1)
            clist2.extend(parent2)

            remainder1.extend(parent1[cxpoint:])
            remainder2.extend(parent2[cxpoint:])

        d1 = {k: v for v, k in enumerate(clist1)}
        d2 = {k: v for v, k in enumerate(clist2)}

        remainder1.sort(key=d2.get)
        remainder2.sort(key=d1.get)

        for i in range(min_len):
            parent1 = min_len_config[i]
            parent2 = max_len_config[i]

            cxpoint = cxpoints[i]

            #C1 Implementation
            # maintain left part
            child1, child2 = parent1[:cxpoint], parent2[:cxpoint]

            # reorder right part
            missing_len1 = len(parent1) - len(child1)
            child1.extend(remainder1[:missing_len1])
            remainder1 = remainder1[missing_len1:]

            missing_len2 = len(parent2) - len(child2)
            child2.extend(remainder2[:missing_len2])
            remainder2 = remainder2[missing_len2:]

            new_config1.append(child1)
            new_config2.append(child2)

        #the inds become children
        ind1[0] = new_config1
        ind2[0] = new_config2

        # profiles are crossed with one point (no need for that when profiles are 1D)

        # if self.interactionsProfileTemplate.dimensionality > 1:
        for i in range(min_len):
            prof1 = ind1[1][i].flattened()
            prof2 = ind2[1][i].flattened()

            new_profiles = tools.cxUniform(prof1, prof2, 0.5)
            # new_profiles = tools.cxOnePoint(prof1, prof2)

            #the inds become children
            ind1[1][i] = self._interactions_profile_template.unflattened(new_profiles[0])
            ind2[1][i] = self._interactions_profile_template.unflattened(new_profiles[1])

        del ind1.fitness.values
        del ind2.fitness.values

        return ind1, ind2

    def __cx_gimme_simple(self, ind1, ind2):

        # configs
        config1 = ind1[0]
        config2 = ind2[0]

        l1 = len(config1)
        l2 = len(config2)

        if l1 > l2:
            max_len_config = config1
            min_len_config = config2
            min_len = l2
        else:
            max_len_config = config2
            min_len_config = config1
            min_len = l1

        clist1 = []
        clist2 = []

        for i in range(min_len):
            parent1 = [None, None]
            parent2 = [None, None]

            parent1[0] = min_len_config[i]
            parent1[1] = ind1[1][i].flattened()
            parent2[0] = max_len_config[i]
            parent2[1] = ind2[1][i].flattened()

            clist1.append(parent1)
            clist2.append(parent2)

        for ind, clist in zip([ind1, ind2], [clist1, clist2]):
            rand_i1 = random.randint(0, len(clist1) - 1)
            rand_i2 = random.randint(0, len(clist1) - 1)

            new_profiles_config = tools.cxOnePoint(ind1=clist[rand_i1][0], ind2=clist[rand_i2][0])
            new_profiles_gip = tools.cxOnePoint(ind1=clist[rand_i1][1], ind2=clist[rand_i2][1])

            ind[0][rand_i1] = new_profiles_config[0]
            ind[1][rand_i1] = self._interactions_profile_template.unflattened(new_profiles_gip[0])

            ind[0][rand_i2] = new_profiles_config[1]
            ind[1][rand_i2] = self._interactions_profile_template.unflattened(new_profiles_gip[1])

        del ind1.fitness.values
        del ind2.fitness.values

        return ind1, ind2

    def __mut_gimme(self, individual, p_profiles, p_configs):

        # mutate config
        if random.uniform(0, 1) <= p_configs:

            num_mutations = 1
            for i in range(num_mutations):
                ind_cpy = copy.copy(individual)

                rand_i1 = random.randint(0, len(ind_cpy[0]) - 1)
                inner_rand_i1 = random.randint(0, len(ind_cpy[0][rand_i1]) - 1)

                rand_i2 = inner_rand_i2 = -1
                while rand_i2 < 0 or rand_i1 == rand_i2:
                    rand_i2 = random.randint(0, len(ind_cpy[0]) - 1)
                while inner_rand_i2 < 0 or inner_rand_i1 == inner_rand_i2:
                    inner_rand_i2 = random.randint(0, len(ind_cpy[0][rand_i2]) - 1)

                elem1 = ind_cpy[0][rand_i1][inner_rand_i1]
                elem2 = ind_cpy[0][rand_i2][inner_rand_i2]

                ind_cpy[0][rand_i1][inner_rand_i1] = elem2
                ind_cpy[0][rand_i2][inner_rand_i2] = elem1

                individual[0] = ind_cpy[0]

        #mutate GIPs
        num_mutations = 1
        for i in range(num_mutations):
            profs = individual[1]
            keys = list(profs[0].dimensions.keys())
            for i in range(len(profs)):
                if random.uniform(0, 1) <= p_profiles:
                    # profs[i].randomize()
                    for key in keys:
                        if random.uniform(0, 1) <= 0.5:
                            profs[i].dimensions[key] += random.uniform(0, min(0.2, 1.0 - profs[i].dimensions[key]))
                        else:
                            profs[i].dimensions[key] -= random.uniform(0, min(0.2, profs[i].dimensions[key]))

            individual[1] = profs

        del individual.fitness.values
        return individual,

    def __calc_fitness_convergence_test(self, individual):
        config = individual[0]
        profiles = individual[1]

        total_fitness = 0.0

        target_config = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19],
                         [20, 21, 22, 23]]

        len_config = len(config)
        for groupI in range(len_config):

            group = config[groupI]
            profile = profiles[groupI]

            for playerI in range(len(group)):
                total_fitness += profile.sqr_distance_between(
                    InteractionsProfile(dimensions={'dim_0': 0.98, 'dim_1': 0.005}))
                total_fitness += abs(config[groupI][playerI] - target_config[groupI][playerI])

        # print(total_fitness)

        total_fitness = total_fitness + 1.0  # helps selection (otherwise Pchoice would always be 0)
        individual.fitness.values = total_fitness,
        return total_fitness,  # must return a tuple

    def __calc_fitness(self, individual):
        config = individual[0]
        profiles = individual[1]

        total_fitness = 0.0

        len_config = len(config)

        all_constrains_satisfied = True
        for groupI in range(len_config):

            group = config[groupI]
            profile = profiles[groupI]

            for constraint in self._joint_players_constraints:
                must_be_grouped = False
                is_not_in_group = False
                for player in constraint:
                    if player in group and is_not_in_group == False:
                        must_be_grouped = True

                    elif player not in group and must_be_grouped == False:
                        is_not_in_group = True

                    else:
                        all_constrains_satisfied = False
                        break

                if all_constrains_satisfied == False:
                    break

            for constraint in self._separated_players_constraints:
                must_be_sep = False
                for player in constraint:
                    if player in group:
                        if must_be_sep:
                            all_constrains_satisfied = False
                            break
                        must_be_sep = True

                if all_constrains_satisfied == False:
                    break

            total_fitness += self.__quality_eval_alg.evaluate(profile, group)

        total_fitness = total_fitness + 1.0  # helps selection (otherwise Pchoice would always be 0)
        if all_constrains_satisfied:
            total_fitness += 1000
        individual.fitness.values = total_fitness,

        return total_fitness,  # must return a tuple

    def __sel_gimme(self, individuals, k, fit_attr="fitness"):
        return tools.selBest(individuals, k, fit_attr)

    def organize(self):
        self.__reset()

        algorithms.eaMuCommaLambda(
            population=self.__pop,
            toolbox=self.__toolbox,

            lambda_=self.__num_children_per_iteration,
            mu=self.__num_survivors,

            cxpb=self.__prob_cross,
            mutpb=self.__prob_mut,

            ngen=self.__num_evolutions_per_iteration,

            halloffame=self.__hof,
            verbose=False
        )

        self._completion_percentage = len(tools.Logbook()) / self.__num_evolutions_per_iteration

        # print(self.__pop)
        # print(self.__hof)

        best_groups = self.__hof[0][0]
        best_profiles = self.__hof[0][1]

        avg_characteristics_array = []
        for group in best_groups:
            group_size = len(group)
            avg_characteristics = PlayerCharacteristics()
            for currPlayer in group:
                curr_state = self._player_model_bridge.get_player_curr_state(currPlayer)
                avg_characteristics.ability += curr_state.characteristics.ability / group_size
                avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                diversity_value_alg = DiversityQualityEvalAlg(self._player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
                avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                    personalities)

            avg_characteristics_array.append(avg_characteristics)

        return {"groups": best_groups, "profiles": best_profiles, "avgCharacteristics": avg_characteristics_array}


# uses the C++ solver for efficiency
class ODPIPConfigsGenAlg(ConfigsGenAlg):
    # private members
    __quality_eval_alg = None
    __pref_est_alg = None

    __coalitions_profiles = None
    __coalitions_avg_characteristics = None
    __coalitions_values = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 pref_est_alg,
                 task_model_bridge=None,
                 preferred_num_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):
        super().__init__(player_model_bridge,
                         interactions_profile_template,
                         task_model_bridge,
                         preferred_num_players_per_group,
                         min_num_players_per_group,
                         max_num_players_per_group,
                         joint_player_constraints=joint_player_constraints,
                         separated_player_constraints=separated_player_constraints)

        self.__quality_eval_alg = quality_eval_alg
        self.__pref_est_alg = pref_est_alg

        self.__coalitions_profiles = []
        self.__coalitions_avg_characteristics = []
        self.__coalitions_values = []

    def __compute_all_coalitions_values(self):
        num_of_agents = len(self.player_ids)
        num_of_coalitions = 1 << num_of_agents

        players_curr_state = {}
        for player in self.player_ids:
            players_curr_state[player] = self._player_model_bridge.get_player_curr_state(player)

        # (the +- 1 accounts for non divisor cases that need one more/less member)
        adjusted_min_size = self._min_num_players_per_group
        adjusted_max_size = self._max_num_players_per_group
        if adjusted_min_size == adjusted_max_size and num_of_agents % adjusted_max_size != 0:
            adjusted_min_size = adjusted_min_size
            adjusted_max_size = adjusted_max_size + (self._min_num_players_per_group - 1)

        # initialize all coalitions
        for coalition in range(num_of_coalitions - 1, 0, -1):
            group = self._get_group_from_bit_format(coalition)
            group_in_ids = self._convert_from_byte_to_ids(group)

            curr_quality = 0.0
            group_size = len(group)

            # calculate the profile and characteristics only for groups in the range defined
            if adjusted_min_size <= group_size <= adjusted_max_size:
                # generate profile as average of the preferences estimates
                profile = self._interactions_profile_template.generate_copy().reset()
                for curr_player in group_in_ids:
                    preferences = self._player_model_bridge.get_player_preferences_est(curr_player)
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / group_size)

                # calculate quality and average state
                curr_avg_characteristics = PlayerCharacteristics()
                curr_avg_characteristics.reset()
                for i in range(group_size):
                    curr_state = players_curr_state[group_in_ids[i]]
                    curr_state.profile = profile

                    curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                    curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                curr_quality += self.__quality_eval_alg.evaluate(profile, group_in_ids)

                diversity_value_alg = DiversityQualityEvalAlg(self._player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group_in_ids)
                curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                    personalities)

                self.__coalitions_avg_characteristics[coalition] = curr_avg_characteristics
                self.__coalitions_profiles[coalition] = profile

            self.__coalitions_values[coalition] = curr_quality

    # function to compute best profile for group according to each players preferences about the task
    # def find_best_profile_for_group(self, group_in_ids):
    #     group_size = len(group_in_ids)
    #     best_profile = self.interactions_profile_template.generate_copy().reset()
    #     tasks = self.task_model_bridge.getAllTasksIds()
    #
    #     for playerId in group_in_ids:
    #         best_quality = -1
    #         best_task_id = -1
    #         for taskId in tasks:
    #             curr_quality = self.quality_eval_alg.predictTasks(taskId, playerId)
    #             if curr_quality > best_quality:
    #                 best_quality = curr_quality
    #                 best_task_id = taskId
    #
    #         task_profile = self.task_model_bridge.get_task_interactions_profile(best_task_id)
    #         for dim in best_profile.dimensions:
    #             best_profile += task_profile.dimensions[dim] / group_size
    #     return best_profile

    def organize(self):
        self.player_ids = self._player_model_bridge.get_all_player_ids()
        for i in range(len(self.player_ids)):
            self.player_ids[i] = str(self.player_ids[i])
        num_players = len(self.player_ids)

        self.__coalitions_profiles = numpy.empty(1 << num_players, dtype=InteractionsProfile)
        self.__coalitions_avg_characteristics = numpy.empty(1 << num_players, dtype=PlayerCharacteristics)
        self.__coalitions_values = numpy.empty(1 << num_players)

        # re-estimate preferences
        self.__pref_est_alg.update_estimates()

        # compute the value for every valid coalition before execution
        self.__compute_all_coalitions_values()
        required_joint_players_in_bits, required_sep_players_in_bits = self._compute_coalitions_restrictions()

        best_cs_in_bits = gs.odpip(num_players, self._min_num_players_per_group,
                                   self._max_num_players_per_group, self.__coalitions_values.tolist(),
                                   required_joint_players_in_bits, required_sep_players_in_bits)
        best_cs_in_bytes = self._convert_set_of_combinations_from_bit_format(best_cs_in_bits)

        del best_cs_in_bits
        gc.collect()
        return self._convert_cs_bytes_to_dict(best_cs_in_bytes, self.__coalitions_profiles,
                                              self.__coalitions_avg_characteristics)


# uses the C++ solver for efficiency
class CLinkConfigsGenAlg(ConfigsGenAlg):
    # private members
    __quality_eval_alg = None
    __pref_est_alg = None

    __coalitions_profiles = None
    __coalitions_avg_characteristics = None
    __coalitions_values = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 pref_est_alg,
                 task_model_bridge=None,
                 preferred_num_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None):
        super().__init__(player_model_bridge,
                         interactions_profile_template,
                         task_model_bridge,
                         preferred_num_players_per_group,
                         min_num_players_per_group,
                         max_num_players_per_group)

        self.quality_eval_alg = quality_eval_alg
        self.pers_est_alg = pref_est_alg

        self.coalitions_profiles = []
        self.coalitions_avg_characteristics = []
        self.coalitions_values = []

    def __compute_all_coalitions_values(self):
        num_of_agents = len(self.player_ids)
        num_of_coalitions = 1 << num_of_agents

        players_curr_state = {}
        for player in self.player_ids:
            players_curr_state[player] = self._player_model_bridge.get_player_curr_state(player)

        # (the +- 1 accounts for non divisor cases that need one more/less member)
        adjusted_min_size = self._min_num_players_per_group
        adjusted_max_size = self._max_num_players_per_group
        if adjusted_min_size == adjusted_max_size and num_of_agents % adjusted_max_size != 0:
            adjusted_min_size = adjusted_min_size
            adjusted_max_size = adjusted_max_size + (self._min_num_players_per_group - 1)

        # initialize all coalitions
        for coalition in range(num_of_coalitions - 1, 0, -1):
            group = self._get_group_from_bit_format(coalition)
            group_in_ids = self._convert_from_byte_to_ids(group)

            curr_quality = 0.0
            group_size = len(group)

            # calculate the profile and characteristics only for groups in the range defined
            if adjusted_min_size <= group_size <= adjusted_max_size:
                # generate profile as average of the preferences estimates
                profile = self._interactions_profile_template.generate_copy().reset()
                for curr_player in group_in_ids:
                    preferences = self._player_model_bridge.get_player_preferences_est(curr_player)
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / group_size)

                # calculate quality and average state
                curr_avg_characteristics = PlayerCharacteristics()
                curr_avg_characteristics.reset()
                for i in range(group_size):
                    curr_state = players_curr_state[group_in_ids[i]]
                    curr_state.profile = profile

                    curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                    curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                curr_quality += self.quality_eval_alg.evaluate(profile, group_in_ids)

                diversity_value_alg = DiversityQualityEvalAlg(self._player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group_in_ids)
                curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                    personalities)

                self.coalitions_avg_characteristics[coalition] = curr_avg_characteristics
                self.coalitions_profiles[coalition] = profile

            self.coalitions_values[coalition] = curr_quality

    def organize(self):
        self.player_ids = self._player_model_bridge.get_all_player_ids()
        for i in range(len(self.player_ids)):
            self.player_ids[i] = str(self.player_ids[i])
        num_players = len(self.player_ids)

        self.coalitions_profiles = numpy.empty(1 << num_players, dtype=InteractionsProfile)
        self.coalitions_avg_characteristics = numpy.empty(1 << num_players, dtype=PlayerCharacteristics)
        self.coalitions_values = numpy.empty(1 << num_players)

        # re-estimate preferences
        self.pers_est_alg.update_estimates()

        # compute the value for every valid coalition before execution
        self.__compute_all_coalitions_values()

        best_cs_in_bits = (
            gs.clink(num_players, self._min_num_players_per_group, self._max_num_players_per_group,
                     self.coalitions_values.tolist()))
        best_cs_in_bytes = self._convert_set_of_combinations_from_bit_format(best_cs_in_bits)

        del best_cs_in_bits
        gc.collect()
        return self._convert_cs_bytes_to_dict(best_cs_in_bytes, self.__coalitions_profiles,
                                              self.__coalitions_avg_characteristics)
