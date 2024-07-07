import gc
import random
import math
import copy
from sys import breakpointhook
import numpy
import re
from abc import ABC, abstractmethod

import GIMMESolver as gs
from ctypes import *

from ..InteractionsProfile import InteractionsProfile
from ..PlayerStructs import *
from ..AlgDefStructs.QualityEvalAlg import *


class ConfigsGenAlg(ABC):

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 task_model_bridge=None,
                 preferred_number_of_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):

        self.group_size_freqs = {}
        self.config_size_freqs = {}

        self.joint_players_constraints = []
        self.separated_players_constraints = []
        self.all_constraints = []

        min_num_players_per_group = 2 if min_num_players_per_group is None else min_num_players_per_group
        max_num_players_per_group = 5 if max_num_players_per_group is None else max_num_players_per_group

        if min_num_players_per_group > max_num_players_per_group:
            raise ValueError('The min number of players per group cannot be higher than the max!')

        if preferred_number_of_players_per_group is None:
            self.max_num_players_per_group = max_num_players_per_group
            self.min_num_players_per_group = min_num_players_per_group
        else:
            self.max_num_players_per_group = preferred_number_of_players_per_group
            self.min_num_players_per_group = preferred_number_of_players_per_group

        self.player_model_bridge = player_model_bridge
        self.task_model_bridge = task_model_bridge
        self.interactions_profile_template = interactions_profile_template

        joint_player_constraints = self.from_string_constraint_to_list(joint_player_constraints)
        separated_player_constraints = self.from_string_constraint_to_list(separated_player_constraints)

        for i in range(len(joint_player_constraints)):
            if joint_player_constraints[i] == ['']:
                continue
            self.add_joint_players_constraints(joint_player_constraints[i])

        for i in range(len(separated_player_constraints)):
            if separated_player_constraints[i] == ['']:
                continue
            self.add_separated_players_constraints(separated_player_constraints[i])

        self.completion_percentage = 0.0

    def init(self):
        self.group_size_freqs = {}
        self.config_size_freqs = {}
        return self

    def reset(self):
        return self.init()

    def random_config_generator(self, player_ids, min_num_groups, max_num_groups):

        returned_config = []
        player_joint_requirements = {}
        player_separated_requirements = {}
        if self.joint_players_constraints != []:
            for player_id in player_ids:
                player_joint_requirements[str(player_id)] = []

            for constraint in self.joint_players_constraints:
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

        if self.separated_players_constraints != []:
            for player_id in player_ids:
                player_separated_requirements[str(player_id)] = []

            for constraint in self.separated_players_constraints:
                for i in range(len(constraint)):

                    for j in range(len(constraint)):
                        if constraint[i] == constraint[j]:
                            continue

                        player_separated_requirements[constraint[i]].append(constraint[j])

        if len(player_ids) < self.min_num_players_per_group:
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
            for p in range(self.min_num_players_per_group):
                curr_player_index = random.randint(0, len(players_without_group) - 1)

                curr_player_id = players_without_group[curr_player_index]
                curr_group.append(curr_player_id)
                del players_without_group[curr_player_index]

            if ((player_separated_requirements != {} or player_joint_requirements != {}) and
                    len(players_without_group) > 0):
                self.verify_coalition_validity(curr_group, player_joint_requirements, player_separated_requirements,
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
            while len(curr_group) > (self.max_num_players_per_group - 1):
                if len(available_groups) < 1:
                    curr_group = random.choice(returned_config)
                    break
                curr_group = random.choice(available_groups)
                available_groups.remove(curr_group)

            curr_group.append(curr_player_id)

            del players_without_group[curr_player_index]
            players_without_group_size = len(players_without_group)

        return returned_config

    def verify_coalition_validity(self, config, player_joint_requirements, player_separated_requirements,
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

    def from_string_constraint_to_list(self, constraints):
        constraints = constraints.split(';')
        for i in range(len(constraints)):
            constraints[i] = re.sub('[^A-Za-z0-9,_]+', '', constraints[i]).split(',')
        return constraints

    def add_joint_players_constraints(self, players):
        self.joint_players_constraints.append(players)
        self.all_constraints.append({"players": players, "type": "JOIN"})

    def add_separated_players_constraints(self, players):
        self.separated_players_constraints.append(players)
        self.all_constraints.append({"players": players, "type": "SEPARATE"})

    def reset_players_constraints(self):
        self.joint_players_constraints = []
        self.separated_players_constraints = []
        self.all_constraints = []

    def get_player_constraints(self):
        return self.all_constraints

    @abstractmethod
    def organize(self):
        pass

    def update_metrics(self, groups):
        # kind of suboptimal, but guarantees encapsulation
        if (self.config_size_freqs.get(len(groups))):
            self.config_size_freqs[len(groups)] += 1
        else:
            self.config_size_freqs[len(groups)] = 1

        for group in groups:
            if (self.config_size_freqs.get(len(group))):
                self.config_size_freqs[len(group)] += 1
            else:
                self.config_size_freqs[len(group)] = 1

    def get_completion_percentage(self):
        return self.completion_percentage


class RandomConfigsGenAlg(ConfigsGenAlg):

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 preferred_number_of_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):
        super().__init__(
            player_model_bridge=player_model_bridge,
            interactions_profile_template=interactions_profile_template,
            preferred_number_of_players_per_group=preferred_number_of_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

    def organize(self):
        player_ids = self.player_model_bridge.get_all_player_ids()
        min_num_groups = math.ceil(len(player_ids) / self.max_num_players_per_group)
        max_num_groups = math.floor(len(player_ids) / self.min_num_players_per_group)

        new_config_profiles = []
        new_avg_characteristics = []
        new_groups = self.random_config_generator(player_ids, min_num_groups, max_num_groups)
        new_config_size = len(new_groups)
        # generate profiles
        for groupI in range(new_config_size):
            group = new_groups[groupI]
            group_size = len(group)

            profile = self.interactions_profile_template.generate_copy().randomize()
            new_config_profiles.append(profile)

            curr_avg_characteristics = PlayerCharacteristics().reset()
            for currPlayer in group:
                curr_state = self.player_model_bridge.get_player_curr_state(currPlayer)
                curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

            diversity_value_alg = DiversityQualityEvalAlg(self.player_model_bridge, 0)
            personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
            curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(personalities)

            new_avg_characteristics.append(curr_avg_characteristics)

            self.completion_percentage = groupI / new_config_size

        self.update_metrics(new_groups)
        return {"groups": new_groups, "profiles": new_config_profiles, "avgCharacteristics": new_avg_characteristics}


class PureRandomSearchConfigsGenAlg(ConfigsGenAlg):

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 pers_est_alg,
                 number_of_config_choices=None,
                 preferred_number_of_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):

        super().__init__(
            player_model_bridge=player_model_bridge,
            interactions_profile_template=interactions_profile_template,
            preferred_number_of_players_per_group=preferred_number_of_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

        self.qualityEvalAlg = quality_eval_alg
        self.persEstAlg = pers_est_alg
        self.numberOfConfigChoices = 100 if number_of_config_choices is None else number_of_config_choices

    def organize(self):
        player_ids = self.player_model_bridge.get_all_player_ids()
        min_num_groups = math.ceil(len(player_ids) / self.max_num_players_per_group)
        max_num_groups = math.floor(len(player_ids) / self.min_num_players_per_group)

        curr_max_quality = -float("inf")
        best_groups = []
        best_config_profiles = []
        best_avg_characteristics = []

        # estimate preferences
        player_pref_estimates = self.persEstAlg.update_estimates()

        # generate several random groups, calculate their fitness and select the best one
        for i in range(self.numberOfConfigChoices):

            # generate several random groups
            new_groups = self.random_config_generator(player_ids, min_num_groups, max_num_groups)
            new_config_size = len(new_groups)
            curr_quality = 0.0
            new_config_profiles = []
            new_avg_characteristics = []

            # generate profiles
            for groupI in range(new_config_size):
                group = new_groups[groupI]
                group_size = len(group)

                # generate profile as average of the preferences estimates
                profile = self.interactions_profile_template.generate_copy().reset()

                for currPlayer in group:
                    preferences = player_pref_estimates[currPlayer]
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / group_size)

                new_config_profiles.append(profile)

                # calculate quality and average state
                curr_avg_characteristics = PlayerCharacteristics()
                curr_avg_characteristics.reset()
                for i in range(len(group)):
                    curr_state = self.player_model_bridge.get_player_curr_state(group[i])
                    curr_state.profile = profile

                    curr_avg_characteristics.ability += curr_state.characteristics.ability / group_size
                    curr_avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                curr_quality += self.qualityEvalAlg.evaluate(profile, group)

                diversity_value_alg = DiversityQualityEvalAlg(self.player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
                curr_avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(
                    personalities)

                new_avg_characteristics.append(curr_avg_characteristics)

            if curr_quality > curr_max_quality:
                best_groups = new_groups
                best_config_profiles = new_config_profiles
                best_avg_characteristics = new_avg_characteristics
                curr_max_quality = curr_quality

            self.completion_percentage = i / self.numberOfConfigChoices

        self.update_metrics(best_groups)

        return {"groups": best_groups, "profiles": best_config_profiles, "avgCharacteristics": best_avg_characteristics}


from deap import base, creator, tools, algorithms
from collections import *


class EvolutionaryConfigsGenAlg(ConfigsGenAlg):
    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 pers_est_alg,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 preferred_number_of_players_per_group=None,

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
            preferred_number_of_players_per_group=preferred_number_of_players_per_group,
            min_num_players_per_group=min_num_players_per_group,
            max_num_players_per_group=max_num_players_per_group,
            joint_player_constraints=joint_player_constraints,
            separated_player_constraints=separated_player_constraints)

        self.quality_eval_alg = quality_eval_alg
        self.pers_est_alg = pers_est_alg

        self.initial_population_size = 100 if initial_population_size is None else initial_population_size

        self.num_evolutions_per_iteration = 500 if num_evolutions_per_iteration is None \
            else num_evolutions_per_iteration

        self.prob_mut = 0.2 if prob_mut is None else prob_mut
        self.prob_cross = 0.7 if prob_cross is None else prob_cross

        self.prob_mut_config = 0.2 if prob_mut_config is None else prob_mut_config
        self.prob_mut_profiles = 0.2 if prob_mut_profiles is None else prob_mut_profiles

        self.num_children_per_iteration = 5 if num_children_per_iteration is None else num_children_per_iteration
        self.num_survivors = 5 if num_survivors is None else num_survivors

        self.quality_eval_alg = KNNRegQualityEvalAlg(player_model_bridge=player_model_bridge, k=5) \
            if quality_eval_alg is None else quality_eval_alg

        self.player_ids = self.player_model_bridge.get_all_player_ids()
        self.min_num_groups = math.ceil(len(self.player_ids) / self.max_num_players_per_group)
        self.max_num_groups = math.floor(len(self.player_ids) / self.min_num_players_per_group)

        self.searchID = str(id(self))

        fitness_func_id = "FitnessMax" + self.searchID
        individual_id = "Individual" + self.searchID

        creator.create(fitness_func_id, base.Fitness, weights=(1.0,))
        creator.create(individual_id, list, fitness=getattr(creator, fitness_func_id))

        # # conv test
        # creator.create(fitness_func_id, base.Fitness, weights=(-1.0,))
        # creator.create(individual_id, list, fitness=getattr(creator, fitness_func_id))

        self.toolbox = base.Toolbox()

        self.toolbox.register("indices", self.random_individual_generator,
                              self.player_ids, self.min_num_groups, self.max_num_groups)

        self.toolbox.register("individual", tools.initIterate, getattr(creator, individual_id),
                              self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.cx_op = "order" if cx_op is None else cx_op
        if self.cx_op == "order":
            self.toolbox.register("mate", self.cx_gimme_order)
        else:
            self.toolbox.register("mate", self.cx_gimme_simple)

        self.toolbox.register("mutate", self.mut_gimme, pGIPs=self.prob_mut_profiles, pConfig=self.prob_mut_config)

        # self.toolbox.register("select", tools.selRoulette)
        # self.toolbox.register("select", tools.selBest, k=self.numFitSurvivors)
        self.toolbox.register("select", self.sel_gimme)

        # self.toolbox.register("evaluate", self.calcFitness_convergenceTest)
        self.toolbox.register("evaluate", self.calc_fitness)

        self.reset_gen_alg()

    def reset_gen_alg(self):

        if hasattr(self, "pop"):
            del self.pop
        if hasattr(self, "hof"):
            del self.hof

        self.pop = self.toolbox.population(n=self.initial_population_size)
        self.hof = tools.HallOfFame(1)

    def random_individual_generator(self, player_ids, min_num_groups, max_num_groups):
        groups = self.random_config_generator(player_ids, min_num_groups, max_num_groups)
        profs = [self.interactions_profile_template.randomized() for i in range(len(groups))]
        return [groups, profs]

    def cx_gimme_order(self, ind1, ind2):

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
            ind1[1][i] = self.interactions_profile_template.unflattened(new_profiles[0])
            ind2[1][i] = self.interactions_profile_template.unflattened(new_profiles[1])

        del ind1.fitness.values
        del ind2.fitness.values

        return ind1, ind2

    def cx_gimme_simple(self, ind1, ind2):

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
            ind[1][rand_i1] = self.interactions_profile_template.unflattened(new_profiles_gip[0])

            ind[0][rand_i2] = new_profiles_config[1]
            ind[1][rand_i2] = self.interactions_profile_template.unflattened(new_profiles_gip[1])

        del ind1.fitness.values
        del ind2.fitness.values

        return ind1, ind2

    def mut_gimme(self, individual, p_profiles, p_configs):

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

    def reset(self):
        super().reset()
        self.reset_gen_alg()

    def calc_fitness_convergence_test(self, individual):
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

        total_fitness = total_fitness + 1.0 # helps selection (otherwise Pchoice would always be 0)
        individual.fitness.values = total_fitness,
        return total_fitness, # must return a tuple

    def sel_gimme(self, individuals, k, fit_attr="fitness"):
        return tools.selBest(individuals, k, fit_attr)

    def calc_fitness(self, individual):
        config = individual[0]
        profiles = individual[1]

        total_fitness = 0.0

        len_config = len(config)

        all_constrains_satisfied = True
        for groupI in range(len_config):

            group = config[groupI]
            profile = profiles[groupI]

            for constraint in self.joint_players_constraints:
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

            for constraint in self.separated_players_constraints:
                must_be_sep = False
                for player in constraint:
                    if player in group:
                        if must_be_sep:
                            all_constrains_satisfied = False
                            break
                        must_be_sep = True

                if all_constrains_satisfied == False:
                    break

            total_fitness += self.quality_eval_alg.evaluate(profile, group)

        total_fitness = total_fitness + 1.0  # helps selection (otherwise Pchoice would always be 0)
        if all_constrains_satisfied:
            total_fitness += 1000
        individual.fitness.values = total_fitness,

        return total_fitness,  # must return a tuple

    def organize(self):
        self.reset_gen_alg()
        # if isinstance(self.quality_eval_alg, TabQualityEvalAlg):
        #     self.playerPrefEstimates = self.pers_est_alg.update_estimates()

        algorithms.eaMuCommaLambda(
            population=self.pop,
            toolbox=self.toolbox,

            lambda_=self.num_children_per_iteration,
            mu=self.num_survivors,

            cxpb=self.prob_cross,
            mutpb=self.prob_mut,

            ngen=self.num_evolutions_per_iteration,

            halloffame=self.hof,
            verbose=False
        )

        self.completion_percentage = len(tools.Logbook()) / self.num_evolutions_per_iteration

        best_groups = self.hof[0][0]
        best_profiles = self.hof[0][1]

        avg_characteristics_array = []
        for group in best_groups:
            group_size = len(group)
            avg_characteristics = PlayerCharacteristics()
            for currPlayer in group:
                curr_state = self.player_model_bridge.get_player_curr_state(currPlayer)
                avg_characteristics.ability += curr_state.characteristics.ability / group_size
                avg_characteristics.engagement += curr_state.characteristics.engagement / group_size

                diversity_value_alg = DiversityQualityEvalAlg(self.player_model_bridge, 0)
                personalities = diversity_value_alg.get_personalities_list_from_player_ids(group)
                avg_characteristics.group_diversity = diversity_value_alg.get_team_personality_diveristy(personalities)

            avg_characteristics_array.append(avg_characteristics)

        return {"groups": best_groups, "profiles": best_profiles, "avgCharacteristics": avg_characteristics_array}


# uses the C++ solver for efficiency
class ODPIPConfigsGenAlg(ConfigsGenAlg):
    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 qualityEvalAlg,
                 persEstAlg,
                 task_model_bridge=None,
                 preferred_number_of_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None,
                 joint_player_constraints="",
                 separated_player_constraints=""):

        super().__init__(player_model_bridge,
                         interactions_profile_template,
                         task_model_bridge,
                         preferred_number_of_players_per_group,
                         min_num_players_per_group,
                         max_num_players_per_group,
                         joint_player_constraints=joint_player_constraints,
                         separated_player_constraints=separated_player_constraints)

        self.qualityEvalAlg = qualityEvalAlg
        self.persEstAlg = persEstAlg

        self.coalitionsProfiles = []
        self.coalitionsAvgCharacteristics = []
        self.coalitionsValues = []

        self.playerIds = []

        self.playerPrefEstimates = {}

    def getCoalitionInByteFormatValue(self, coalitionInByteFormat):
        coalitionInBitFormat = self.convertCoalitionFromByteToBitFormat(coalitionInByteFormat,
                                                                        len(coalitionInByteFormat))
        return self.f[coalitionInBitFormat]

    def getCoalitionStructureInByteFormatValue(self, coalitionStructure):
        valueOfCS = 0
        for i in range(len(coalitionStructure)):
            valueOfCS += self.getCoalitionInByteFormatValue(coalitionStructure[i])

        return valueOfCS

    def convertCoalitionFromByteToBitFormat(self, coalitionInByteFormat, coalitionSize):
        coalitionInBitFormat = 0

        for i in range(coalitionSize):
            coalitionInBitFormat += 1 << (coalitionInByteFormat[i] - 1)

        return coalitionInBitFormat

    # convert group in bit format to group with the player ids
    def getGroupFromBitFormat(self, coalition):
        group = []
        tempCoalition = coalition
        playerNumber = 0
        while tempCoalition != 0:
            if tempCoalition & 1:
                group.append(playerNumber + 1)

            playerNumber += 1
            tempCoalition >>= 1

        return group

    def convertFromByteToIds(self, coalition):
        group = []

        for agent in coalition:
            group.append(self.playerIds[agent - 1])

        return group

    def convertFromIdsToBytes(self, coalition):
        group = []

        for agent in coalition:
            for i in range(len(self.playerIds)):
                if self.playerIds[i] == agent:
                    group.append(i + 1)

        return group

    def getSizeOfCombinationInBitFormat(self, combinationInBitFormat):
        return bin(combinationInBitFormat).count("1")

    def convertSetOfCombinationsFromBitFormat(self, setOfCombinationsInBitFormat):
        setOfCombinationsInByteFormat = numpy.empty(len(setOfCombinationsInBitFormat), dtype=list)
        for i in range(len(setOfCombinationsInBitFormat)):
            setOfCombinationsInByteFormat[i] = self.getGroupFromBitFormat(setOfCombinationsInBitFormat[i])
        return setOfCombinationsInByteFormat

    def computeAllCoalitionsValues(self):
        numOfAgents = len(self.playerIds)
        numOfCoalitions = 1 << (numOfAgents)

        playersCurrState = {}
        for player in self.playerIds:
            playersCurrState[player] = self.player_model_bridge.get_player_curr_state(player)

        # (the +- 1 accounts for non divisor cases that need one more/less member)
        adjustedMinSize = self.min_num_players_per_group
        adjustedMaxSize = self.max_num_players_per_group
        if (adjustedMinSize == adjustedMaxSize and numOfAgents % adjustedMaxSize != 0):
            adjustedMinSize = adjustedMinSize
            adjustedMaxSize = adjustedMaxSize + (self.min_num_players_per_group - 1)

        # initialize all coalitions
        for coalition in range(numOfCoalitions - 1, 0, -1):
            group = self.getGroupFromBitFormat(coalition)
            groupInIds = self.convertFromByteToIds(group)

            currQuality = 0.0
            groupSize = len(group)

            # calculate the profile and characteristics only for groups in the range defined
            if groupSize >= adjustedMinSize and groupSize <= adjustedMaxSize:
                # generate profile as average of the preferences estimates
                profile = self.interactions_profile_template.generate_copy().reset()

                # if (self.qualityEvalAlg.isTabular()):
                # 	profile = self.findBestProfileForGroup(groupInIds)

                # else:
                for currPlayer in groupInIds:
                    preferences = self.playerPrefEstimates[currPlayer]
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / groupSize)

                # calculate fitness and average state
                currAvgCharacteristics = PlayerCharacteristics()
                currAvgCharacteristics.reset()
                for i in range(groupSize):
                    currState = playersCurrState[groupInIds[i]]
                    currState.profile = profile

                    currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
                    currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize

                currQuality += self.qualityEvalAlg.evaluate(profile, groupInIds)

                diversityValueAlg = DiversityQualityEvalAlg(self.player_model_bridge, 0)
                personalities = diversityValueAlg.get_personalities_list_from_player_ids(groupInIds)
                currAvgCharacteristics.group_diversity = diversityValueAlg.get_team_personality_diveristy(personalities)

                self.coalitionsAvgCharacteristics[coalition] = currAvgCharacteristics
                self.coalitionsProfiles[coalition] = profile

            self.coalitionsValues[coalition] = currQuality

    def computeCoalitionsRestrictions(self):
        jointPlayersConstraintInBitFormat = self.joint_players_constraints[:]
        separatedPlayersConstraintInBitFormat = self.separated_players_constraints[:]

        for i in range(len(jointPlayersConstraintInBitFormat)):
            jointPlayersConstraintInBitFormat[i] = self.convertFromIdsToBytes(jointPlayersConstraintInBitFormat[i])
            jointPlayersConstraintInBitFormat[i] = self.convertCoalitionFromByteToBitFormat(
                jointPlayersConstraintInBitFormat[i], len(jointPlayersConstraintInBitFormat[i]))

        for i in range(len(separatedPlayersConstraintInBitFormat)):
            separatedPlayersConstraintInBitFormat[i] = self.convertFromIdsToBytes(
                separatedPlayersConstraintInBitFormat[i])
            separatedPlayersConstraintInBitFormat[i] = self.convertCoalitionFromByteToBitFormat(
                separatedPlayersConstraintInBitFormat[i], len(separatedPlayersConstraintInBitFormat[i]))

        return jointPlayersConstraintInBitFormat, separatedPlayersConstraintInBitFormat

    def results(self, cSInByteFormat):
        bestGroups = []
        bestGroupsInBitFormat = []
        bestConfigProfiles = []
        avgCharacteristicsArray = []

        for coalition in cSInByteFormat:
            bestGroups.append(self.convertFromByteToIds(coalition))
            bestGroupsInBitFormat.append(self.convertCoalitionFromByteToBitFormat(coalition, len(coalition)))

        for group in bestGroupsInBitFormat:
            bestConfigProfiles.append(self.coalitionsProfiles[group])
            avgCharacteristicsArray.append(self.coalitionsAvgCharacteristics[group])

        return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": avgCharacteristicsArray}

    # function to compute best profile for group according to each players preferences about the task
    def findBestProfileForGroup(self, groupInIds):
        groupSize = len(groupInIds)
        bestProfile = self.interactions_profile_template.generate_copy().reset()

        tasks = self.task_model_bridge.getAllTasksIds()

        for playerId in groupInIds:
            bestQuality = -1
            bestTaskId = -1

            for taskId in tasks:
                currQuality = self.qualityEvalAlg.predictTasks(taskId, playerId)

                if currQuality > bestQuality:
                    bestQuality = currQuality
                    bestTaskId = taskId

            taskProfile = self.task_model_bridge.get_task_interactions_profile(bestTaskId)
            for dim in bestProfile.dimensions:
                bestProfile += taskProfile.dimensions[dim] / groupSize

        return bestProfile

    def organize(self):
        self.playerIds = self.player_model_bridge.get_all_player_ids()
        for i in range(len(self.playerIds)):
            self.playerIds[i] = str(self.playerIds[i])
        self.numPlayers = len(self.playerIds)

        self.coalitionsProfiles = numpy.empty(1 << self.numPlayers, dtype=InteractionsProfile)
        self.coalitionsAvgCharacteristics = numpy.empty(1 << self.numPlayers, dtype=PlayerCharacteristics)
        self.coalitionsValues = numpy.empty(1 << self.numPlayers)

        # estimate preferences
        self.playerPrefEstimates = self.persEstAlg.update_estimates()

        # initialization(compute the value for every coalition between min and max number of players)
        self.computeAllCoalitionsValues()
        requiredJointPlayersInBitFormat, restrictedPlayersToJoinInBitFormat = self.computeCoalitionsRestrictions()

        bestCSFound_bitFormat = gs.odpip(self.numPlayers, self.min_num_players_per_group,
                                         self.max_num_players_per_group, self.coalitionsValues.tolist(),
                                         requiredJointPlayersInBitFormat, restrictedPlayersToJoinInBitFormat)
        bestCSFound_byteFormat = self.convertSetOfCombinationsFromBitFormat(bestCSFound_bitFormat)

        del bestCSFound_bitFormat

        gc.collect()
        return self.results(bestCSFound_byteFormat)


# uses the C++ solver for efficiency
class CLinkConfigsGenAlg(ConfigsGenAlg):
    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 qualityEvalAlg,
                 persEstAlg,
                 task_model_bridge=None,
                 preferred_number_of_players_per_group=None,
                 min_num_players_per_group=None,
                 max_num_players_per_group=None):

        super().__init__(player_model_bridge,
                         interactions_profile_template,
                         task_model_bridge,
                         preferred_number_of_players_per_group,
                         min_num_players_per_group,
                         max_num_players_per_group)

        self.qualityEvalAlg = qualityEvalAlg
        self.persEstAlg = persEstAlg

        self.coalitionsProfiles = []
        self.coalitionsAvgCharacteristics = []
        self.coalitionsValues = []

        self.playerIds = []

        self.playerPrefEstimates = {}

    def getCoalitionInByteFormatValue(self, coalitionInByteFormat):
        coalitionInBitFormat = self.convertCoalitionFromByteToBitFormat(coalitionInByteFormat,
                                                                        len(coalitionInByteFormat))
        return self.f[coalitionInBitFormat]

    def getCoalitionStructureInByteFormatValue(self, coalitionStructure):
        valueOfCS = 0
        for i in range(len(coalitionStructure)):
            valueOfCS += self.getCoalitionInByteFormatValue(coalitionStructure[i])

        return valueOfCS

    def convertCoalitionFromByteToBitFormat(self, coalitionInByteFormat, coalitionSize):
        coalitionInBitFormat = 0

        for i in range(coalitionSize):
            coalitionInBitFormat += 1 << (coalitionInByteFormat[i] - 1)

        return coalitionInBitFormat

    # convert group in bit format to group with the player ids
    def getGroupFromBitFormat(self, coalition):
        group = []
        tempCoalition = coalition
        playerNumber = 0
        while tempCoalition != 0:
            if tempCoalition & 1:
                group.append(playerNumber + 1)

            playerNumber += 1
            tempCoalition >>= 1

        return group

    def convertFromByteToIds(self, coalition):
        group = []

        for agent in coalition:
            group.append(self.playerIds[agent - 1])

        return group

    def getSizeOfCombinationInBitFormat(self, combinationInBitFormat):
        return bin(combinationInBitFormat).count("1")

    def convertSetOfCombinationsFromBitFormat(self, setOfCombinationsInBitFormat):
        setOfCombinationsInByteFormat = numpy.empty(len(setOfCombinationsInBitFormat), dtype=list)
        for i in range(len(setOfCombinationsInBitFormat)):
            setOfCombinationsInByteFormat[i] = self.getGroupFromBitFormat(setOfCombinationsInBitFormat[i])

        return setOfCombinationsInByteFormat

    def computeAllCoalitionsValues(self):
        numOfAgents = len(self.playerIds)
        numOfCoalitions = 1 << (numOfAgents)

        playersCurrState = {}
        for player in self.playerIds:
            playersCurrState[player] = self.player_model_bridge.get_player_curr_state(player)

        # (the +- 1 accounts for non divisor cases that need one more/less member)
        adjustedMinSize = self.min_num_players_per_group
        adjustedMaxSize = self.max_num_players_per_group
        if (adjustedMinSize == adjustedMaxSize and numOfAgents % adjustedMaxSize != 0):
            adjustedMinSize = adjustedMinSize
            adjustedMaxSize = adjustedMaxSize + (self.min_num_players_per_group - 1)

        # initialize all coalitions
        for coalition in range(numOfCoalitions - 1, 0, -1):
            group = self.getGroupFromBitFormat(coalition)
            groupInIds = self.convertFromByteToIds(group)

            currQuality = 0.0
            groupSize = len(group)

            # calculate the profile and characteristics only for groups in the range defined
            if groupSize >= adjustedMinSize and groupSize <= adjustedMaxSize:
                # generate profile as average of the preferences estimates
                profile = self.interactions_profile_template.generate_copy().reset()

                for currPlayer in groupInIds:
                    preferences = self.playerPrefEstimates[currPlayer]
                    for dim in profile.dimensions:
                        profile.dimensions[dim] += (preferences.dimensions[dim] / groupSize)

                # calculate average state
                currAvgCharacteristics = PlayerCharacteristics()
                currAvgCharacteristics.reset()
                for i in range(groupSize):
                    currState = playersCurrState[groupInIds[i]]
                    currState.profile = profile

                    currAvgCharacteristics.ability += currState.characteristics.ability / groupSize
                    currAvgCharacteristics.engagement += currState.characteristics.engagement / groupSize

                currQuality += self.qualityEvalAlg.evaluate(profile, groupInIds)

                diversityValueAlg = DiversityQualityEvalAlg(self.player_model_bridge, 0)
                personalities = diversityValueAlg.get_personalities_list_from_player_ids(groupInIds)
                currAvgCharacteristics.group_diversity = diversityValueAlg.get_team_personality_diveristy(personalities)

                self.coalitionsAvgCharacteristics[coalition] = currAvgCharacteristics
                self.coalitionsProfiles[coalition] = profile

            self.coalitionsValues[coalition] = currQuality

    def results(self, cSInByteFormat):
        bestGroups = []
        bestGroupsInBitFormat = []
        bestConfigProfiles = []
        avgCharacteristicsArray = []
        for coalition in cSInByteFormat:
            bestGroups.append(self.convertFromByteToIds(coalition))
            bestGroupsInBitFormat.append(self.convertCoalitionFromByteToBitFormat(coalition, len(coalition)))

        for group in bestGroupsInBitFormat:
            bestConfigProfiles.append(self.coalitionsProfiles[group])
            avgCharacteristicsArray.append(self.coalitionsAvgCharacteristics[group])

        return {"groups": bestGroups, "profiles": bestConfigProfiles, "avgCharacteristics": avgCharacteristicsArray}

    # function to compute best profile for group according to each players preferences about the task
    def findBestProfileForGroup(self, groupInIds):
        groupSize = len(groupInIds)
        bestProfile = self.interactions_profile_template.generate_copy().reset()

        tasks = self.task_model_bridge.getAllTasksIds()

        for playerId in groupInIds:
            bestQuality = -1
            bestTaskId = -1

            for taskId in tasks:
                currQuality = self.qualityEvalAlg.predictTasks(taskId, playerId)

                if currQuality > bestQuality:
                    bestQuality = currQuality
                    bestTaskId = taskId

            taskProfile = self.task_model_bridge.get_task_interactions_profile(bestTaskId)
            for dim in bestProfile.dimensions:
                bestProfile += taskProfile.dimensions[dim] / groupSize

        return bestProfile

    def organize(self):
        self.playerIds = self.player_model_bridge.get_all_player_ids()
        for i in range(len(self.playerIds)):
            self.playerIds[i] = str(self.playerIds[i])
        self.numPlayers = len(self.playerIds)

        self.coalitionsProfiles = numpy.empty(1 << self.numPlayers, dtype=InteractionsProfile)
        self.coalitionsAvgCharacteristics = numpy.empty(1 << self.numPlayers, dtype=PlayerCharacteristics)
        self.coalitionsValues = numpy.empty(1 << self.numPlayers)

        # estimate preferences
        self.playerPrefEstimates = self.persEstAlg.update_estimates()

        # initialization(compute the value for every coalition between min and max number of players)
        self.computeAllCoalitionsValues()

        bestCSFound_bitFormat = (
            gs.clink(self.numPlayers, self.min_num_players_per_group, self.max_num_players_per_group,
                     self.coalitionsValues.tolist()))
        bestCSFound_byteFormat = self.convertSetOfCombinationsFromBitFormat(bestCSFound_bitFormat)

        del bestCSFound_bitFormat

        gc.collect()

        return self.results(bestCSFound_byteFormat)
