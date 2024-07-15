# import copy
import math
import numpy
import pandas as pd

# from abc import ABC, abstractmethod
from ..PlayerStructs import *


class QualityEvalAlg(ABC):
    # protected members
    _player_model_bridge = None
    _completion_percentage = None

    def __init__(self, player_model_bridge):
        self._player_model_bridge = player_model_bridge
        self._completion_percentage = 0.0

    @abstractmethod
    def evaluate(self, profile, group_player_ids):
        pass

    def get_completion_percentage(self):
        return self._completion_percentage


# ---------------------- Group-Based Quality Evaluation ---------------------------
class GroupQualityEvalAlg(QualityEvalAlg, ABC):

    def __init__(self, player_model_bridge):
        super().__init__(player_model_bridge)


# Personality Diversity
class DiversityQualityEvalAlg(GroupQualityEvalAlg):
    #  Consider the task preferences of students in addition to team diversity.
    #  People with the same personality can still have different preferences
    #  Diversity weight is the value determined by the teacher (0 = aligned, 1 = diverse)

    # private members
    __diversity_weight = None

    def __init__(self, player_model_bridge, diversity_weight):
        super().__init__(player_model_bridge)
        self.__diversity_weight = diversity_weight

    def get_personalities_list_from_player_ids(self, group_player_ids):
        personalities = []  # list of PlayerPersonality objects

        for player_id in group_player_ids:
            personality = self._player_model_bridge.get_player_personality(player_id)
            if personality:
                personalities.append(personality)

        return personalities

    def get_team_personality_diveristy(self, personalities):
        if len(personalities) <= 0:
            return -1

        diversity = -1

        if isinstance(personalities[0], PersonalityMBTI):
            diversity = PersonalityMBTI.get_team_personality_diversity(personalities)

        return diversity

    def evaluate(self, _, group_player_ids):
        # list of PlayerPersonality objects
        personalities = self.get_personalities_list_from_player_ids(group_player_ids)
        diversity = self.get_team_personality_diveristy(personalities)

        # inverse of distance squared
        # lower distance = higher quality
        distance = abs(diversity - self.__diversity_weight)

        if distance == 0.0:
            return 1.0

        # not iterative, so it can jump to comp_percentage = 1
        self.comp_percentage = 1.0
        return 1.0 / (distance * distance)


# ---------------------- Regression-Based Characteristic Functions ---------------------------
class RegQualityEvalAlg(QualityEvalAlg, ABC):
    # protected members
    _quality_weights = None

    def __init__(self, player_model_bridge, quality_weights=None):
        super().__init__(player_model_bridge)
        self._quality_weights = PlayerCharacteristics(ability=0.5,
                                                      engagement=0.5) if quality_weights is None else quality_weights


class KNNRegQualityEvalAlg(RegQualityEvalAlg):
    # private members
    __k = None

    def __init__(self, player_model_bridge, k, quality_weights=None):
        super().__init__(player_model_bridge, quality_weights)
        self.__k = k

    def __calc_quality(self, state):
        return self._quality_weights.ability * state.characteristics.ability + self._quality_weights.engagement * state.characteristics.engagement

    def __dist_sort(self, elem):
        return elem.dist

    def evaluate(self, profile, group_player_ids):
        total_quality = 0
        group_size = len(group_player_ids)
        for player_id in group_player_ids:
            past_model_incs = self._player_model_bridge.get_player_states_data_frame(player_id).get_all_states().copy()
            predicted_state = PlayerState(profile=profile, characteristics=PlayerCharacteristics())

            for modelInc in past_model_incs:
                modelInc.dist = profile.sqr_distance_between(modelInc.profile)

            past_model_incs = sorted(past_model_incs, key=self.__dist_sort)

            number_of_iterations = min(self.__k, len(past_model_incs))
            past_model_incs = past_model_incs[:number_of_iterations]

            triangular_number_of_it = sum(range(number_of_iterations + 1))
            for i in range(number_of_iterations):
                self.comp_percentage = i / number_of_iterations

                curr_state = past_model_incs[i]
                past_characteristics = curr_state.characteristics
                ratio = (number_of_iterations - i) / triangular_number_of_it

                predicted_state.characteristics.ability += past_characteristics.ability * ratio
                predicted_state.characteristics.engagement += past_characteristics.engagement * ratio

            total_quality += self.__calc_quality(predicted_state) / group_size

        return total_quality


# ---------------------- Tabular Characteristic Functions -------------------------------------
class TabQualityEvalAlg(QualityEvalAlg):

    def __init__(self, player_model_bridge):
        super().__init__(player_model_bridge)


class SynergiesTabQualityEvalAlg(TabQualityEvalAlg):
    # private members
    __synergy_matrix = None

    def __init__(self, player_model_bridge, synergy_table_path):
        super().__init__(player_model_bridge)

        temp_table = pd.read_csv(synergy_table_path, sep=",", dtype={'agent_1': object, 'agent_2': object})
        synergy_table = temp_table.pivot_table(values='synergy', index='agent_1', columns='agent_2')

        self.__synergy_matrix = synergy_table.to_numpy()
        self.__synergy_matrix[numpy.isnan(self.__synergy_matrix)] = 0
        self.__synergy_matrix = self.__symmetrize(self.__synergy_matrix)

    def __symmetrize(self, table):
        return table + table.T - numpy.diag(table.diagonal())

    def evaluate(self, _, group_player_ids):
        total_quality = 0
        group_size = len(group_player_ids)
        num_elem_combs = math.comb(group_size, 2)
        for i in range(group_size - 1):
            first_player_id = group_player_ids[i]
            first_player_preferences = self._player_model_bridge.get_player_preferences_est(first_player_id)
            first_player_preferences_in_binary = ''
            for dim in first_player_preferences.dimensions:
                first_player_preferences_in_binary += str(round(first_player_preferences.dimensions[dim]))
            first_player_preferences_index = int(first_player_preferences_in_binary, 2)

            # assumes synergy matrix symmetry
            for j in range(i + 1, len(group_player_ids)):
                second_player_id = group_player_ids[j]
                second_player_preferences = self._player_model_bridge.get_player_preferences_est(second_player_id)
                second_player_preference_in_binary = ''
                for dim in second_player_preferences.dimensions:
                    second_player_preference_in_binary += str(round(second_player_preferences.dimensions[dim]))

                second_player_preferences_index = int(second_player_preference_in_binary, 2)
                total_quality += (self.__synergy_matrix[first_player_preferences_index][second_player_preferences_index]
                                  / num_elem_combs)

        return total_quality
