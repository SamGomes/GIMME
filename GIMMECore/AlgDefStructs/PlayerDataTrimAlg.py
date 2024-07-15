# from abc import ABC, abstractmethod
# import copy
# import json
from ..PlayerStructs import *

class PlayerDataTrimAlg(ABC):
    # protected members
    _max_num_model_elements = None

    def __init__(self, max_num_model_elements):
        self._max_num_model_elements = max_num_model_elements

    @abstractmethod
    def trimmed_list(self, past_model_incs):
        pass


# ---------------------- KNNRegression stuff ---------------------------
class AgeSortPlayerDataTrimAlg(PlayerDataTrimAlg):

    def __init__(self, max_num_model_elements):
        super().__init__(max_num_model_elements)

    def __creation_time_sort(self, elem):
        return elem.creation_time

    def trimmed_list(self, past_model_incs):
        if len(past_model_incs) <= self._max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted = sorted(past_model_incs, key=self.__creation_time_sort)
        removed_i = past_model_incs.index(past_model_incs_sorted[0])
        past_model_incs.pop(removed_i)
        return [past_model_incs, [removed_i]]


class QualitySortPlayerDataTrimAlg(PlayerDataTrimAlg):
    # private members
    __quality_weights = None
    __acc_state_residue = None

    def __init__(self, max_num_model_elements, quality_weights=None, acc_state_residue=None):
        super().__init__(max_num_model_elements)
        self.__quality_weights = PlayerCharacteristics(ability=0.5,
                                                       engagement=0.5) if quality_weights is None else quality_weights
        self.__acc_state_residue = False if acc_state_residue is None else acc_state_residue

    def __state_type_filter(self, elem):
        return elem.type == 0

    def __q_sort(self, elem):
        return elem.quality

    def __calc_quality(self, state):
        total = (self.__quality_weights.ability * state.characteristics.ability +
                 self.__quality_weights.engagement * state.characteristics.engagement)
        return total

    def consider_state_residue(self, acc_state_residue):
        self.__acc_state_residue = acc_state_residue

    def trimmed_list(self, past_model_incs):
        for modelInc in past_model_incs:
            if modelInc.quality == -1:
                modelInc.quality = self.__calc_quality(modelInc)
                if self.__acc_state_residue:
                    modelInc.quality += modelInc.type

        if len(past_model_incs) <= self._max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted = sorted(past_model_incs, key=self.__q_sort)
        removed_i = past_model_incs.index(past_model_incs_sorted[0])
        past_model_incs.pop(removed_i)
        return [past_model_incs, [removed_i]]


class ProximitySortPlayerDataTrimAlg(PlayerDataTrimAlg):
    # private members
    __epsilon = None
    __acc_state_residue = None

    def __init__(self, max_num_model_elements, epsilon=None, acc_state_residue=None):
        super().__init__(max_num_model_elements)
        self.__epsilon = 0.01 if epsilon is None else epsilon
        self.__acc_state_residue = False if acc_state_residue is None else acc_state_residue

    def consider_state_residue(self, acc_state_residue):
        self.__acc_state_residue = acc_state_residue

    def __proximity_sort(self, elem):
        return elem.quality

    def __creation_time_sort(self, elem):
        return elem.creation_time

    def trimmed_list(self, past_model_incs):
        if len(past_model_incs) <= self._max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted_age = sorted(past_model_incs, key=self.__creation_time_sort)
        last_data_point = past_model_incs_sorted_age[-1]
        for modelInc in past_model_incs:
            modelInc.quality = last_data_point.profile.sqr_distance_between(modelInc.profile)
            if self.__acc_state_residue:
                modelInc.quality += modelInc.type

        # check if there is already a close point
        past_model_incs_sorted = sorted(past_model_incs, key=self.__proximity_sort)
        # remove the point to be tested
        past_model_incs_sorted.remove(last_data_point)
        closest_point = past_model_incs_sorted[0]

        if (self.__acc_state_residue and closest_point.type == 0) or closest_point.quality > (
                self.__epsilon + closest_point.type):
            removed_i = past_model_incs.index(closest_point)
            past_model_incs.pop(removed_i)
        else:
            removed_i = past_model_incs.index(last_data_point)
            past_model_incs.pop(removed_i)

        return [past_model_incs, [removed_i]]
