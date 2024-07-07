from abc import ABC, abstractmethod
import copy
import json

from ..PlayerStructs import *


class PlayerDataTrimAlg(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def trimmed_list(self, past_model_incs):
        pass


# ---------------------- KNNRegression stuff ---------------------------
class AgeSortPlayerDataTrimAlg(PlayerDataTrimAlg):
    def __init__(self, max_num_model_elements):
        super().__init__()
        self.max_num_model_elements = max_num_model_elements

    def creation_time_sort(self, elem):
        return elem.creation_time

    def trimmed_list(self, past_model_incs):
        if len(past_model_incs) <= self.max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted = sorted(past_model_incs, key=self.creation_time_sort)
        removed_i = past_model_incs.index(past_model_incs_sorted[0])
        past_model_incs.pop(removed_i)
        return [past_model_incs, [removed_i]]


class QualitySortPlayerDataTrimAlg(PlayerDataTrimAlg):

    def __init__(self, max_num_model_elements, quality_weights=None, acc_state_residue=None):
        super().__init__()
        self.max_num_model_elements = max_num_model_elements
        self.quality_weights = PlayerCharacteristics(ability=0.5,
                                                     engagement=0.5) if quality_weights is None else quality_weights
        self.acc_state_residue = False if acc_state_residue is None else acc_state_residue

    def consider_state_residue(self, acc_state_residue):
        self.acc_state_residue = acc_state_residue

    def state_type_filter(self, element):
        return element.stateType == 0

    def q_sort(self, elem):
        return elem.quality

    def calc_quality(self, state):
        total = self.quality_weights.ability * state.characteristics.ability + self.quality_weights.engagement * state.characteristics.engagement
        return total

    def trimmed_list(self, past_model_incs):
        for modelInc in past_model_incs:
            if modelInc.quality == -1:
                modelInc.quality = self.calc_quality(modelInc)
                if self.acc_state_residue:
                    modelInc.quality += modelInc.stateType

        if len(past_model_incs) <= self.max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted = sorted(past_model_incs, key=self.q_sort)
        removed_i = past_model_incs.index(past_model_incs_sorted[0])
        past_model_incs.pop(removed_i)
        return [past_model_incs, [removed_i]]


class ProximitySortPlayerDataTrimAlg(PlayerDataTrimAlg):

    def __init__(self, max_num_model_elements, epsilon=None, acc_state_residue=None):
        super().__init__()
        self.max_num_model_elements = max_num_model_elements
        self.epsilon = 0.01 if epsilon is None else epsilon
        self.acc_state_residue = False if acc_state_residue is None else acc_state_residue

    def consider_state_residue(self, acc_state_residue):
        self.acc_state_residue = acc_state_residue

    def proximity_sort(self, elem):
        return elem.quality

    def creation_time_sort(self, elem):
        return elem.creation_time

    def trimmed_list(self, past_model_incs):
        if len(past_model_incs) <= self.max_num_model_elements:
            return [past_model_incs, []]

        past_model_incs_sorted_age = sorted(past_model_incs, key=self.creation_time_sort)
        last_data_point = past_model_incs_sorted_age[-1]
        for modelInc in past_model_incs:
            modelInc.quality = last_data_point.profile.sqr_distance_between(modelInc.profile)
            if self.acc_state_residue:
                modelInc.quality += modelInc.stateType

        # check if there is already a close point
        past_model_incs_sorted = sorted(past_model_incs, key=self.proximity_sort)
        # remove the point to be tested
        past_model_incs_sorted.remove(last_data_point)
        closest_point = past_model_incs_sorted[0]

        if (self.acc_state_residue and closest_point.stateType == 0) or closest_point.quality > (
                self.epsilon + closest_point.stateType):
            removed_i = past_model_incs.index(closest_point)
            past_model_incs.pop(removed_i)
        else:
            removed_i = past_model_incs.index(last_data_point)
            past_model_incs.pop(removed_i)

        return [past_model_incs, [removed_i]]
