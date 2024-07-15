# from abc import ABC, abstractmethod
from ..PlayerStructs import *


class PreferencesEstAlg(ABC):
    # protected members
    _player_model_bridge = None

    def __init__(self, player_model_bridge):
        self._player_model_bridge = player_model_bridge

    @abstractmethod
    def update_estimates(self):
        pass


class ExploitationPreferencesEstAlg(PreferencesEstAlg):
    # private members
    __quality_weights = None
    __best_qualities = None

    def __init__(self,
                 player_model_bridge,
                 quality_weights=None):

        super().__init__(player_model_bridge)

        self.__quality_weights = PlayerCharacteristics(ability=0.5,
                                                       engagement=0.5) if quality_weights is None else quality_weights
        self.__best_qualities = {}

    def __calc_quality(self, state):
        return (self.__quality_weights.ability * state.characteristics.ability +
                self.__quality_weights.engagement * state.characteristics.engagement)

    def update_estimates(self):
        player_ids = self._player_model_bridge.get_all_player_ids()
        for playerId in player_ids:
            curr_preferences_quality = self.__best_qualities.get(playerId, 0.0)
            last_data_point = self._player_model_bridge.get_player_curr_state(playerId)
            quality = self.__calc_quality(last_data_point)
            if quality > curr_preferences_quality:
                self.__best_qualities[playerId] = curr_preferences_quality
                self._player_model_bridge.set_player_preferences_est(playerId, last_data_point.profile)


class ExplorationPreferencesEstAlg(PreferencesEstAlg):
    # private members
    __interactions_profile_template = None
    __quality_eval_alg = None
    __num_tested_player_profiles = None

    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 num_tested_player_profiles=None):

        super().__init__(player_model_bridge)

        self.__quality_eval_alg = quality_eval_alg

        self.__num_tested_player_profiles = 100 if num_tested_player_profiles is None else num_tested_player_profiles
        self.__interactions_profile_template = interactions_profile_template

    def update_estimates(self):
        player_ids = self._player_model_bridge.get_all_player_ids()
        for player_id in player_ids:

            curr_preferences_est = self._player_model_bridge.get_player_preferences_est(player_id)
            new_preferences_est = curr_preferences_est
            if curr_preferences_est is not None:
                best_quality = self.__quality_eval_alg.evaluate(curr_preferences_est, [player_id])
            else:
                best_quality = -1

            for i in range(self.__num_tested_player_profiles):
                profile = self.__interactions_profile_template.generate_copy().randomize()
                curr_quality = self.__quality_eval_alg.evaluate(profile, [player_id])
                if curr_quality >= best_quality:
                    best_quality = curr_quality
                    new_preferences_est = profile

            self._player_model_bridge.set_player_preferences_est(player_id, new_preferences_est)
