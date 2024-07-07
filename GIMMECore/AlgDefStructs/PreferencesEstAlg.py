from abc import ABC, abstractmethod
from ..PlayerStructs import *
import json


class PreferencesEstAlg(ABC):

    def __init__(self, player_model_bridge):
        self.player_model_bridge = player_model_bridge

    @abstractmethod
    def update_estimates(self):
        pass


class ExploitationPreferencesEstAlg(PreferencesEstAlg):
    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 quality_weights=None):

        super().__init__(player_model_bridge)

        self.player_model_bridge = player_model_bridge
        self.quality_weights = PlayerCharacteristics(ability=0.5,
                                                     engagement=0.5) if quality_weights is None else quality_weights

        self.interactions_profile_template = interactions_profile_template
        self.quality_eval_alg = quality_eval_alg
        self.best_qualities = {}

    def calc_quality(self, state):
        return (self.quality_weights.ability * state.characteristics.ability +
                self.quality_weights.engagement * state.characteristics.engagement)

    def update_estimates(self):
        player_ids = self.player_model_bridge.get_all_player_ids()
        for playerId in player_ids:
            curr_preferences_quality = self.best_qualities.get(playerId, 0.0)
            last_data_point = self.player_model_bridge.get_player_curr_state(playerId)
            quality = self.calc_quality(last_data_point)
            if quality > curr_preferences_quality:
                self.best_qualities[playerId] = curr_preferences_quality
                self.player_model_bridge.set_player_preferences_est(playerId, last_data_point.profile)


class ExplorationPreferencesEstAlg(PreferencesEstAlg):
    def __init__(self,
                 player_model_bridge,
                 interactions_profile_template,
                 quality_eval_alg,
                 num_tested_player_profiles=None):

        super().__init__(player_model_bridge)

        self.player_model_bridge = player_model_bridge
        self.quality_eval_alg = quality_eval_alg

        self.numTestedPlayerProfiles = 100 if num_tested_player_profiles is None else num_tested_player_profiles
        self.interactionsProfileTemplate = interactions_profile_template

    def update_estimates(self):
        player_ids = self.player_model_bridge.get_all_player_ids()
        updated_estimates = {}
        for playerId in player_ids:

            curr_preferences_est = self.player_model_bridge.get_player_preferences_est(playerId)
            new_preferences_est = curr_preferences_est
            if curr_preferences_est is not None:
                best_quality = self.quality_eval_alg.evaluate(curr_preferences_est, [playerId])
            else:
                best_quality = -1

            for i in range(self.numTestedPlayerProfiles):
                profile = self.interactionsProfileTemplate.generate_copy().randomize()
                curr_quality = self.quality_eval_alg.evaluate(profile, [playerId])
                if curr_quality >= best_quality:
                    best_quality = curr_quality
                    new_preferences_est = profile

            self.player_model_bridge.set_player_preferences_est(playerId, new_preferences_est)
            updated_estimates[str(playerId)] = new_preferences_est

        return updated_estimates
