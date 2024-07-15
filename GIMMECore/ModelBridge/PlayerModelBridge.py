# from abc import ABC, abstractmethod
from ..PlayerStructs import *


class PlayerModelBridge(ABC):
    @abstractmethod
    def reset_player(self, player_id):
        pass

    @abstractmethod
    def get_all_player_ids(self):
        pass

    @abstractmethod
    def get_player_personality(self, player_id):
        pass

    @abstractmethod
    def get_player_states_data_frame(self, player_id):
        pass

    @abstractmethod
    def get_player_preferences_est(self, player_id):
        pass

    @abstractmethod
    def set_player_preferences_est(self, player_id, preferences):
        pass

    @abstractmethod
    def set_and_save_player_state_to_data_frame(self, player_id, increases, new_state):
        pass

    @abstractmethod
    def set_player_characteristics(self, player_id, characteristics):
        pass

    @abstractmethod
    def set_player_group(self, player_id, group):
        pass

    @abstractmethod
    def set_player_tasks(self, player_id, tasks):
        pass

    @abstractmethod
    def set_player_profile(self, player_id, profile):
        pass
