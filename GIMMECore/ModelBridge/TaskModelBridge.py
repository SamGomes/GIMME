# from abc import ABC, abstractmethod
from ..PlayerStructs import *


class TaskModelBridge(ABC):

    @abstractmethod
    def get_all_task_ids(self):
        pass

    @abstractmethod
    def get_task_interactions_profile(self, task_id):
        pass

    @abstractmethod
    def get_min_task_required_ability(self, task_id):
        pass

    @abstractmethod
    def get_min_task_duration(self, task_id):
        pass

    @abstractmethod
    def get_task_difficulty_weight(self, task_id):
        pass

    @abstractmethod
    def get_task_profile_weight(self, task_id):
        pass

    @abstractmethod
    def get_task_diversity_weight(self, task_id):
        pass

    @abstractmethod
    def get_task_init_date(self, task_id):
        pass

    @abstractmethod
    def get_task_final_date(self, task_id):
        pass
