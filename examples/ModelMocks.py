from GIMMECore import TaskModelBridge
from GIMMECore import PlayerModelBridge


class PlayerModelMock(object):
    def __init__(self, id, name, curr_state, past_model_increases_data_frame, curr_model_increases, preferences_est,
                 real_preferences):
        self.curr_state = curr_state
        self.id = id
        self.name = name
        self.past_model_increases_data_frame = past_model_increases_data_frame

        self.preferences_est = {}
        self.real_preferences = {}
        self.base_learning_rate = None


class TaskModelMock(object):
    def __init__(self, id, description, min_required_ability, profile, min_duration, difficulty_weight, profile_weight):
        self.id = id
        self.description = description
        self.min_required_ability = min_required_ability
        self.profile = profile
        self.difficulty_weight = difficulty_weight
        self.profile_weight = profile_weight
        self.min_duration = min_duration


class CustomTaskModelBridge(TaskModelBridge):
    def __init__(self):
        self.tasks = []

    def register_new_task(self, task_id, description, min_required_ability, profile, min_duration, difficulty_weight,
                          profile_weight):
        new_task = TaskModelMock(task_id, description, min_required_ability, profile, min_duration,
                                 difficulty_weight, profile_weight)
        if int(task_id) < len(self.tasks):
            self.tasks[task_id] = new_task
        else:
            self.tasks.append(new_task)

    def get_all_task_ids(self):
        return [str(task.id) for task in self.tasks]

    def get_task_interactions_profile(self, task_id):
        return self.tasks[int(task_id)].profile

    def get_min_task_required_ability(self, task_id):
        return self.tasks[int(task_id)].min_required_ability

    def get_min_task_duration(self, task_id):
        return self.tasks[int(task_id)].min_duration

    def get_task_difficulty_weight(self, task_id):
        return self.tasks[int(task_id)].difficulty_weight

    def get_task_profile_weight(self, task_id):
        return self.tasks[int(task_id)].profile_weight

    def get_task_init_date(self, task_id):
        return self.tasks[int(task_id)].initDate

    def get_task_final_date(self, task_id):
        return self.tasks[int(task_id)].finalDate

    def get_task_diversity_weight(self, task_id):
        return 0.5


class CustomPlayerModelBridge(PlayerModelBridge):
    def __init__(self):
        self.players = []

    def register_new_player(self, player_id, name, curr_state, past_model_increases_data_frame, curr_model_increases,
                            preferences_est, real_preferences):
        new_player = PlayerModelMock(player_id, name, curr_state, past_model_increases_data_frame,
                                     curr_model_increases, preferences_est, real_preferences)
        if int(player_id) < len(self.players):
            self.players[int(player_id)] = new_player
        else:
            self.players.append(new_player)

    def reset_player(self, player_id):
        self.players[int(player_id)].curr_state.reset()
        self.players[int(player_id)].past_model_increases_data_frame.reset()

    def reset_state(self, playerId):
        self.players[int(playerId)].curr_state.reset()

    def set_and_save_player_state_to_data_frame(self, player_id, increases, new_state):
        self.players[int(player_id)].curr_state = new_state
        self.players[int(player_id)].past_model_increases_data_frame.push_to_data_frame(increases)

    def set_base_learning_rate(self, player_id, blr):
        self.players[int(player_id)].base_learning_rate = blr

    def get_base_learning_rate(self, player_id):
        return self.players[int(player_id)].base_learning_rate

    def get_all_player_ids(self):
        return [str(player.id) for player in self.players]

    def get_player_name(self, player_id):
        return self.players[int(player_id)].name

    def get_player_curr_state(self, player_id):
        return self.players[int(player_id)].curr_state

    def get_player_curr_profile(self, player_id):
        return self.players[int(player_id)].curr_state.profile

    def get_player_states_data_frame(self, player_id):
        return self.players[int(player_id)].past_model_increases_data_frame

    def get_player_curr_characteristics(self, player_id):
        return self.players[int(player_id)].curr_state.characteristics

    def get_player_preferences_est(self, player_id):
        return self.players[int(player_id)].preferences_est

    def get_player_personality(self, player_id):
        return "<MOCKED PERSONALITY>"

    def set_player_preferences_est(self, player_id, preferences_est):
        self.players[int(player_id)].preferences_est = preferences_est

    def set_player_characteristics(self, player_id, characteristics):
        self.players[int(player_id)].curr_state.characteristics = characteristics

    def set_player_profile(self, player_id, profile):
        self.players[int(player_id)].curr_state.profile = profile

    def set_player_group(self, player_id, group):
        self.players[int(player_id)].curr_state.group = group

    def set_player_tasks(self, player_id, tasks):
        self.players[int(player_id)].curr_state.tasks = tasks

    def set_player_real_preferences(self, player_id, real_preferences):
        self.players[int(player_id)].real_preferences = real_preferences

    def get_player_real_preferences(self, player_id):
        return self.players[int(player_id)].real_preferences
