from GIMMECore import TaskModelBridge
from GIMMECore import PlayerModelBridge

class PlayerModelMock(object):
	def __init__(self, id, name, currState, pastModelIncreasesDataFrame, currModelIncreases, preferencesEst, realPreferences):
		self.currState = currState
		self.id = id
		self.name = name
		self.pastModelIncreasesDataFrame = pastModelIncreasesDataFrame

		# self.preferencesEst = preferencesEst.normalized()
		# self.realPreferences = realPreferences.normalized()
		self.baseLearningRate = None

class TaskModelMock(object):
	def __init__(self, id, description, minRequiredAbility, profile, minDuration, difficultyWeight, profileWeight):
		self.id = id
		self.description = description
		self.minRequiredAbility = minRequiredAbility
		self.profile = profile
		self.difficultyWeight = difficultyWeight
		self.profileWeight = profileWeight
		self.minDuration = minDuration

class CustomTaskModelBridge(TaskModelBridge):
	def __init__(self, tasks):
		self.tasks = tasks
		self.numTasks = len(tasks)

	def registerNewTask(self, taskId, description, minRequiredAbility, profile, minDuration, difficultyWeight, profileWeight):
		self.tasks[taskId] = TaskModelMock(taskId, description, minRequiredAbility, profile, minDuration, difficultyWeight, profileWeight)
	def get_all_task_ids(self):
		return [int(i) for i in range(self.numTasks)]
	def get_task_interactions_profile(self, task_id):
		return self.tasks[task_id].profile
	def get_min_task_required_ability(self, task_id):
		return self.tasks[task_id].minRequiredAbility
	def get_min_task_duration(self, task_id):
		return self.tasks[task_id].minDuration
	def get_task_difficulty_weight(self, task_id):
		return self.tasks[task_id].difficultyWeight
	def get_task_profile_weight(self, task_id):
		return self.tasks[task_id].profileWeight
	def get_task_init_date(self, task_id):
		return self.tasks[task_id].initDate
	def get_task_final_date(self, task_id):
		return self.tasks[task_id].finalDate
		
	def get_task_diversity_weight(self, task_id):
		return 0.5


class CustomPlayerModelBridge(PlayerModelBridge):
	def __init__(self, players):
		self.players = players
		self.numPlayers = len(players)


	def registerNewPlayer(self, playerId, name, currState, pastModelIncreasesDataFrame, currModelIncreases, preferencesEst, realPreferences):
		self.players[int(playerId)] = PlayerModelMock(playerId, name, currState, pastModelIncreasesDataFrame, currModelIncreases,  preferencesEst, realPreferences)	
	def reset_player(self, player_id):
		self.players[int(player_id)].currState.reset()
		self.players[int(player_id)].pastModelIncreasesDataFrame.reset()
	def resetState(self, playerId):
		self.players[int(playerId)].currState.reset()
	def set_and_save_player_state_to_data_frame(self, player_id, increases, new_state):
		self.players[int(player_id)].currState = new_state
		self.players[int(player_id)].pastModelIncreasesDataFrame.push_to_data_frame(increases)

	def setBaseLearningRate(self, playerId, blr):
		self.players[int(playerId)].baseLearningRate = blr
	def getBaseLearningRate(self, playerId):
		return self.players[int(playerId)].baseLearningRate

	def get_all_player_ids(self):
		return [str(i) for i in range(self.numPlayers)]

	def get_player_name(self, player_id):
		return self.players[int(player_id)].name
	def get_player_curr_state(self, player_id):
		return self.players[int(player_id)].currState
	def getPlayerCurrProfile(self,  playerId):
		return self.players[int(playerId)].currState.profile
	def get_player_states_data_frame(self, player_id):
		return self.players[int(player_id)].pastModelIncreasesDataFrame
	def get_player_curr_characteristics(self, player_id):
		return self.players[int(player_id)].currState.characteristics
	def get_player_preferences_est(self, player_id):
		return self.players[int(player_id)].preferencesEst
		
	def get_player_personality(self, player_id):
		return "<MOCKED PERSONALITY>"

	def set_player_preferences_est(self, player_id, preferencesEst):
		self.players[int(player_id)].preferencesEst = preferencesEst

	def set_player_characteristics(self, player_id, characteristics):
		self.players[int(player_id)].currState.characteristics = characteristics
	def set_player_profile(self, player_id, profile):
		self.players[int(player_id)].currState.profile = profile

	def set_player_group(self, player_id, group):
		self.players[int(player_id)].currState.group = group

	def set_player_tasks(self, player_id, tasks):
		self.players[int(player_id)].currState.tasks = tasks

	def setPlayerRealPreferences(self, playerId, realPreferences):
		self.players[int(playerId)].realPreferences = realPreferences
	def getPlayerRealPreferences(self, playerId):
		return self.players[int(playerId)].realPreferences
