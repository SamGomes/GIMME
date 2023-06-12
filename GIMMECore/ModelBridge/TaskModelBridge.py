from abc import ABC, abstractmethod
from ..PlayerStructs import *

class TaskModelBridge(ABC):

	@abstractmethod
	def getAllTaskIds(self):
		pass

	@abstractmethod
	def getTaskInteractionsProfile(self, taskId):
		pass

	@abstractmethod
	def getMinTaskRequiredAbility(self, taskId):
		pass

	@abstractmethod
	def getMinTaskDuration(self, taskId):
		pass

	@abstractmethod
	def getTaskDifficultyWeight(self, taskId):
		pass

	@abstractmethod
	def getTaskProfileWeight(self, taskId):
		pass

	@abstractmethod
	def getTaskDiversityWeight(self, taskId):
		pass

	@abstractmethod
	def getTaskInitDate(self, taskId):
		pass

	@abstractmethod
	def getTaskFinalDate(self, taskId):
		pass