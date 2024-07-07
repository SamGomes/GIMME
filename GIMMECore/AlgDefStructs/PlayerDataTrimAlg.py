from abc import ABC, abstractmethod
import copy
from ..PlayerStructs import *
import json

class PlayerDataTrimAlg(ABC):

	def __init__(self):
		pass

	@abstractmethod
	def trimmedList(self, pastModelIncs):
		pass


# ---------------------- KNNRegression stuff ---------------------------
class AgeSortPlayerDataTrimAlg(PlayerDataTrimAlg):

	def __init__(self, maxNumModelElements):
		super().__init__()
		self.maxNumModelElements = maxNumModelElements

	def creationTimeSort(self, elem):
		return elem.creationTime

	def trimmedList(self, pastModelIncs):

		if(len(pastModelIncs) <= self.maxNumModelElements):
			return [pastModelIncs, []]

		pastModelIncsSorted = sorted(pastModelIncs, key=self.creationTimeSort)
		removedI = pastModelIncs.index(pastModelIncsSorted[0])
		pastModelIncs.pop(removedI)
		return [pastModelIncs, [removedI]]


class QualitySortPlayerDataTrimAlg(PlayerDataTrimAlg):

	def __init__(self, maxNumModelElements, qualityWeights = None, accStateResidue = None):
		super().__init__()
		self.maxNumModelElements = maxNumModelElements
		self.qualityWeights = PlayerCharacteristics(ability = 0.5, engagement = 0.5) if qualityWeights==None else qualityWeights
		self.accStateResidue = False if accStateResidue == None else accStateResidue

	def considerStateResidue(self, accStateResidue):
		self.accStateResidue = accStateResidue

	def stateTypeFilter(self, element):
		return element.stateType == 0

	def qSort(self, elem):
		return elem.quality

	def calcQuality(self, state):
		total = self.qualityWeights.ability*state.characteristics.ability + self.qualityWeights.engagement*state.characteristics.engagement
		return total

	def trimmedList(self, pastModelIncs):

		for modelInc in pastModelIncs:
			if(modelInc.quality == -1):
				modelInc.quality = self.calcQuality(modelInc)
				if(self.accStateResidue):
					modelInc.quality += modelInc.stateType


		if(len(pastModelIncs) <= self.maxNumModelElements):
			return [pastModelIncs, []]

		pastModelIncsSorted = sorted(pastModelIncs, key=self.qSort)
		removedI = pastModelIncs.index(pastModelIncsSorted[0])
		pastModelIncs.pop(removedI)
		return [pastModelIncs, [removedI]]


class ProximitySortPlayerDataTrimAlg(PlayerDataTrimAlg):

	def __init__(self, maxNumModelElements, epsilon = None, accStateResidue = None):
		super().__init__()
		self.maxNumModelElements = maxNumModelElements
		self.epsilon = 0.01 if epsilon == None else epsilon
		self.accStateResidue = False if accStateResidue == None else accStateResidue

	def considerStateResidue(self, accStateResidue):
		self.accStateResidue = accStateResidue
	
	def proximitySort(self, elem):
		return elem.quality

	def creationTimeSort(self, elem):
		return elem.creationTime

	def trimmedList(self, pastModelIncs):

		if(len(pastModelIncs) <= self.maxNumModelElements):
			return [pastModelIncs, []]

		pastModelIncsSortedAge = sorted(pastModelIncs, key=self.creationTimeSort)
		lastDataPoint = pastModelIncsSortedAge[-1]
		for modelInc in pastModelIncs:
			modelInc.quality = lastDataPoint.profile.sqr_distance_between(modelInc.profile)
			if(self.accStateResidue):
				modelInc.quality += modelInc.stateType

		# check if there is already a close point
		pastModelIncsSorted = sorted(pastModelIncs, key=self.proximitySort)
		pastModelIncsSorted.remove(lastDataPoint) #remove the point to be tested
		removedI = None
		closestPoint = pastModelIncsSorted[0]
		
		# print(json.dumps(closestPoint, default=lambda o: [o.__dict__["quality"],o.__dict__["stateType"],o.__dict__["creationTime"]], sort_keys=True))

		if (self.accStateResidue and closestPoint.stateType == 0) or closestPoint.quality > (self.epsilon + closestPoint.stateType):
			removedI = pastModelIncs.index(closestPoint)
			pastModelIncs.pop(removedI)
		else:
			removedI = pastModelIncs.index(lastDataPoint)
			pastModelIncs.pop(removedI)
		return [pastModelIncs, [removedI]]
