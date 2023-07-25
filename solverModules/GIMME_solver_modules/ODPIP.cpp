#include "ODPIP.h"

ODPIP::ODPIP(int numPlayers, double* coalitionValues, int minNumberOfPlayersPerGroup, int maxNumberOfPlayersPerGroup, std::vector<int> requiredJoinedPlayers, std::vector<int> restrictedJointPlayers)
{
	this->numPlayers = numPlayers;
	this->coalitionValues = coalitionValues;
	this->minNumberOfPlayersPerGroup = minNumberOfPlayersPerGroup;
	this->maxNumberOfPlayersPerGroup = maxNumberOfPlayersPerGroup;

	this->ipNumOfExpansions = 0;
	this->ipValueOfBestCSFound = -1;
	this->ipBestCSFound = std::vector<std::vector<int>>();
	this->totalNumOfExpansions = 0;
	this->valueOfBestPartitionFound = std::vector<double>((unsigned int)pow(2, numPlayers));

	this->max_f = new double[numPlayers];

	this->feasibleCoalitions = General::generateFeasibleCoalitionsInBitFormat(numPlayers, minNumberOfPlayersPerGroup, maxNumberOfPlayersPerGroup, requiredJoinedPlayers, restrictedJointPlayers, coalitionValues);
}

ODPIP::~ODPIP() 
{
	delete[] max_f;
	delete ipIntegerPartitionGraph;
}

int* ODPIP::IP() 
{

	searchFirstAndLastLevel();
	std::vector<double> avgValueForEachSize(numPlayers);

	computeAverage(avgValueForEachSize);

	std::vector<std::vector<double>> maxValueForEachSize = setMaxValueForEachSize();

	std::vector<std::vector<Subspace>> subspaces = initSubspaces(avgValueForEachSize, maxValueForEachSize);

	ipIntegerPartitionGraph = new IntegerPartitionGraph(subspaces, numPlayers, 1);

	totalNumOfExpansions = computeTotalNumOfExpansions();

	long numOfRemainingSubspaces = IntegerPartition::getNumOfIntegerPartitions(numPlayers);

	numOfRemainingSubspaces -= disableSubspacesThatWereSearchedWhileScanningTheInput();

	setUpperAndLowerBoundsOnOptimalValue();

	numOfRemainingSubspaces -= disableSubspacesWithUBLowerThanTheHighestLB();

	if (numOfRemainingSubspaces == 0)
	{
		finalize();
		return NULL;
	}

	initValueOfBestPartition(maxValueForEachSize);

	ODP();

	double acceptableValue = ipUpperBoundOnOptimalValue;
	
	std::vector<Node*> sortedNodes = getListOfSortedNodes(subspaces);

	while (((double)ipLowerBoundOnOptimalValue) < acceptableValue)
	{
		ipIntegerPartitionGraph->updateEdges(numPlayers, odpMaxSizeComputedSoFar);

		numOfRemainingSubspaces -= disableSubspacesReachableFromBottomNode();

		Node* nodeToBeSearched = getFirstEnabledNode(sortedNodes);
		if (nodeToBeSearched == NULL) break;

		std::vector<ElementOfMultiset> subsetToBePutAtTheBeginning = getRelevantNodes(nodeToBeSearched, 1);

		putSubsetAtTheBeginning(nodeToBeSearched, subsetToBePutAtTheBeginning);

		std::vector<double> sumOfMax = computeSumOfMax_splitOneInteger(nodeToBeSearched, maxValueForEachSize);

		int numOfIntegersToSplit = 0;
		
		if (!subsetToBePutAtTheBeginning.empty())
			numOfIntegersToSplit = (int)nodeToBeSearched->subspace.integers.size() - General::getCardinalityOfMultiset(subsetToBePutAtTheBeginning);
		
		numOfRemainingSubspaces -= nodeToBeSearched->subspace.search(this, acceptableValue, avgValueForEachSize, sumOfMax, numOfIntegersToSplit);

		setUpperAndLowerBoundsOnOptimalValue();
		acceptableValue = ipUpperBoundOnOptimalValue;

		if (ipLowerBoundOnOptimalValue < ipValueOfBestCSFound)
		{
			ipLowerBoundOnOptimalValue = ipValueOfBestCSFound;
			numOfRemainingSubspaces -= disableSubspacesWithUBLowerThanTheHighestLB();
		}

		if (numOfRemainingSubspaces == 0) break;
	}

	/*for (int i = 0; i < ipBestCSFound.size(); i++)
	{
		printf("{%ld, %lf}, \n", General::convertCombinationFromByteToBitFormat(ipBestCSFound[i]), coalitionValues[General::convertCombinationFromByteToBitFormat(ipBestCSFound[i])]);
	}

	for (int i = 0; i < feasibleCoalitions.size(); i++)
		if (feasibleCoalitions[i] && General::getSizeOfCombinationInBitFormat(i) == 5)
			printf("%ld\n", i);*/

	finalize();
	
	//int sortedNodesSize = sortedNodes.size();
	//for (int i = 0; i < sortedNodesSize; i++)
	//	delete[] sortedNodes[i];


	return NULL;
}


void ODPIP::finalize() 
{	

	this->bestCSInBitFormat = getBestCSFoundInBitFormat();
	
	long smallCoalition = -1;
	bool smallCoalitionFound = false;
	do 
	{	
		smallCoalition = -1;
		for (int i = 0; i < bestCSInBitFormat.size(); i++)
		{
			// find coalition with lower number of players than min number of players
			if (General::getSizeOfCombinationInBitFormat(bestCSInBitFormat[i]) < minNumberOfPlayersPerGroup) 
			{
				smallCoalition = bestCSInBitFormat[i];
				std::vector<long>::iterator it = bestCSInBitFormat.begin() + i;
				bestCSInBitFormat.erase(it);
				smallCoalitionFound = true;
				break;
			}
		}

		if (smallCoalitionFound)
		{
			for (int i = 0; i < bestCSInBitFormat.size(); i++)
			{
				// if there is another small coalition, merge the two
				if (General::getSizeOfCombinationInBitFormat(bestCSInBitFormat[i]) < minNumberOfPlayersPerGroup) 
				{
					bestCSInBitFormat[i] |= smallCoalition;
					smallCoalitionFound = false;
					break;
				}
			}

			if (smallCoalitionFound)
			{
				// if not, go member by member of small coalition and see where he fits best
				for (long mask = 1; mask <= smallCoalition; mask <<= 1)
				{	
					int tempCoalition = mask & smallCoalition;
					if (tempCoalition == 0)
						continue;

					int indexOfBestCoalition = -1;
					long bestCoalition = -1;
					double tempBestValue = -1;
					for (int i = 0; i < bestCSInBitFormat.size(); i++)
					{
						if (General::getSizeOfCombinationInBitFormat(bestCSInBitFormat[i] ^ tempCoalition) <= maxNumberOfPlayersPerGroup)
						{
							double value = getCoalitionValue(bestCSInBitFormat[i] ^ tempCoalition);
							if (value >= tempBestValue)
							{
								tempBestValue = value;
								indexOfBestCoalition = i;
								bestCoalition = bestCSInBitFormat[i] ^ tempCoalition;
							}
						}
					}
					if (indexOfBestCoalition != -1) {
						bestCSInBitFormat[indexOfBestCoalition] = bestCoalition;
						smallCoalition ^= mask;
					}
				}
			}
		}
		if(smallCoalitionFound && smallCoalition != -1 && smallCoalition != 0)
		{
			bestCSInBitFormat.push_back(smallCoalition);
			smallCoalition = -1;
		}

	} while (smallCoalition != -1);
}


std::vector<long> ODPIP::getBestCSFoundInBitFormat()
{
	std::vector<long> bestCSInBitFormat(ipBestCSFound.size());
	for (int i = 0; i < ipBestCSFound.size(); i++)
	{
		bestCSInBitFormat[i] = General::convertCombinationFromByteToBitFormat(ipBestCSFound[i]);
	}

	return bestCSInBitFormat;
}




void ODPIP::computeAverage(std::vector<double> &avgValueForEachSize)
{
	int totalNumOfCoalitions = (int)pow(2, numPlayers);
	double bestValue = ipValueOfBestCSFound;
	int bestCoalition1 = 0; int bestCoalition2 = 0;
	std::vector<double> sumOfValues(numPlayers);
	for (unsigned int size = 1; size <= numPlayers; size++)
		sumOfValues[size - 1] = 0;

	//Initialize the variables that are only used if there are constraints
	bool coalition1IsFeasible = true;
	bool coalition2IsFeasible = true;
	bool constraintsExist = false;
	
	if (feasibleCoalitions.empty())
		constraintsExist = false;
	else
		constraintsExist = true;
	

	for (int coalition1 = totalNumOfCoalitions / 2; coalition1 < totalNumOfCoalitions - 1; coalition1++) 
	{
		int sizeOfCoalition1 = General::getSizeOfCombinationInBitFormat(coalition1);

		int coalition2 = totalNumOfCoalitions - 1 - coalition1;
		int sizeOfCoalition2 = numPlayers - sizeOfCoalition1;

		sumOfValues[sizeOfCoalition1 - 1] += getCoalitionValue(coalition1);
		sumOfValues[sizeOfCoalition2 - 1] += getCoalitionValue(coalition2);

		
		if (constraintsExist) {
			coalition1IsFeasible = feasibleCoalitions[coalition1];
			coalition2IsFeasible = feasibleCoalitions[coalition2];
		}

		double value = 0;
		if ((constraintsExist == false) || ((coalition1IsFeasible) && (coalition2IsFeasible))) {
			value = getCoalitionValue(coalition1) + getCoalitionValue(coalition2);
		}

		if (bestValue < value) 
		{
			bestCoalition1 = coalition1;
			bestCoalition2 = coalition2;
			bestValue = value;
		}
	}

	for (unsigned int size = 1; size <= numPlayers; size++)
		avgValueForEachSize[size - 1] = (double)sumOfValues[size - 1] / General::binomialCoefficient(numPlayers, size);
}

void ODPIP::searchFirstAndLastLevel() 
{
	bool CS1IsFeasible = true;
	bool CS2IsFeasible = true;

	std::vector<std::vector<int>> CS1 = std::vector<std::vector<int>>(numPlayers);
	for (unsigned int i = 0; i < numPlayers; i++) {
		CS1[i] = std::vector<int>(1,i+1);
	}

	std::vector<std::vector<int>> CS2 = std::vector<std::vector<int>>(1);
	CS2[0] = std::vector<int>(numPlayers);
	for (unsigned int i = 0; i < numPlayers; i++) {
		CS2[0][i] = i + 1;
	}

	
	if (!feasibleCoalitions.empty()) 
	{
		//Check whether CS1 is feasible
		for (int i = 0; i < CS2.size(); i++)
		{
			int curCoalitionInBitFormat = General::convertCombinationFromByteToBitFormat(CS2[i]);
			if (feasibleCoalitions[curCoalitionInBitFormat] == false)
			{
				CS1IsFeasible = false;
				break;
			}
		}

		//Check whether CS2 is feasible
		for (int i = 0; i < CS2.size(); i++) 
		{
			int curCoalitionInBitFormat = General::convertCombinationFromByteToBitFormat(CS2[i]);
			if (feasibleCoalitions[curCoalitionInBitFormat] == false) 
			{
				CS2IsFeasible = false;
				break;
			}
		}
	}
	

	double valueOfCS1 = getCoalitionStructureValue(CS1);
	double valueOfCS2 = getCoalitionStructureValue(CS2);

	if (((CS1IsFeasible) && (CS2IsFeasible == false)) || ((CS1IsFeasible) && (CS2IsFeasible) && (valueOfCS1 >= valueOfCS2))) {
		updateIpSolution(CS1, valueOfCS1);
	}
	if (((CS2IsFeasible) && (CS1IsFeasible == false)) || ((CS2IsFeasible) && (CS1IsFeasible) && (valueOfCS2 >= valueOfCS1))) {
		updateIpSolution(CS2, valueOfCS2);
	}
	updateNumOfSearchedAndRemainingCoalitionsAndCoalitionStructures();
}

std::vector<std::vector<Subspace>> ODPIP::initSubspaces(std::vector<double> avgValueForEachSize, std::vector<std::vector<double>> maxValueForEachSize)
{
	std::vector<std::vector<std::vector<int>>> integers = IntegerPartition::getIntegerPartitions(numPlayers);
	std::vector<std::vector<Subspace>> subspaces(integers.size());

	for (unsigned int level = 0; level < numPlayers; level++) 
	{
		subspaces[level].resize(integers[level].size());
		for (int i = 0; i < integers[level].size(); i++) 
		{
			subspaces[level][i] = Subspace(integers[level][i], avgValueForEachSize, maxValueForEachSize, numPlayers);
		}
	}

	return subspaces;
}

long ODPIP::disableSubspacesThatWereSearchedWhileScanningTheInput() 
{
	std::vector<std::vector<Node*>> nodes = ipIntegerPartitionGraph->nodes;
	long numOfSubspacesThatThisMethodHasDisabled = 0;

	for (int level = 0; level < nodes.size(); level++)
	{
		for (int i = 0; i < nodes[level].size(); i++)
		{
			if ((level == 0) || (level == 1) || (level == numPlayers - 1))
			{
				if (nodes[level][i]->subspace.enabled)
				{
					nodes[level][i]->subspace.enabled = false;
					numOfSubspacesThatThisMethodHasDisabled++;
				}
			}
		}
	}
	ipIntegerPartitionGraph->nodes = nodes;
	return numOfSubspacesThatThisMethodHasDisabled;
}

long ODPIP::disableSubspacesWithUBLowerThanTheHighestLB()
{
	std::vector<std::vector<Node*>> nodes = ipIntegerPartitionGraph->nodes;

	long numOfSubspacesThatThisMethodHasDisabled = 0;

	for (int level = 0; level < nodes.size(); level++)
	{
		for (int i = 0; i < nodes[level].size(); i++)
		{
			if (nodes[level][i]->subspace.enabled)
				if (nodes[level][i]->subspace.UB - ipLowerBoundOnOptimalValue < -0.000000000005f)
				{
					nodes[level][i]->subspace.enabled = false;
					numOfSubspacesThatThisMethodHasDisabled++;
				}
		}
	}

	ipIntegerPartitionGraph->nodes = nodes;
	return numOfSubspacesThatThisMethodHasDisabled;
}

long ODPIP::disableSubspacesReachableFromBottomNode()
{
	Node* bottomNode = ipIntegerPartitionGraph->nodes[0][0];

	std::vector<Node*> reachableNodes = ipIntegerPartitionGraph->getReachableNodes(bottomNode);

	int numOfDisabledNodes = 0;
	if (!reachableNodes.empty())
		for (int i = 0; i < reachableNodes.size(); i++)
			if (reachableNodes[i]->subspace.enabled)
			{
				reachableNodes[i]->subspace.enabled = false;
				numOfDisabledNodes++;
			}

	return numOfDisabledNodes;
}

void ODPIP::setUpperAndLowerBoundsOnOptimalValue() 
{
	ipUpperBoundOnOptimalValue = ipValueOfBestCSFound;
	ipLowerBoundOnOptimalValue = ipValueOfBestCSFound;

	std::vector<std::vector<Node*>> nodes = ipIntegerPartitionGraph->nodes;

	for (int level = 0; level < nodes.size(); level++)
	{
		for (int i = 0; i < nodes[level].size(); i++)
		{
			Subspace curSubspace = nodes[level][i]->subspace;
			if (curSubspace.enabled)
			{
				if (ipUpperBoundOnOptimalValue < curSubspace.UB)
					ipUpperBoundOnOptimalValue = curSubspace.UB;

				if (ipLowerBoundOnOptimalValue < curSubspace.LB)
					ipLowerBoundOnOptimalValue = curSubspace.LB;
			}
		}
	}
}


std::vector<Node*> ODPIP::getListOfSortedNodes(std::vector<std::vector<Subspace>> subspaces)
{
	std::vector<std::vector<Node*>>nodes = ipIntegerPartitionGraph->nodes;
	for (int level = 0; level < nodes.size(); level++) {
		for (int i = 0; i < nodes[level].size(); i++)
		{
			nodes[level][i]->subspace.priority = nodes[level][i]->subspace.UB;
		}
	}

	ipIntegerPartitionGraph->nodes = nodes;

	std::vector<Node*> sortedNodes(IntegerPartition::getNumOfIntegerPartitions(numPlayers));
	int k = 0;

	for (int level = 0; level < nodes.size(); level++)
		for (int i = 0; i < nodes[level].size(); i++)
		{
			sortedNodes[k] = nodes[level][i];
			k++;
		}

	int sortedNodesSize = (int)sortedNodes.size();
	for (int i = sortedNodesSize - 1; i >= 0; i--)
	{
		int indexOfSmallestElement = i;
		for (int j = i; j >= 0; j--)
		{
			if ((sortedNodes[j]->subspace.priority < sortedNodes[indexOfSmallestElement]->subspace.priority)
				|| ((sortedNodes[j]->subspace.priority == sortedNodes[indexOfSmallestElement]->subspace.priority)
					&& (sortedNodes[j]->subspace.UB < sortedNodes[indexOfSmallestElement]->subspace.UB)))
				indexOfSmallestElement = j;
		}

		Node* temp = sortedNodes[i];
		sortedNodes[i] = sortedNodes[indexOfSmallestElement];
		sortedNodes[indexOfSmallestElement] = temp;
	}

	return sortedNodes;
}

Node* ODPIP::getFirstEnabledNode(std::vector<Node*> sortedNodes) 
{
	for (int i = 0; i < sortedNodes.size(); i++)
	{
		if (sortedNodes[i]->subspace.enabled)
			return sortedNodes[i];
	}

	return NULL;
}





void ODPIP::updateIpSolution(std::vector<std::vector<int>> CS, double value) {
	if (ipValueOfBestCSFound <= value) {
		ipValueOfBestCSFound = value;
		ipBestCSFound = CS;
	}
}

void ODPIP::updateOdpSolution(std::vector<std::vector<int>> CS, double value) {
	if (odpValueOfBestCSFound <= value) {
		odpValueOfBestCSFound = value;
		odpBestCSFound = CS;
	}
}


double ODPIP::getCoalitionValue(int coalitionInBitFormat) {
	return coalitionValues[coalitionInBitFormat];
}

double ODPIP::getCoalitionValue(std::vector<int> coalitionInByteFormat) 
{
	int	coalitionInBitFormat = General::convertCombinationFromByteToBitFormat(coalitionInByteFormat);

	return getCoalitionValue(coalitionInBitFormat);
}


double ODPIP::getCoalitionStructureValue(std::vector<std::vector<int>> coalitionStructure) 
{
	double valueOfCS = 0;
	for (int i = 0; i < coalitionStructure.size(); i++)
		valueOfCS += getCoalitionValue(coalitionStructure[i]);

	return valueOfCS;
}

void ODPIP::updateNumOfSearchedAndRemainingCoalitionsAndCoalitionStructures() 
{
	for (int size1 = 1; size1 <= floor(numPlayers / (double)2); size1++) 
	{
		int size2 = (int)(numPlayers - size1);

		int numOfCombinationsOfSize1 = (int)General::binomialCoefficient(numPlayers, size1);

		int temp;
		if (size1 != size2)
			temp = numOfCombinationsOfSize1;
		else
			temp = numOfCombinationsOfSize1 / 2;

		ipNumOfExpansions += 2 * temp;
	}
}

std::vector<std::vector<double>> ODPIP::setMaxValueForEachSize() 
{
	int* numOfRequiredMaxValues = new int[numPlayers];
	long* numOfCoalitions = new long[numPlayers];
	std::vector<std::vector<double>> maxValue(numPlayers);

	for (unsigned int size = 1; size <= numPlayers; size++) 
	{
		numOfRequiredMaxValues[size - 1] = (int)floor(numPlayers / (double)size);
		numOfCoalitions[size - 1] = General::binomialCoefficient(numPlayers, size);
		maxValue[size - 1] = std::vector<double>(numOfRequiredMaxValues[size - 1]);
		for (int i = 0; i < maxValue[size - 1].size(); i++) 
		{
			maxValue[size - 1][i] = 0;
		}
	}
	bool constraintsExist = false;
	
	if (feasibleCoalitions.empty())
		constraintsExist = false;
	else
		constraintsExist = true;
	
	for (int coalitionInBitFormat = (int)pow(2, numPlayers) - 1; coalitionInBitFormat > 0; coalitionInBitFormat--)
		if ((constraintsExist == false) || feasibleCoalitions[coalitionInBitFormat]) 
		{
			int size = (int)General::getSizeOfCombinationInBitFormat(coalitionInBitFormat);
			std::vector<double> curMaxValue = maxValue[size - 1];
			int j = numOfRequiredMaxValues[size - 1] - 1;
			if (getCoalitionValue(coalitionInBitFormat) > curMaxValue[j]) 
			{

				while ((j > 0) && (getCoalitionValue(coalitionInBitFormat) > curMaxValue[j - 1])) 
				{
					curMaxValue[j] = curMaxValue[j - 1];
					j--;
				}

				curMaxValue[j] = getCoalitionValue(coalitionInBitFormat);
			}
			maxValue[size - 1] = curMaxValue;
		}
	delete[] numOfRequiredMaxValues;
	delete[] numOfCoalitions;
	return maxValue;
}

long ODPIP::computeTotalNumOfExpansions() 
{
	long totalNumOfExpansions = 0;
	std::vector<std::vector<Node*>> nodes = ipIntegerPartitionGraph->nodes;
	for (int level = 0; level < nodes.size(); level++) 
	{
		for (int i= 0; i < nodes[level].size(); i++) 
		{
			Subspace curSubspace = nodes[level][i]->subspace;
			totalNumOfExpansions += curSubspace.totalNumOfExpansionsInSubspace;
		}
	}

	return totalNumOfExpansions;
}

std::vector<ElementOfMultiset> ODPIP::getRelevantNodes(Node* node, int numOfIntegersToSplitAtTheEnd)
{
	int numOfIntegersInNode = (int)node->integerPartition.partsSortedAscendingly.size();

	std::vector<Node*> reachableNodes = ipIntegerPartitionGraph->getReachableNodes(node);
	if (reachableNodes.empty())
	{
		node->subspace.relevantNodes = std::vector<Node*>();
		return std::vector<ElementOfMultiset>();
	}

	SubsetOfMultiset* subsetIterators = new SubsetOfMultiset[numOfIntegersToSplitAtTheEnd];
	for (int s = 0; s < numOfIntegersToSplitAtTheEnd; s++)
		subsetIterators[s] = SubsetOfMultiset(node->integerPartition.sortedMultiset, (int)(numOfIntegersInNode - (s + 1)), false);

	std::vector<ElementOfMultiset> bestSubset;
	long bestSavings = 0;
	int bestNumOfRelevantNodes = 0;

	for (int s = 0; s < numOfIntegersToSplitAtTheEnd; s++)
	{
		std::vector<ElementOfMultiset> subset = subsetIterators[s].getNextSubset();
		while (!subset.empty())
		{
			long savings = 0;
			int numOfRelevantSubspaces = 0;

			for (int i = 0; i < reachableNodes.size(); i++)
			{
				if ((reachableNodes[i]->integerPartition.contains(subset)) && (reachableNodes[i]->subspace.enabled))
				{
					savings += reachableNodes[i]->subspace.totalNumOfExpansionsInSubspace;
					numOfRelevantSubspaces++;
				}
			}

			if (bestSavings < savings)
			{
				bestSavings = savings;
				bestSubset = subset;
				bestNumOfRelevantNodes = numOfRelevantSubspaces;
			}

			subset = subsetIterators[s].getNextSubset();
		}
	}

	delete[] subsetIterators;
	int index = 0;
	if (bestNumOfRelevantNodes == 0)
	{
		node->subspace.relevantNodes = std::vector<Node*>();
		return std::vector<ElementOfMultiset>();
	}
	else
	{
		node->subspace.relevantNodes = std::vector<Node*>(bestNumOfRelevantNodes);
		for (int i = 0; i < reachableNodes.size(); i++)
		{
			if ((reachableNodes[i]->integerPartition.contains(bestSubset)) && (reachableNodes[i]->subspace.enabled))
			{
				node->subspace.relevantNodes[index] = reachableNodes[i];
				index++;
			}
		}

		return bestSubset;
	}
}

void ODPIP::putSubsetAtTheBeginning(Node* node, std::vector<ElementOfMultiset> subset)
{
	if (subset.empty())
		return;
	std::vector<ElementOfMultiset> remainingIntegers_multiset = node->integerPartition.sortedMultiset;

	for (int i = 0; i < subset.size(); i++)
		for (int j = 0; j < remainingIntegers_multiset.size(); j++)
			if (remainingIntegers_multiset[j].element == subset[i].element) 
			{
				remainingIntegers_multiset[j].repetition -= subset[i].repetition;
				break;
			}
	
	int counter = 0;
	for (int i = 0; i < remainingIntegers_multiset.size(); i++)
		counter += remainingIntegers_multiset[i].repetition;

	std::vector<int> remainingIntegers_array(counter);
	int index = 0;
	for (int i = 0; i < remainingIntegers_multiset.size(); i++)
		for (int j = 0; j < remainingIntegers_multiset[i].repetition; j++) 
		{
			remainingIntegers_array[index] = remainingIntegers_multiset[i].element;
			index++;
		}
	
	std::vector<int> newIntegers(node->subspace.integers.size());
	int index1 = 0;
	int index2 = (int)newIntegers.size() - counter;
	for (int i = 0; i < node->subspace.integers.size(); i++)
	{
		bool found = false;
		for (int j = 0; j < remainingIntegers_array.size(); j++) {
			if (remainingIntegers_array[j] == node->subspace.integers[i])
			{
				newIntegers[index2] = node->subspace.integers[i];
				index2++;
				remainingIntegers_array[j] = -1;
				found = true;
				break;
			}
		}
		if (found == false) {
			newIntegers[index1] = node->subspace.integers[i];
			index1++;
		}
	}
	node->subspace.integers = newIntegers;
}

std::vector<double> ODPIP::computeSumOfMax_splitOneInteger(Node* node, std::vector<std::vector<double>> maxValueForEachSize)
{
	if (node->subspace.relevantNodes.empty())
		return computeSumOfMax_splitNoIntegers(node->subspace.integers, maxValueForEachSize);
	
	std::vector<int> integers = node->subspace.integers;
	std::vector<double> sumOfMax(integers.size() + 1);

	double maxUB = node->subspace.UB;
	for (int i = 0; i < node->subspace.relevantNodes.size(); i++)
		if (maxUB < node->subspace.relevantNodes[i]->subspace.UB)
			maxUB = node->subspace.relevantNodes[i]->subspace.UB;

	sumOfMax[integers.size()] = maxValueForEachSize[integers[integers.size() - 1] - 1][0] + (maxUB - node->subspace.UB);
	double max_f = get_max_f(integers[integers.size() - 1] - 1);
	if ((max_f != 0) && (sumOfMax[integers.size()] > max_f))
		sumOfMax[integers.size()] = max_f;

	sumOfMax[integers.size() - 1] = sumOfMax[integers.size()] + maxValueForEachSize[integers[integers.size() - 2] - 1][0];
	int k = 2;

	int x = (int)integers.size() - k;
	int j = 0;
	for (int i = x; i > 0; i--)
	{
		if (integers[i - 1] == integers[i])
			j++;
		else
			j = 0;
		sumOfMax[i] = sumOfMax[i + 1] + maxValueForEachSize[integers[i - 1] - 1][j];
		k++;
	}
	return(sumOfMax);
}


std::vector<double> ODPIP::computeSumOfMax_splitNoIntegers(std::vector<int> integers, std::vector<std::vector<double>> maxValueForEachSize)
{
	std::vector<double> sumOfMax(integers.size() + 1);

	sumOfMax[integers.size()] = maxValueForEachSize[integers[integers.size() - 1] - 1][0];

	int j = 0;
	int integersSize = (int)integers.size();
	for (int i = integersSize - 1; i > 0; i--)
	{
		if (integers[i - 1] == integers[i])
			j++;
		else
			j = 0;
		sumOfMax[i] = sumOfMax[i + 1] + maxValueForEachSize[integers[i - 1] - 1][j];
	}

	return sumOfMax;
}

// ODP algorithm

void ODPIP::ODP()
{
	updateValueOfBestPartitionFound(0, 0);

	int grandCoalition = (1 << numPlayers) - 1;
	int numOfCoalitions = 1 << numPlayers;
	int bestHalfOfGrandCoalition = -1;

	for (unsigned int curSize = 2; curSize <= numPlayers; curSize++)
	{
		if (((unsigned int)(floor((2 * numPlayers) / (double)3)) < curSize) && (curSize < numPlayers)) continue;

		if (curSize < numPlayers)
		{
			int numOfCoalitionsOfCurSize = (int)General::binomialCoefficient(numPlayers, curSize);

			std::vector<int> curCoalition = General::getCombinationAtGivenIndex(curSize, numOfCoalitionsOfCurSize - 1, numPlayers);
			evaluateSplits(curCoalition, curSize);
			for (int i = 1; i < numOfCoalitionsOfCurSize; i++)
			{
				General::getPreviousCombination(numPlayers, curSize, curCoalition);
				evaluateSplits(curCoalition, curSize);
			}
		}
		else
			bestHalfOfGrandCoalition = evaluateSplitsOfGrandCoalition();

		if (curSize < numPlayers)
		{
			bestHalfOfGrandCoalition = evaluateSplitsOfGrandCoalition();
			std::vector<int> bestCSFoundSoFar = getOptimalSplit(grandCoalition, bestHalfOfGrandCoalition);
			std::vector<std::vector<int>> bestCSFoundSoFar_byteFormat = General::convertSetOfCombinationsFromBitToByteFormat(bestCSFoundSoFar, numPlayers);
			updateOdpSolution(bestCSFoundSoFar_byteFormat, +getCoalitionStructureValue(bestCSFoundSoFar_byteFormat));
			updateIpSolution(bestCSFoundSoFar_byteFormat, getCoalitionStructureValue(bestCSFoundSoFar_byteFormat));
		}

		odpMaxSizeComputedSoFar = curSize;
	}

	std::vector<int> bestCSFound = getOptimalSplit(grandCoalition, bestHalfOfGrandCoalition);
	std::vector<std::vector<int>> odpBestCSInByteFormat = General::convertSetOfCombinationsFromBitToByteFormat(bestCSFound, numPlayers);
	updateOdpSolution(odpBestCSInByteFormat, +getCoalitionStructureValue(odpBestCSInByteFormat));
	updateIpSolution(odpBestCSInByteFormat, getCoalitionStructureValue(odpBestCSInByteFormat));
}


void ODPIP::evaluateSplits(std::vector<int> coalitionInByteFormat, int coalitionSize)
{
	double curValue = -1;
	double bestValue = -1;
	int coalitionInBitFormat = General::convertCombinationFromByteToBitFormat(coalitionInByteFormat);

	bestValue = getCoalitionValue(coalitionInBitFormat);

	for (int firstHalfInBitFormat = coalitionInBitFormat - 1 & coalitionInBitFormat;; firstHalfInBitFormat = firstHalfInBitFormat - 1 & coalitionInBitFormat)
	{
		int secondHalfInBitFormat = coalitionInBitFormat ^ firstHalfInBitFormat;

		curValue = getValueOfBestPartitionFound(firstHalfInBitFormat) + getValueOfBestPartitionFound(secondHalfInBitFormat);

		if (bestValue <= curValue)
		{
			bestValue = curValue;
		}

		if ((firstHalfInBitFormat & (firstHalfInBitFormat - 1)) == 0) break;
	}

	if (get_max_f(coalitionSize - 1) < bestValue)
		set_max_f(coalitionSize - 1, bestValue);

	updateValueOfBestPartitionFound(coalitionInBitFormat, bestValue);
}

int ODPIP::evaluateSplitsOfGrandCoalition()
{
	double curValue = -1;
	double bestValue = -1;
	int bestHalfOfGrandCoalitionInBitFormat = -1;
	int numOfCoalitions = 1 << numPlayers;
	int grandCoalition = (1 << numPlayers) - 1;

	for (int firstHalfOfGrandCoalition = (numOfCoalitions / 2) - 1; firstHalfOfGrandCoalition < numOfCoalitions; firstHalfOfGrandCoalition++)
	{
		int secondHalfOfGrandCoalition = numOfCoalitions - 1 - firstHalfOfGrandCoalition;
		curValue = getValueOfBestPartitionFound(firstHalfOfGrandCoalition) + getValueOfBestPartitionFound(secondHalfOfGrandCoalition);
		if (curValue > bestValue)
		{
			bestValue = curValue;
			bestHalfOfGrandCoalitionInBitFormat = firstHalfOfGrandCoalition;
		}
	}

	int firstHalfOfGrandCoalition = grandCoalition;
	curValue = getValueOfBestPartitionFound(firstHalfOfGrandCoalition);
	if (curValue > bestValue)
	{
		bestValue = curValue;
		bestHalfOfGrandCoalitionInBitFormat = firstHalfOfGrandCoalition;
	}

	updateValueOfBestPartitionFound(grandCoalition, bestValue);

	return bestHalfOfGrandCoalitionInBitFormat;
}

std::vector<int> ODPIP::getOptimalSplit(int coalitionInBitFormat, int bestHalfOfCoalition)
{
	std::vector<int> optimalSplit;
	if (bestHalfOfCoalition == coalitionInBitFormat)
	{
		optimalSplit.resize(1);
		optimalSplit[0] = coalitionInBitFormat;
	}
	else
	{
		std::vector<int> arrayOfBestHalf(2);
		std::vector<std::vector<int>> arrayOfOptimalSplit(2);
		std::vector<int> arrayOfCoalitionInBitFormat(2);

		arrayOfCoalitionInBitFormat[0] = bestHalfOfCoalition;
		arrayOfCoalitionInBitFormat[1] = coalitionInBitFormat - bestHalfOfCoalition;

		for (int i = 0; i < 2; i++)
		{
			arrayOfBestHalf[i] = getBestHalf(arrayOfCoalitionInBitFormat[i]);

			arrayOfOptimalSplit[i] = getOptimalSplit(arrayOfCoalitionInBitFormat[i], arrayOfBestHalf[i]);
		}

		optimalSplit.resize(arrayOfOptimalSplit[0].size() + arrayOfOptimalSplit[1].size());
		int k = 0;
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < arrayOfOptimalSplit[i].size(); j++)
			{
				optimalSplit[k] = arrayOfOptimalSplit[i][j];
				k++;
			}
	}

	return optimalSplit;
}

std::vector<std::vector<int>> ODPIP::getOptimalSplit(std::vector<std::vector<int>> CS)
{
	std::vector<std::vector<int>> optimalSplit(CS.size());
	int numOfCoalitionsInFinalResult = 0;

	for (int i = 0; i < CS.size(); i++)
	{
		int coalitionInBitFormat = General::convertCombinationFromByteToBitFormat(CS[i]);
		int bestHalfOfGrandCoalitionInBitFormat = getBestHalf(coalitionInBitFormat);
		optimalSplit[i] = getOptimalSplit(coalitionInBitFormat, bestHalfOfGrandCoalitionInBitFormat);
		numOfCoalitionsInFinalResult += (int)optimalSplit[i].size();
	}

	std::vector<std::vector<int>> finalResult(numOfCoalitionsInFinalResult);
	int k = 0;
	for (int i = 0; i < CS.size(); i++)
		for (int j = 0; j < optimalSplit[i].size(); j++)
		{
			finalResult[k] = General::convertCombinationFromBitToByteFormat(optimalSplit[i][j], numPlayers);
			k++;
		}

	return finalResult;
}




int ODPIP::getBestHalf(int coalitionInBitFormat)
{
	double valueOfBestSplit = getCoalitionValue(coalitionInBitFormat);
	int best_firstHalfInBitFormat = coalitionInBitFormat;

	int* bit = new int[numPlayers + 1];
	for (unsigned int i = 0; i < numPlayers; i++)
		bit[i + 1] = 1 << i;

	std::vector<int> coalitionInByteFormat = General::convertCombinationFromBitToByteFormat(coalitionInBitFormat, numPlayers);

	int coalitionSize = (int)coalitionInByteFormat.size();
	for (int sizeOfFirstHalf = (int)ceil(coalitionSize / (double)2); sizeOfFirstHalf < coalitionSize; sizeOfFirstHalf++)
	{
		int sizeOfSecondHalf = coalitionSize - sizeOfFirstHalf;

		/*if ((sizeOfFirstHalf > maxNumberOfPlayersPerGroup || sizeOfFirstHalf < minNumberOfPlayersPerGroup)
			&& (sizeOfSecondHalf > maxNumberOfPlayersPerGroup || sizeOfSecondHalf < minNumberOfPlayersPerGroup))
			continue;*/


		long numOfPossibleSplits;
		if (((coalitionSize % 2) == 0) && (sizeOfFirstHalf == sizeOfSecondHalf))
			numOfPossibleSplits = (int)(General::binomialCoefficient(coalitionSize, sizeOfFirstHalf) / 2);
		else
			numOfPossibleSplits = General::binomialCoefficient(coalitionSize, sizeOfFirstHalf);

		std::vector<int> indicesOfMembersOfFirstHalf(sizeOfFirstHalf);
		for (int i = 0; i < sizeOfFirstHalf; i++)
			indicesOfMembersOfFirstHalf[i] = i + 1;

		int firstHalfInBitFormat = 0;
		for (int i = 0; i < sizeOfFirstHalf; i++)
			firstHalfInBitFormat += bit[coalitionInByteFormat[indicesOfMembersOfFirstHalf[i] - 1]];

		int secondHalfInBitFormat = coalitionInBitFormat - firstHalfInBitFormat;
		double curValue = getValueOfBestPartitionFound(firstHalfInBitFormat) + getValueOfBestPartitionFound(secondHalfInBitFormat);
		if (curValue > valueOfBestSplit)
		{
			//if (sizeOfFirstHalf <= maxNumberOfPlayersPerGroup && sizeOfFirstHalf >= minNumberOfPlayersPerGroup)
				best_firstHalfInBitFormat = firstHalfInBitFormat;

			/*else
				best_firstHalfInBitFormat = secondHalfInBitFormat;
			*/valueOfBestSplit = curValue;
			
		}

		/*else if (abs(curValue - valueOfBestSplit) < 0.0000005)
		{
			int sizeOfBestFirstHalf = General::getSizeOfCombinationInBitFormat(best_firstHalfInBitFormat);
			if (sizeOfBestFirstHalf > maxNumberOfPlayersPerGroup || sizeOfBestFirstHalf < minNumberOfPlayersPerGroup)
			{
				if (sizeOfFirstHalf <= maxNumberOfPlayersPerGroup && sizeOfFirstHalf >= minNumberOfPlayersPerGroup)
					best_firstHalfInBitFormat = firstHalfInBitFormat;
				
				else
					best_firstHalfInBitFormat = secondHalfInBitFormat;
				
				valueOfBestSplit = curValue;

			}
		}*/

		for (int j = 1; j < numOfPossibleSplits; j++)
		{
			General::getPreviousCombination(coalitionSize, sizeOfFirstHalf, indicesOfMembersOfFirstHalf);

			int firstHalfInBitFormat = 0;
			for (int i = 0; i < sizeOfFirstHalf; i++)
				firstHalfInBitFormat += bit[coalitionInByteFormat[indicesOfMembersOfFirstHalf[i] - 1]];

			int secondHalfInBitFormat = coalitionInBitFormat - firstHalfInBitFormat;

			double curValue = getValueOfBestPartitionFound(firstHalfInBitFormat) + getValueOfBestPartitionFound(secondHalfInBitFormat);
			if (curValue > valueOfBestSplit)
			{
				//if (sizeOfFirstHalf <= maxNumberOfPlayersPerGroup && sizeOfFirstHalf >= minNumberOfPlayersPerGroup)
					best_firstHalfInBitFormat = firstHalfInBitFormat;

				/*else
					best_firstHalfInBitFormat = secondHalfInBitFormat;
				*/valueOfBestSplit = curValue;

			}

			/*else if (abs(curValue - valueOfBestSplit) < 0.0000005)
			{
				int sizeOfBestFirstHalf = General::getSizeOfCombinationInBitFormat(best_firstHalfInBitFormat);
				if (sizeOfBestFirstHalf > maxNumberOfPlayersPerGroup || sizeOfBestFirstHalf < minNumberOfPlayersPerGroup)
				{
					if (sizeOfFirstHalf <= maxNumberOfPlayersPerGroup && sizeOfFirstHalf >= minNumberOfPlayersPerGroup)
						best_firstHalfInBitFormat = firstHalfInBitFormat;

					else
						best_firstHalfInBitFormat = secondHalfInBitFormat;
					
					valueOfBestSplit = curValue;
				}
			}*/

		}

	}

	delete[] bit;

	return best_firstHalfInBitFormat;
}





void ODPIP::updateValueOfBestPartitionFound(int index, double value)
{
	if (valueOfBestPartitionFound[index] < value)
		valueOfBestPartitionFound[index] = value;
}

double ODPIP::getValueOfBestPartitionFound(int index)
{
	return valueOfBestPartitionFound[index];
}


void ODPIP::set_max_f(int index, double value)
{
	max_f[index] = value;
}

double ODPIP::get_max_f(int index) 
{
	return max_f[index];
}

void ODPIP::init_max_f(std::vector<std::vector<double>> maxValueForEachSize)
{
	for (unsigned int i = 0; i < numPlayers; i++)
		set_max_f(i, 0);
	for (unsigned int i = 0; i < numPlayers; i++)
	{
		double value = getCoalitionValue((1 << i));
		if (get_max_f(0) < value)
			set_max_f(0, value);
	}

	for (unsigned int i = 1; i < numPlayers; i++)
		set_max_f(i, maxValueForEachSize[i][0]);
}



void ODPIP::initValueOfBestPartition(std::vector<std::vector<double>> maxValueForEachSize)
{
	valueOfBestPartitionFound[0] = 0;

	init_max_f(maxValueForEachSize);

	int valueOfBestPartitionFoundSize = (int)valueOfBestPartitionFound.size();
	for (int coalitionInBitFormat = valueOfBestPartitionFoundSize - 1; coalitionInBitFormat >= 1; coalitionInBitFormat--)
		valueOfBestPartitionFound[coalitionInBitFormat] = getCoalitionValue((int)coalitionInBitFormat);

}



