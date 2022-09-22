#include "IntegerPartition.h"
#include "General.h"


IntegerPartition::IntegerPartition(std::vector<int> parts) 
{
	numPlayers = 0;
	for (int i = 0; i < parts.size(); i++)
	{
		numPlayers += parts[i];
	}

	partsSortedAscendingly = General::sortArray(parts);
	sortedUnderlyingSet = General::getUnderlyingSet(partsSortedAscendingly);
	sortedMultiplicity.resize(sortedUnderlyingSet.size());

	int indexInMultiplicity = 0;
	sortedMultiplicity[indexInMultiplicity] = 1;

	for (int i = 1; i < partsSortedAscendingly.size(); i++) 
	{
		if (partsSortedAscendingly[i] == partsSortedAscendingly[i - 1])
			sortedMultiplicity[indexInMultiplicity]++;
		else 
		{
			indexInMultiplicity++;
			sortedMultiplicity[indexInMultiplicity] = 1;
		}
	}
	sortedUnderlyingSetInBitFormat = General::convertCombinationFromByteToBitFormat(sortedUnderlyingSet);
	sortedMultiset.resize(sortedMultiplicity.size());
	for (int i = 0; i < sortedMultiplicity.size(); i++) 
	{
		sortedMultiset[i] = ElementOfMultiset(sortedUnderlyingSet[i], sortedMultiplicity[i]);
	}
}


std::vector<IntegerPartition> IntegerPartition::getListOfDirectedlyConnectedIntegerPartitions(int largestIntegerBeingSplit, int prev_largestIntegerBeingSplit) 
{
	int counter = getNumOfDirectedlyConnectedIntegerPartitions(largestIntegerBeingSplit, prev_largestIntegerBeingSplit);
	if (counter == 0)
		return std::vector<IntegerPartition>();

	std::vector<IntegerPartition> listOfDirectlyConnectedIntegerPartitions(counter);

	if (sortedUnderlyingSet[0] == numPlayers) 
	{
		int index = 0;
		for (int i = 1; i <= (int)floor(numPlayers / (double)2); i++)
		{
			std::vector<int> newSortedParts;
			newSortedParts.push_back(i);
			newSortedParts.push_back(numPlayers - i);
			listOfDirectlyConnectedIntegerPartitions[index] = IntegerPartition(newSortedParts);
			listOfDirectlyConnectedIntegerPartitions[index].tempIntegersThatResultedFromASplit = std::vector<int> {i, numPlayers - i};
			index++;

		}
	}
	else {
		int index = 0;
		for (int i = 0; i < sortedUnderlyingSet.size(); i++) 
		{
			int curPart = sortedUnderlyingSet[i];
			if ((curPart > largestIntegerBeingSplit) || (curPart <= prev_largestIntegerBeingSplit))
				continue;

			for (int j = 1; j <= (int)floor(curPart / (double)2); j++) 
			{
				int smallHalf = (int)j;
				int largeHalf = (int)(curPart - j);
				if (largeHalf > numPlayers - smallHalf - largeHalf)
					continue;

				std::vector<int> newSortedParts(partsSortedAscendingly.size() + 1);
				int i1 = 0;
				int i2 = 0;
				while (partsSortedAscendingly[i1] < smallHalf) 
				{
					newSortedParts[i2] = partsSortedAscendingly[i1];
					i1++;
					i2++;
				}
				newSortedParts[i2] = smallHalf;
				i2++;
				while (partsSortedAscendingly[i1] < largeHalf) 
				{
					newSortedParts[i2] = partsSortedAscendingly[i1];
					i1++;
					i2++;
				}
				newSortedParts[i2] = largeHalf;
				i2++;
				bool curPartHasBeenSeen = false;
				while (i1 < partsSortedAscendingly.size()) {
					if ((partsSortedAscendingly[i1] == curPart) && (curPartHasBeenSeen == false)) 
					{
						curPartHasBeenSeen = true;
						i1++;
					}
					else 
					{
						newSortedParts[i2] = partsSortedAscendingly[i1];
						i1++;
						i2++;
					}
				}

				listOfDirectlyConnectedIntegerPartitions[index] = IntegerPartition(newSortedParts);
				listOfDirectlyConnectedIntegerPartitions[index].tempIntegersThatResultedFromASplit = std::vector<int>{smallHalf, largeHalf};
				index++;
			}
		}
	}
	return listOfDirectlyConnectedIntegerPartitions;
}

int IntegerPartition::getNumOfDirectedlyConnectedIntegerPartitions(int largestIntegerBeingSplit, int prev_largestIntegerBeingSplit) 
{
	if (sortedUnderlyingSet[0] == numPlayers)
		return (int)floor(numPlayers / (double)2);

	int counter = 0;
	for (int i = 0; i < sortedUnderlyingSet.size(); i++) 
	{
		if ((sortedUnderlyingSet[i] > largestIntegerBeingSplit) || (sortedUnderlyingSet[i] <= prev_largestIntegerBeingSplit))
			continue;
		
		for (int j = 1; j <= (int)floor(sortedUnderlyingSet[i] / (double)2); j++) 
		{
			int smallHalf = j;
			int largeHalf = (int)(sortedUnderlyingSet[i] - j);
			if (largeHalf > numPlayers - smallHalf - largeHalf)
				continue;
			counter++;
		}
	}
	return counter;
}

std::vector<std::vector<std::vector<int>>> IntegerPartition::getIntegerPartitions(int n)
{
	std::vector<std::vector<std::vector<int>>> integerPartitions = allocateMemoryForIntegerPartitions(n);

	std::vector<int> indexOfNewPartition(n);
	for (int i = 0; i < n; i++)
		indexOfNewPartition[i] = 0;

	std::vector<int> x(n + 1);
	for (int i = 1; i <= n; i++) x[i] = 1;

	x[1] = n; int m = 1; int h = 1;
	fill_x_in_partitions(x, integerPartitions, m, indexOfNewPartition);


	while (x[1] != 1)
	{
		if (x[h] == 2) 
		{
			m = m + 1; x[h] = 1; h = h - 1;
		}
		else 
		{
			int r = x[h] - 1; int t = m - h + 1; x[h] = r;
			while (t >= r)
			{
				h = h + 1;
				x[h] = r;
				t = t - r;
			}
			if (t == 0) m = h;
			else 
			{
				m = h + 1;
				if (t > 1) 
				{
					h = h + 1;
					x[h] = t;
				}
			}

		}
		fill_x_in_partitions(x, integerPartitions, m, indexOfNewPartition);
	
	}
	return integerPartitions;

}


std::vector<std::vector<std::vector<int>>> IntegerPartition::allocateMemoryForIntegerPartitions(int n) 
{
	int* numOfIntegerPartitionsInLevel = new int[n];
	
	for (int level = 1; level <= n; level++) 
		numOfIntegerPartitionsInLevel[level - 1] = getNumOfIntegerPartitionsInLevel(n, level);
	
	std::vector<std::vector<std::vector<int>>> integerPartitions(n);
	for (int level = 1; level <= n; level++) 
	{
		integerPartitions[level - 1].resize(numOfIntegerPartitionsInLevel[level - 1]);
		for (int i = 0; i < numOfIntegerPartitionsInLevel[level - 1]; i++)
		{
			integerPartitions[level - 1][i].resize(level);
		}
	}

	return integerPartitions;
}


int IntegerPartition::getNumOfIntegerPartitions(int n) 
{
	int numOfIntegerPartitions = 0;
	for (int level = 1; level <= n; level++) 
	{
		numOfIntegerPartitions += getNumOfIntegerPartitionsInLevel(n, level);
	}

	return numOfIntegerPartitions;
}


int IntegerPartition::getNumOfIntegerPartitionsInLevel(int n, int level) {
	return(getNumOfIntegerPartitionsInLevel_additionalParameter(n, level, (int)(n - level + 1)));
}


int IntegerPartition::getNumOfIntegerPartitionsInLevel_additionalParameter(int n, int level, int M) 
{
	if ((level == 1) || (level == n)) 
		return 1;

	int sum = 0;
	for (int M1 = (int)ceil(n / (double)level); M1 <= fmin(n - level + 1, M); M1++)
	{
		sum += getNumOfIntegerPartitionsInLevel_additionalParameter((int)(n - M1), (int)(level - 1), M1);
	}

	return sum;
}


void IntegerPartition::fill_x_in_partitions(std::vector<int> x, std::vector<std::vector<std::vector<int>>>& integerPartitions, int m, std::vector<int> &indexOfNewPartition)
{
	for (int i = 1; i <= m; i++)
		integerPartitions[m - 1][indexOfNewPartition[m - 1]][i - 1] = (int)x[i];
	indexOfNewPartition[m - 1]++;
}


bool IntegerPartition::contains(std::vector<ElementOfMultiset> multiset)
{
	if (sortedMultiset.size() < multiset.size()) return false;

	for (int i = 0; i < multiset.size(); i++) {
		bool found = false;
		for (int j = 0; j < sortedMultiset.size(); j++) 
		{
			if (sortedMultiset[j].element == multiset[i].element) 
			{
				if (sortedMultiset[j].repetition < multiset[i].repetition) 
				{
					return false;
				}
				found = true;
				break;
			}
		}
		if (found == false) {
			return false;
		}
	}
	return true;
}
