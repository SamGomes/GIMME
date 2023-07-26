#include "General.h"

using namespace std;

static vector<vector<long>> pascalMatrix;

vector<vector<long>> General::initPascalMatrix(int numOfLines, int numOfColumns) {
	if ((pascalMatrix.empty()) || numOfLines > pascalMatrix.size()) {
		if (pascalMatrix.empty()) {
			pascalMatrix = vector<vector<long>>(numOfLines);
			for (int i = 0; i < numOfLines; i++) {
				pascalMatrix[i] = vector<long>(numOfColumns);
				pascalMatrix[i][0] = 1;
			}

			for (int j = 1; j < numOfColumns; j++) {
				pascalMatrix[0][j] = j + 1;
			}

			for (int i = 1; i < numOfLines; i++)
				for (int j = 1; j < numOfColumns; j++)
					pascalMatrix[i][j] = pascalMatrix[i - 1][j] + pascalMatrix[i][j - 1];
		}

		else
		{
			vector<vector<long>> prev_pascalMatrix = pascalMatrix;
			int prev_numLines = (int)prev_pascalMatrix.size();
			int prev_numColumns = (int)prev_pascalMatrix[0].size();

			pascalMatrix = vector<vector<long>>(numOfLines);
			for (int i = 0; i < numOfLines; i++)
			{
				pascalMatrix[i] = vector<long>(numOfColumns);
				pascalMatrix[i][0] = 1;
			}

			for (int j = 1; j < numOfColumns; j++)
			{
				pascalMatrix[0][j] = j + 1;
			}

			for (int i = 1; i < prev_numLines; i++)
			{
				for (int j = 1; j < prev_numColumns; j++)
					pascalMatrix[i][j] = prev_pascalMatrix[i][j];

				for (int j = prev_numColumns; j < numOfColumns; j++)
					pascalMatrix[i][j] = pascalMatrix[i - 1][j] + pascalMatrix[i][j - 1];
			}

			for (int i = prev_numLines; i < numOfLines; i++)
				for (int j = 1; j < numOfColumns; j++)
					pascalMatrix[i][j] = pascalMatrix[i - 1][j] + pascalMatrix[i][j - 1];
		}
	}
	return pascalMatrix;
}


vector<int> General::sortArray(vector<int> array) {
	int arrayLength = (int)array.size();
	vector<int> sortedArray(arrayLength);

	for (int i = 0; i < arrayLength; i++)
		sortedArray[i] = array[i];

	for (int i = arrayLength - 1; i >= 0; i--) {
		int highestIndex = i;
		for (int j = i; j >= 0; j--) {
			if (sortedArray[highestIndex] < sortedArray[j])
				highestIndex = j;
		}

		int temp = sortedArray[i];
		sortedArray[i] = sortedArray[highestIndex];
		sortedArray[highestIndex] = temp;
	}

	return sortedArray;
}

vector<int> General::getUnderlyingSet(vector<int> array) {
	int arrayLength = (int)array.size();
	int numOfUniqueElements = 0;
	vector<int> uniqueElements(arrayLength, 0);

	for (int i = 0; i < arrayLength; i++) {
		bool weHaveSeenThisElementBefore = false;
		for (int j = 0; j < numOfUniqueElements; j++) {
			if (uniqueElements[j] == array[i]) {
				weHaveSeenThisElementBefore = true;
				break;
			}
		}

		if (weHaveSeenThisElementBefore == false) {
			uniqueElements[numOfUniqueElements] = array[i];
			numOfUniqueElements++;
		}
	}

	vector<int> underlyingSet(numOfUniqueElements);
	for (int i = 0; i < numOfUniqueElements; i++) {
		underlyingSet[i] = uniqueElements[i];
	}

	return underlyingSet;
}

int General::convertCombinationFromByteToBitFormat(std::vector<int> combinationInByteFormat, int combinationSize) {
	int combinationInBitFormat = 0;
	for (int i = 0; i < combinationSize; i++)
		combinationInBitFormat += 1 << (combinationInByteFormat[i] - 1);

	return combinationInBitFormat;
}

int General::convertCombinationFromByteToBitFormat(std::vector<int> combinationInByteFormat) {
	return convertCombinationFromByteToBitFormat(combinationInByteFormat, (int)combinationInByteFormat.size());
}



std::vector<int> General::convertCombinationFromBitToByteFormat(int combinationInBitFormat, int numPlayers, int combinationSize)
{
	std::vector<int> combinationInByteFormat(combinationSize);
	int j = 0;
	for (int i = 0; i < numPlayers; i++) {
		if ((combinationInBitFormat & (1 << i)) != 0) 
		{
			combinationInByteFormat[j] = (int)(i + 1);
			j++;
		}
	}
	return combinationInByteFormat;
}

std::vector<int> General::convertCombinationFromBitToByteFormat(int combinationInBitFormat, int numPlayers)
{
	int combinationSize = getSizeOfCombinationInBitFormat(combinationInBitFormat);
	return convertCombinationFromBitToByteFormat(combinationInBitFormat, numPlayers, combinationSize);
}


std::vector<std::vector<int>> General::convertSetOfCombinationsFromBitToByteFormat(std::vector<int> setOfCombinationsInBitFormat, int numPlayers, std::vector<int> sizeOfEachCombination)
{
	std::vector<std::vector<int>> setOfCombinationsInByteFormat(setOfCombinationsInBitFormat.size());
	for (int i = 0; i < setOfCombinationsInBitFormat.size(); i++)
		setOfCombinationsInByteFormat[i] = convertCombinationFromBitToByteFormat(setOfCombinationsInBitFormat[i], numPlayers);

	return setOfCombinationsInByteFormat;
}

std::vector<std::vector<int>> General::convertSetOfCombinationsFromBitToByteFormat(std::vector<int> setOfCombinationsInBitFormat, int numPlayers)
{
	std::vector<int> sizeOfEachCombination(setOfCombinationsInBitFormat.size());
	for (int i = (int)setOfCombinationsInBitFormat.size() - 1; i >= 0; i--)
		sizeOfEachCombination[i] = (int)(getSizeOfCombinationInBitFormat(setOfCombinationsInBitFormat[i]));

	return convertSetOfCombinationsFromBitToByteFormat(setOfCombinationsInBitFormat, numPlayers, sizeOfEachCombination);
}



void General::getPreviousCombination(int numPlayers, int size, std::vector<int> &combination)
{
	int maxPossibleValueForFirstAgent = (int)(numPlayers - size + 1);
	for (int i = size - 1; i >= 0; i--) {
		if (combination[i] < maxPossibleValueForFirstAgent + i) {
			combination[i]++;
			for (int j = i + 1; j < size; j++) {
				combination[j] = combination[j - 1] + 1;
			}
			break;
		}
	}
}

std::vector<int> General::getCombinationAtGivenIndex(int size, int index, int numPlayers)
{
	index++;
	initPascalMatrix(numPlayers + 1, numPlayers + 1);
	std::vector<int> M(size);
	bool done = false;

	int j = 1; int s1 = size;
	do
	{
		int x = 1; while (pascalMatrix[s1 - 1][x - 1] < index) x++;

		M[j - 1] = (int)((numPlayers - s1 + 1) - x + 1);

		if (pascalMatrix[s1 - 1][x - 1] == index)
		{
			for (int k = j; k <= size - 1; k++) M[k] = (int)(M[k - 1] + 1);
			done = true;
		}
		else
		{
			j++; index -= pascalMatrix[s1 - 1][x - 2]; s1--;
		}
	} while (done == false);
	return M;
}



int General::getCardinalityOfMultiset(std::vector<ElementOfMultiset> multiset)
{
	if (multiset.empty())
		return 0;

	int counter = 0;
	for (int i = 0; i < multiset.size(); i++)
		counter += multiset[i].repetition;

	return counter;
}



long General::binomialCoefficient(int x, int y) {
	if (x == y) return 1;

	initPascalMatrix(x, x);

	return pascalMatrix[x - y - 1][y];
}

unsigned int General::getSizeOfCombinationInBitFormat(int combinationInBitFormat) {
	int count = 0;
	while (combinationInBitFormat) {
		count += combinationInBitFormat & 1;
		combinationInBitFormat >>= 1;
	}

	return count;
}

std::vector<int> General::getCombinationOfGivenSizeInBitFormat(int numPlayers, int size)
{
	std::vector<int> onesBeforeIndex(numPlayers + 1);
	for (int k = numPlayers; k > 0; k--)
		onesBeforeIndex[k] = (1 << k) - 1;

	std::vector<int> list(binomialCoefficient(numPlayers, size));

	int index = (int)list.size() - 1;

	list[index] = 0;
	for (int i = 1; i <= size; i++)
		list[index] += (1 << (i - 1));

	int maxPossibleValueForFirstAgent = numPlayers - size + 1;
	while (index > 0)
	{
		int i = size - 1;
		for (int k = numPlayers; k > 0; k--)
		{
			if ((list[index] & (1 << (k-1))) != 0)
			{
				if (k < maxPossibleValueForFirstAgent + i)
				{
					index--;

					list[index] = (list[index + 1] & onesBeforeIndex[k - 1]);

					list[index] += (1 << k);

					for (int j = 1; j < size - i; j++)
						list[index] += (1 << (k + j));

					i--;
					break;
				}
				i--;
			}
		}
	}

	return list;
}



std::vector<bool> General::generateFeasibleCoalitionsInBitFormat(int numPlayers, int minNumberOfPlayersPerGroup, int maxNumberOfPlayersPerGroup, std::vector<int> requiredJoinedPlayers, std::vector<int> restrictedJointPlayers, double* coalitionValues)
{
	std::vector<bool> feasibleCoalitions((unsigned int)pow(2, numPlayers), false);

	for (int curSize = minNumberOfPlayersPerGroup; curSize <= maxNumberOfPlayersPerGroup; curSize++)
	{
		
		std::vector<int> combinationsOfCurSize = getCombinationOfGivenSizeInBitFormat(numPlayers, curSize);
		for (int i = 0; i < combinationsOfCurSize.size(); i++)
		{
			feasibleCoalitions[combinationsOfCurSize[i]] = true;
			coalitionValues[combinationsOfCurSize[i]] += 2;
		}
		
	}

	if (requiredJoinedPlayers.empty() && restrictedJointPlayers.empty())
		return feasibleCoalitions;

	for (int i = 0; i < feasibleCoalitions.size(); i++)
	{
		//if (i == 49538)
		//	printf("1\n");

		if (feasibleCoalitions[i]) 
		{
			for (std::vector<int>::iterator required = requiredJoinedPlayers.begin(); required != requiredJoinedPlayers.end(); ++required)
			{
				/*if (i == 49538)
					printf("%ld\n", *required);*/
				int isValidCoalition = i & *required;
				if (isValidCoalition != 0 && isValidCoalition != *required)
				{
				/*	if (i == 49538)
						printf("2\n");*/

					feasibleCoalitions[i] = false;
					coalitionValues[i] = 0.1;
				}

				else if (feasibleCoalitions[i] && isValidCoalition) {
					/*if (i == 49538)
						printf("3\n");*/

					coalitionValues[i] += 1;
				}
			}

			
			for (std::vector<int>::iterator restricted = restrictedJointPlayers.begin(); restricted != restrictedJointPlayers.end(); ++restricted)
			{
				int isValidCoalition = i & *restricted;
				if (isValidCoalition == *restricted)
				{
					feasibleCoalitions[i] = false;
					coalitionValues[i] = 0.1;
					break;
				}
			}
		}
	}

	
	return feasibleCoalitions;
}

bool General::vectorContainsCoalition(std::vector<int> vector, int coalitionInBitFormat)
{
	return std::find(vector.begin(), vector.end(), coalitionInBitFormat) != vector.end();
}
