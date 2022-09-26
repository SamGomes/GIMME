#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include "ElementOfMultiset.h"

class General {
public:
	static std::vector<std::vector<long>> initPascalMatrix(int numOfLines, int numOfColumns);

	static unsigned int getSizeOfCombinationInBitFormat(int combinationInBitFormat);


	static std::vector<int> sortArray(std::vector<int> array);

	static std::vector<int> getUnderlyingSet(std::vector<int> array);

	static int convertCombinationFromByteToBitFormat(std::vector<int> combinationInByteFormat, int combinationSize);

	static int convertCombinationFromByteToBitFormat(std::vector<int> combinationInByteFormat);


	static std::vector<int> convertCombinationFromBitToByteFormat(int combinationInBitFormat, int numPlayers, int combinationSize);

	static std::vector<int> convertCombinationFromBitToByteFormat(int combinationInBitFormat, int numPlayers);

	static std::vector<std::vector<int>> convertSetOfCombinationsFromBitToByteFormat(std::vector<int> setOfCombinationsInBitFormat, int numPlayers, std::vector<int> sizeOfEachCombination);
	static std::vector<std::vector<int>> convertSetOfCombinationsFromBitToByteFormat(std::vector<int> setOfCombinationsInBitFormat, int numPlayers);



	static void getPreviousCombination(int numPlayers, int size, std::vector<int>& combination);
	static std::vector<int> getCombinationAtGivenIndex(int size, int index, int numPlayers);


	static long binomialCoefficient(int x, int y);

	static int getCardinalityOfMultiset(std::vector<ElementOfMultiset> multiset);

	static std::vector<int> getCombinationOfGivenSizeInBitFormat(int numPlayers, int size);

	static std::vector<bool> generateFeasibleCoalitionsInBitFormat(int numPlayers, int minNumberOfPlayersPerGroup, int maxNumberOfPlayersPerGroup, std::vector<int> requiredJoinedPlayers, std::vector<int> restrictedJointPlayers, double* coalitionValues);

	static bool vectorContainsCoalition(std::vector<int> vector, int coalitionInBitFormat);

};
