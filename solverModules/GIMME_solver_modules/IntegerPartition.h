#pragma once
#include <vector>
#include <cmath>
#include "ElementOfMultiset.h"

class General;

class IntegerPartition
{
public:
	std::vector<int> partsSortedAscendingly;
	std::vector<int> sortedMultiplicity;
	std::vector<int> sortedUnderlyingSet;
	int sortedUnderlyingSetInBitFormat;
	int numPlayers;
	std::vector<ElementOfMultiset> sortedMultiset;
	std::vector<int> tempIntegersThatResultedFromASplit;

	IntegerPartition() {}
	IntegerPartition(std::vector<int> parts);

	std::vector<IntegerPartition> getListOfDirectedlyConnectedIntegerPartitions(int largestIntegerBeingSplit, int prev_largestIntegerBeingSplit);
	int getNumOfDirectedlyConnectedIntegerPartitions(int largestIntegerBeingSplit, int prev_largestIntegerBeingSplit);

	static std::vector<std::vector<std::vector<int>>> getIntegerPartitions(int n);

	static int getNumOfIntegerPartitionsInLevel(int n, int level);
	static int getNumOfIntegerPartitions(int n);

	bool contains(std::vector<ElementOfMultiset> multiset);

private:
	static std::vector<std::vector<std::vector<int>>> allocateMemoryForIntegerPartitions(int n);
	static int getNumOfIntegerPartitionsInLevel_additionalParameter(int n, int level, int M);
	static void fill_x_in_partitions(std::vector<int> x, std::vector<std::vector<std::vector<int>>>& integerPartitions, int m, std::vector<int> &indexOfNewPartition);

};

