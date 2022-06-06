#pragma once
#include <vector>
#include "ElementOfMultiset.h"

class SubsetOfMultiset
{
public:
	SubsetOfMultiset() {}
	SubsetOfMultiset(std::vector<ElementOfMultiset> multiset, int sizeOfSubsets, bool keepTrackOfNumOfInstancesOutsideSubset);

	void resetParameters();
	std::vector<ElementOfMultiset> getNextSubset();


private:
	std::vector<ElementOfMultiset> multiset;
	int sizeOfSubsets;
	int totalNumOfElementsInMultiset;
	bool keepTrackOfNumOfInstancesOutsideSubset;
	int* numOfInstancesOutsideSubset;
	
	std::vector<ElementOfMultiset> multisetWithIncrementalElements;

	bool currentSubsetIsFirstSubset;
	std::vector<ElementOfMultiset> currentSubset;

	int numOfUniqueElementsInCurrentSubset;
	std::vector<ElementOfMultiset> lastSubset;

	void setCurrentSubsetToFirstSubset();
	void setLastSubset();

	std::vector<ElementOfMultiset> prepareResult();
	void fillRemainingAgents(int totalNumOfAgentsToBeAdded, int indexAtWhichToStartFilling);
};

