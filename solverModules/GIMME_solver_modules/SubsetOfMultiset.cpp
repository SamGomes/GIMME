#include "SubsetOfMultiset.h"

SubsetOfMultiset::SubsetOfMultiset(std::vector<ElementOfMultiset> multiset, int sizeOfSubsets, bool keepTrackOfNumOfInstancesOutsideSubset)
{
	this->multiset = multiset;
	this->sizeOfSubsets = sizeOfSubsets;
	this->keepTrackOfNumOfInstancesOutsideSubset = keepTrackOfNumOfInstancesOutsideSubset;
	resetParameters();
}

void SubsetOfMultiset::resetParameters()
{
	currentSubsetIsFirstSubset = true;

	multisetWithIncrementalElements = std::vector<ElementOfMultiset>(multiset.size());
	for (int i = 0; i < multiset.size(); i++)
		multisetWithIncrementalElements[i] = ElementOfMultiset(i + 1, multiset[i].repetition);

	setLastSubset();
	currentSubset = std::vector<ElementOfMultiset>(multiset.size());

	totalNumOfElementsInMultiset = 0;
	for (int i = 0; i < multiset.size(); i++)
		totalNumOfElementsInMultiset += multiset[i].repetition;
}

std::vector<ElementOfMultiset> SubsetOfMultiset::getNextSubset()
{
	if (currentSubsetIsFirstSubset)
	{
		setCurrentSubsetToFirstSubset();
		currentSubsetIsFirstSubset = false;
		return prepareResult();
	}
	else
	{
		int totalNumberOfElementsSeenSoFar = 0;
		int indexInLastSubset = (int)lastSubset.size() - 1;
		for (int indexInCurrentSubset = numOfUniqueElementsInCurrentSubset - 1; indexInCurrentSubset >= 0; indexInCurrentSubset++)
		{
			if (currentSubset[indexInCurrentSubset].element != lastSubset[indexInLastSubset].element)
			{
				if (currentSubset[indexInCurrentSubset].repetition > 1)
				{
					currentSubset[indexInCurrentSubset].repetition--;
					currentSubset[indexInCurrentSubset + 1].element = currentSubset[indexInCurrentSubset].element;
					currentSubset[indexInCurrentSubset + 1].repetition = 1;
					numOfUniqueElementsInCurrentSubset++;
					fillRemainingAgents(totalNumberOfElementsSeenSoFar, indexInCurrentSubset + 1);
				}
				else
				{
					currentSubset[indexInCurrentSubset].element++;
					fillRemainingAgents(totalNumberOfElementsSeenSoFar, indexInCurrentSubset);
				}
				return prepareResult();
			}
			else 
			{
				if (currentSubset[indexInCurrentSubset].repetition < lastSubset[indexInLastSubset].repetition)
				{
					totalNumberOfElementsSeenSoFar += currentSubset[indexInCurrentSubset].repetition;
					numOfUniqueElementsInCurrentSubset--;
					indexInCurrentSubset--;

					if (currentSubset[indexInCurrentSubset].repetition > 1)
					{
						currentSubset[indexInCurrentSubset].repetition--;
						currentSubset[indexInCurrentSubset + 1].element = currentSubset[indexInCurrentSubset].element + 1;
						currentSubset[indexInCurrentSubset + 1].repetition = 1;
						numOfUniqueElementsInCurrentSubset++;
						fillRemainingAgents(totalNumberOfElementsSeenSoFar, indexInCurrentSubset + 1);
					}
					else 
					{
						currentSubset[indexInCurrentSubset].element++;
						fillRemainingAgents(totalNumberOfElementsSeenSoFar, indexInCurrentSubset);
					}
					return prepareResult();
				}
				else 
				{
					totalNumberOfElementsSeenSoFar += currentSubset[indexInCurrentSubset].repetition;
					indexInLastSubset--;
					numOfUniqueElementsInCurrentSubset--;
				}
			}
		}
		return std::vector<ElementOfMultiset>();
		
	}
}




void SubsetOfMultiset::fillRemainingAgents(int totalNumOfAgentsToBeAdded, int indexAtWhichToStartFilling)
{
	if (totalNumOfAgentsToBeAdded == 0) {
		return;
	}
	int firstUniqueAgentToBeAdded = currentSubset[indexAtWhichToStartFilling].element;

	int max = multisetWithIncrementalElements[firstUniqueAgentToBeAdded - 1].repetition - currentSubset[indexAtWhichToStartFilling].repetition;
	if (max > 0) 
	{
		if (totalNumOfAgentsToBeAdded <= max) 
		{
			currentSubset[indexAtWhichToStartFilling].repetition += totalNumOfAgentsToBeAdded;
			return;
		}
		else 
		{
			currentSubset[indexAtWhichToStartFilling].repetition += max;
			totalNumOfAgentsToBeAdded -= max;
		}
	}
	int k = 1;
	do 
	{
		numOfUniqueElementsInCurrentSubset++;
		if (totalNumOfAgentsToBeAdded <= multisetWithIncrementalElements[firstUniqueAgentToBeAdded + k - 1].repetition) 
		{
			currentSubset[k + indexAtWhichToStartFilling] = ElementOfMultiset(firstUniqueAgentToBeAdded + k, totalNumOfAgentsToBeAdded);
			break;
		}
		else 
		{
			currentSubset[k + indexAtWhichToStartFilling] = ElementOfMultiset(firstUniqueAgentToBeAdded + k, multisetWithIncrementalElements[firstUniqueAgentToBeAdded + k - 1].repetition);
			totalNumOfAgentsToBeAdded -= multisetWithIncrementalElements[firstUniqueAgentToBeAdded + k - 1].repetition;
			k++;
		}
	} while (true);
}


void SubsetOfMultiset::setCurrentSubsetToFirstSubset()
{
	currentSubset = std::vector<ElementOfMultiset>(multisetWithIncrementalElements.size(), ElementOfMultiset(0, 0));

	if (keepTrackOfNumOfInstancesOutsideSubset)
		numOfInstancesOutsideSubset = new int[multisetWithIncrementalElements.size()];

	int totalNumOfAgentsToBeAdded = sizeOfSubsets;
	int i = 0;
	for (int j = 0; j < multisetWithIncrementalElements.size(); j++)
	{
		if (totalNumOfAgentsToBeAdded <= multisetWithIncrementalElements[j].repetition)
		{
			currentSubset[i] = ElementOfMultiset(multisetWithIncrementalElements[j].element, totalNumOfAgentsToBeAdded);
			if (keepTrackOfNumOfInstancesOutsideSubset)
				numOfInstancesOutsideSubset[i] = multisetWithIncrementalElements[j].repetition - totalNumOfAgentsToBeAdded;
			break;

		}
		else {
			currentSubset[i] = ElementOfMultiset(multisetWithIncrementalElements[j].element, multisetWithIncrementalElements[j].repetition);
			totalNumOfAgentsToBeAdded -= multisetWithIncrementalElements[j].repetition;
			i++;

			if (keepTrackOfNumOfInstancesOutsideSubset)
				numOfInstancesOutsideSubset[i] = 0;
		}
	}
	numOfUniqueElementsInCurrentSubset = i + 1;
}

void SubsetOfMultiset::setLastSubset()
{
	std::vector<ElementOfMultiset> temp(multisetWithIncrementalElements.size(), ElementOfMultiset(0, 0));

	int totalNumOfAgentsToBeAdded = sizeOfSubsets;
	int i = (int)temp.size() - 1;
	for (int j = (int)multisetWithIncrementalElements.size() - 1; j >= 0; j--)
	{
		if (totalNumOfAgentsToBeAdded <= multisetWithIncrementalElements[j].repetition)
		{
			temp[i] = ElementOfMultiset(multisetWithIncrementalElements[j].element, totalNumOfAgentsToBeAdded);
			break;
		}
		else
		{
			temp[i] = ElementOfMultiset(multisetWithIncrementalElements[j].element, multisetWithIncrementalElements[j].repetition);
			totalNumOfAgentsToBeAdded -= multisetWithIncrementalElements[j].repetition;
			i--;
		}
	}

	lastSubset = std::vector<ElementOfMultiset>(multisetWithIncrementalElements.size() - i);
	for (int j = (int)lastSubset.size() - 1; j >= 0; j--) {
		lastSubset[j] = temp[temp.size() - lastSubset.size() + j];
	}
}


std::vector<ElementOfMultiset> SubsetOfMultiset::prepareResult()
{
	std::vector<ElementOfMultiset> subsetWithOriginalElements(numOfUniqueElementsInCurrentSubset);
	if (keepTrackOfNumOfInstancesOutsideSubset)
		numOfInstancesOutsideSubset = new int[numOfUniqueElementsInCurrentSubset];

	for (int i = 0; i < numOfUniqueElementsInCurrentSubset; i++)
	{
		ElementOfMultiset originalElement = multiset[currentSubset[i].element - 1];
		subsetWithOriginalElements[i] = ElementOfMultiset(originalElement.element, currentSubset[i].repetition);

		if (keepTrackOfNumOfInstancesOutsideSubset)
			numOfInstancesOutsideSubset[i] = originalElement.repetition - currentSubset[i].repetition;
	}

	return subsetWithOriginalElements;
}
