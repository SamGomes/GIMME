#include "Subspace.h"
#include "Node.h"
#include "ODPIP.h"


Subspace::Subspace(std::vector<int> integers) {
	this->integers = integers;
	this->integersSortedAscendingly = General::sortArray(integers);
	this->sizeOfSubspace = computeNumOfCSInSubspace(integers);
}
Subspace::Subspace(std::vector<int>integers, std::vector<double> avgValueForEachSize, std::vector<std::vector<double>> maxValueForEachSize, int numPlayers) {

	this->integers = integers;

	this->integersSortedAscendingly = General::sortArray(integers);
	this->enabled = true;

	this->sizeOfSubspace = computeNumOfCSInSubspace(this->integers);

	this->totalNumOfExpansionsInSubspace = computeTotalNumOfExpansionsInSubspace(this->integers);

	if (this->integers.size() == 2) {
		int size1 = this->integers[0]; int size2 = this->integers[1];
		int numOfCombinationsOfSize1 = (int)General::binomialCoefficient(numPlayers, size1);
		int temp;
		if (size1 != size2)
			temp = numOfCombinationsOfSize1;
		else
			temp = numOfCombinationsOfSize1 / 2;

		this->numOfSearchedCoalitionsInThisSubspace = 2 * temp;
	}

	else
		if ((this->integers.size() == 1) || (this->integers.size() == numPlayers))
			this->numOfSearchedCoalitionsInThisSubspace = (long)this->integers.size();
		else
			this->numOfSearchedCoalitionsInThisSubspace = 0;

	int j = 0;
	this->UB = 0;
	for (int k = 0; k < this->integers.size(); k++) {
		if ((k > 0) && (this->integers[k] == this->integers[k - 1]))
			j++;
		else
			j = 0;

		this->UB += maxValueForEachSize[this->integers[k] - 1][j];

	}

	this->Avg = 0;
	for (int k = 0; k < this->integers.size(); k++)
		this->Avg += avgValueForEachSize[this->integers[k] - 1];

	this->LB = this->Avg;
} 

long Subspace::computeNumOfCSInSubspace(std::vector<int> integers) {
	int numPlayers = 0;
	for (int i = 0; i < integers.size(); i++) numPlayers += integers[i];

	if ((integers.size() == 1) || (integers.size() == numPlayers))
		return 1;

	std::vector<int> lengthOfA = init_length_of_A(numPlayers, integers);
	
	std::vector<int> maxFirstMemberOfM = init_max_first_member_of_M(integers, lengthOfA);
	std::vector<std::vector<long>> numOfCombinations = init_numOfCombinations(integers, lengthOfA, maxFirstMemberOfM);
	std::vector<long> sumOfNumOfCombinations = init_sumOfNumOfCombinations(numOfCombinations, integers, lengthOfA, maxFirstMemberOfM);
	std::vector<long> numOfRemovedCombinations = init_numOfRemovedCombinations(integers, lengthOfA, maxFirstMemberOfM);
	std::vector<std::vector<long>> increment = init_increment(integers, numOfCombinations, sumOfNumOfCombinations, maxFirstMemberOfM);

	long sizeOfSubspace = 0;
	
	if (numOfRemovedCombinations[0] == 0) 
		sizeOfSubspace = increment[0][0] * sumOfNumOfCombinations[0];
	else
		for (int i = 0; i < maxFirstMemberOfM[0]; i++) 
			sizeOfSubspace += increment[0][i] * numOfCombinations[0][i];
	
	return sizeOfSubspace;
}

int Subspace::search(ODPIP* odpip, double acceptableValue, std::vector<double> avgValueForEachSize, std::vector<double> sumOfMax, int numOfIntegersToSplit)
{
	search_useBranchAndBound(odpip, acceptableValue, sumOfMax, numOfIntegersToSplit);

	std::vector<std::vector<int>> CS = odpip->getOptimalSplit(odpip->ipBestCSFound);
	odpip->updateIpSolution(CS, odpip->getCoalitionStructureValue(CS));

	this->enabled = false;
	int numOfSearchedSubspaces = 1;
	
	
	if( !relevantNodes.empty() ){
		for(int i=0; i<relevantNodes.size(); i++)
			if( relevantNodes[i]->subspace.enabled ){
				relevantNodes[i]->subspace.enabled = false;
				numOfSearchedSubspaces++;
			}
	}
	

	return numOfSearchedSubspaces;
}



void Subspace::search_useBranchAndBound(ODPIP* odpip, double acceptableValue, std::vector<double> sumOfMax, int numOfIntegersToSplit)
{
	if ((integers.size() == 1) || (integers.size() == odpip->numPlayers))
	{
		searchFirstOrLastLevel(odpip, integers);
		return;
	}

	int numOfIntegers = (int)integers.size();
	long ipNumOfSearchedCoalitions_beforeSearchingThisSubspace = odpip->ipNumOfExpansions;
	int numPlayers = odpip->numPlayers;
	int numOfIntsToSplit = numOfIntegersToSplit;
	bool constraintsExist = false;
	bool this_CS_is_useless;
	double valueOfCS = 0;
	
	if (odpip->feasibleCoalitions.empty()) constraintsExist = false;
	else constraintsExist = true;
	
	std::vector<int> bit = init_bit(numPlayers);
	std::vector<int> lengthOfA = init_length_of_A(numPlayers, integers);
	std::vector<int> maxFirstMemberOfM = init_max_first_member_of_M(integers, lengthOfA);
	std::vector<std::vector<long>> numOfCombinations = init_numOfCombinations(integers, lengthOfA, maxFirstMemberOfM);
	std::vector<long> sumOfNumOfCombinations = init_sumOfNumOfCombinations(numOfCombinations, integers, lengthOfA, maxFirstMemberOfM);
	std::vector<long> numOfRemovedCombinations = init_numOfRemovedCombinations(integers, lengthOfA, maxFirstMemberOfM);
	std::vector<std::vector<long>> increment = init_increment(integers, numOfCombinations, sumOfNumOfCombinations, maxFirstMemberOfM);
	std::vector<long> indexToStartAt = init_indexToStartAt(numOfIntegers, numOfRemovedCombinations, sumOfNumOfCombinations);
	std::vector<long> indexToStopAt = init_indexToStopAt(numOfIntegers, numOfRemovedCombinations);
	std::vector<long> indexOfM = init_index_of_M(1, integers, increment, maxFirstMemberOfM, numOfCombinations, numOfRemovedCombinations, sumOfNumOfCombinations);
	std::vector<std::vector<int>> M = init_M(indexOfM, integers, lengthOfA, numPlayers);
	std::vector<std::vector<int>> A = init_A(numPlayers, integers, M, lengthOfA);
	std::vector<int> CS = init_CS(M, A, lengthOfA, bit, numPlayers);
	std::vector<int> sumOfAgents = init_sumOf_agents(numOfIntegers, CS);
	std::vector<double> sumOfValues = init_sumOf_values(numOfIntegers, CS, odpip);

	odpip->ipNumOfExpansions += (int)integers.size() - 2;

main_loop: while (true)
{
	do
	{
		setTheLastTwoCoalitionsInCS(CS, M[numOfIntegers - 2], A, numOfIntegers, bit);

		this_CS_is_useless = false;
		if ((constraintsExist) && (checkIfLastTwoCoalitionsSatisfyConstraints(CS, odpip->feasibleCoalitions) == false))
			this_CS_is_useless = true;
		// check feasability

		if (this_CS_is_useless == false)
		{
			switch (numOfIntsToSplit) {
			case 0: valueOfCS = sumOfValues[numOfIntegers - 2] + odpip->getCoalitionValue(CS[numOfIntegers - 2]) + odpip->getCoalitionValue(CS[numOfIntegers - 1]); break;
			case 1:  valueOfCS = sumOfValues[numOfIntegers - 2] + odpip->getCoalitionValue(CS[numOfIntegers - 2]) + odpip->getValueOfBestPartitionFound(CS[numOfIntegers - 1]); break;
			default: valueOfCS = sumOfValues[numOfIntegers - 2] + odpip->getValueOfBestPartitionFound(CS[numOfIntegers - 2]) + odpip->getValueOfBestPartitionFound(CS[numOfIntegers - 1]);
			}

			if (odpip->ipValueOfBestCSFound < valueOfCS)
			{
				std::vector<std::vector<int>> CSInByteFormat = General::convertSetOfCombinationsFromBitToByteFormat(CS, numPlayers, integers);
				odpip->updateIpSolution(CSInByteFormat, valueOfCS);
				if (odpip->ipValueOfBestCSFound >= acceptableValue)
				{
					numOfSearchedCoalitionsInThisSubspace = odpip->ipNumOfExpansions - ipNumOfSearchedCoalitions_beforeSearchingThisSubspace;
					return;
				}
			}
		}
		indexOfM[numOfIntegers - 2]--;
		General::getPreviousCombination(lengthOfA[numPlayers - 2], integers[numOfIntegers - 2], M[numOfIntegers - 2]);
	} while (indexOfM[numPlayers - 2] >= indexToStopAt[numOfIntegers - 2]);

	int s1 = numPlayers - 3;
	while (s1 >= 0)
	{
		if (indexOfM[s1] > indexToStopAt[s1])
		{
			for (int s2 = s1; s2 <= numPlayers - 3; s2++)
			{
				bool firstTime = true;
				do
				{
					odpip->ipNumOfExpansions++;

					if (firstTime && s2 > s1) {
						set_M_and_index_of_M(M, indexOfM, lengthOfA, indexToStartAt, s2);
						firstTime = false;
					}
					else
					{
						General::getPreviousCombination(lengthOfA[s2], integers[s2], M[s2]);
						indexOfM[s2]--;
					}

					int temp3 = 0;
					for (int j1 = integers[s2] - 1; j1 >= 0; j1--)
						temp3 |= bit[A[s2][M[s2][j1] - 1]];
					CS[s2] = temp3;

					this_CS_is_useless = false;
					if (constraintsExist)
						if (odpip->feasibleCoalitions[CS[s2]] == false)
							this_CS_is_useless = true;
					// check constraints

					if (this_CS_is_useless == false)
					{
						int newCoalition = CS[s2];
						double valueOfNewCoalition;
						if (s2 >= numOfIntegers - numOfIntsToSplit)
							valueOfNewCoalition = odpip->getValueOfBestPartitionFound(CS[s2]);
						else
							valueOfNewCoalition = odpip->getCoalitionValue(CS[s2]);

						sumOfValues[s2 + 1] = sumOfValues[s2] + valueOfNewCoalition;
						sumOfAgents[s2 + 1] = sumOfAgents[s2] + CS[s2];

						double upperBoundForRemainingAgents = sumOfMax[s2 + 2];

						if (((sumOfValues[s2 + 1] + upperBoundForRemainingAgents) - odpip->ipValueOfBestCSFound < -0.000000000005)
							|| useLocalBranchAndBound(odpip, sumOfValues, sumOfAgents, s2, newCoalition, valueOfNewCoalition))
							this_CS_is_useless = true;
					}
					if (this_CS_is_useless == false) break;
				} while (indexOfM[s2] > indexToStopAt[s2]);

				if (this_CS_is_useless)
				{
					s1 = s2 - 1;
					continue;
				}

				update_A(A[s2 + 1], A[s2], lengthOfA[s2], M[s2], integers[s2]);
			}
			int s2 = numOfIntegers - 2;
			set_M_and_index_of_M(M, indexOfM, lengthOfA, indexToStartAt, s2);

			goto main_loop;
		}
		s1--;
	}
	goto after_loop;

}
after_loop:
numOfSearchedCoalitionsInThisSubspace = odpip->ipNumOfExpansions - ipNumOfSearchedCoalitions_beforeSearchingThisSubspace;
}

bool Subspace::checkIfLastTwoCoalitionsSatisfyConstraints(std::vector<int> CS, std::vector<bool> feasibleCoalitions)
{
	if (feasibleCoalitions[CS[CS.size() - 1]] == false)
		return false;
	if (feasibleCoalitions[CS[CS.size() - 2]] == false)
		return false;
	return true;
}



void Subspace::searchFirstOrLastLevel(ODPIP* odpip, std::vector<int> integers)
{
	int numPlayers = odpip->numPlayers;
	std::vector<std::vector<int>> curCS;

	if (integers.size() == 1)
	{
		curCS = std::vector<std::vector<int>>(1);
		curCS[0] = std::vector<int>(numPlayers);
		for (int i = 0; i < numPlayers; i++)
			curCS[0][i] = (int)i + 1;
	}
	else
	{
		curCS = std::vector<std::vector<int>>(numPlayers);
		for (int i = 0; i < numPlayers; i++)
		{
			curCS[i] = std::vector<int>(1,i+1);
		}
	}

	double valueOfCurCS = odpip->getCoalitionStructureValue(curCS);
	odpip->updateIpSolution(curCS, valueOfCurCS);
}



long Subspace::computeTotalNumOfExpansionsInSubspace(std::vector<int> integers) {
	int numPlayers = 0;
	for (int i = 0; i < integers.size(); i++) {
		numPlayers += integers[i];
	}

	std::vector<int> sortedIntegers = General::sortArray(integers);

	std::vector<int> alpha(sortedIntegers.size());
	std::vector<std::vector<long>> gamma(sortedIntegers.size());
	for (int j = 0; j < sortedIntegers.size(); j++) {
		int maxIndex = 0;
		for (int k = 0; k < sortedIntegers.size(); k++) {
			if (sortedIntegers[k] == sortedIntegers[j])
				maxIndex = k;
		}

		alpha[j] = numPlayers + 1;
		for (int k = 0; k <= maxIndex; k++)
			alpha[j] -= sortedIntegers[k];

		gamma[j].resize(alpha[j]);
		for (int beta = 0; beta < alpha[j]; beta++) {
			int sumOfPreviousIntegers = 0;
			for (int k = 0; k < j; k++)
				sumOfPreviousIntegers += sortedIntegers[k];

			if (j == 0)
				gamma[j][beta] = General::binomialCoefficient(numPlayers - sumOfPreviousIntegers - beta + 1, sortedIntegers[j] - 1);

			else {
				int lambda;
				if (sortedIntegers[j] == sortedIntegers[j - 1])
					lambda = beta;
				else
					lambda = alpha[j - 1] - 1;

				long sum = 0;
				for (int k = 0; k <= lambda; k++)
					sum += gamma[j - 1][k];
				gamma[j][beta] = sum * General::binomialCoefficient(numPlayers - sumOfPreviousIntegers - beta + 1, sortedIntegers[j] - 1);
			}
		}

	}

	long numOfExpansionsInSubspace = 0;
	for (int j = 0; j < sortedIntegers.size(); j++)
		for (int beta = 0; beta < alpha[j]; beta++)
			numOfExpansionsInSubspace += gamma[j][beta];

	return numOfExpansionsInSubspace;
}





std::vector<int> Subspace::init_bit(int numPlayers) {
	std::vector<int> bit(numPlayers + 1);
	for (int i = 0; i < numPlayers; i++)
		bit[i + 1] = 1 << i;
	return bit;
}




std::vector<int> Subspace::init_length_of_A(int numPlayers, std::vector<int> integers) {
	std::vector<int> lengthOfA(integers.size());

	lengthOfA[0] = numPlayers;
	if (integers.size() > 1)
		for (int s = 1; s < integers.size(); s++)
			lengthOfA[s] = (int)(lengthOfA[s - 1] - integers[s - 1]);

	return lengthOfA;
}

std::vector<int> Subspace::init_max_first_member_of_M(std::vector<int> integers, std::vector<int> lengthOfA) {
	std::vector<int> maxFirstMemberOfM(integers.size());
	int i = (int)integers.size() - 1;

	if ((!relevantNodes.empty()) && integers.size() > 2) {
		maxFirstMemberOfM[i] = (int)(lengthOfA[i] - integers[i] + 1);
		i--;
	}

	while (i >= 0) {
		maxFirstMemberOfM[i] = (int)(lengthOfA[i] - integers[i] + 1);
		i--;
		while ((i >= 0) && (integers[i] == integers[i + 1])) {
			maxFirstMemberOfM[i] = maxFirstMemberOfM[i + 1];
			i--;
		}
	}

	return maxFirstMemberOfM;
}

std::vector<std::vector<long>> Subspace::init_numOfCombinations(std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM) {
	std::vector<std::vector<long>> numOfCombinations(integers.size());

	for (int i = 0; i < integers.size(); i++) 
	{
		if (lengthOfA[i] == integers[i]) 
		{
			numOfCombinations[i].resize(1);
			numOfCombinations[i][0] = 1;
		}
		else 
		{
			numOfCombinations[i].resize(maxFirstMemberOfM[i]);
			for (int j = 0; j < maxFirstMemberOfM[i]; j++) 
			{
				numOfCombinations[i][j] = General::binomialCoefficient((int)(lengthOfA[i] - j + 1), (int)(integers[i] - 1));
			}
		}
	}
	return numOfCombinations;
}

std::vector<long> Subspace::init_sumOfNumOfCombinations(std::vector<std::vector<long>> numOfCombinations, std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM) {
	std::vector<long> sumOfNumOfCombinations(integers.size());

	for (int i = 0; i < integers.size(); i++) {
		if (lengthOfA[i] == integers[i])
			sumOfNumOfCombinations[i] = 1;

		else {
			sumOfNumOfCombinations[i] = 0;
			for (int j = 0; j < maxFirstMemberOfM[i]; j++)
				sumOfNumOfCombinations[i] = sumOfNumOfCombinations[i] + numOfCombinations[i][j];
		}
	}
	return sumOfNumOfCombinations;
}

std::vector<long> Subspace::init_numOfRemovedCombinations(std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM) {
	std::vector<long> numOfRemovedCombinations(integers.size());

	for (int i = 0; i < integers.size(); i++) {
		if (lengthOfA[i] == integers[i])
			numOfRemovedCombinations[i] = 0;

		else {
			numOfRemovedCombinations[i] = 0;
			for (int j = 0; j < maxFirstMemberOfM[i]; j++)
				numOfRemovedCombinations[i] = numOfRemovedCombinations[i] + General::binomialCoefficient((int)(lengthOfA[i] - j + 1), (int)(integers[i] - 1));
		}
	}
	return numOfRemovedCombinations;
}

std::vector<std::vector<long>> Subspace::init_increment(std::vector<int> integers, std::vector<std::vector<long>> numOfCombinations, std::vector<long> sumOfNumOfCombinations, std::vector<int> maxFirstMemberOfM) {
	std::vector<std::vector<long>> increment(integers.size());
	increment[integers.size() - 1].resize(1);
	increment[integers.size() - 1][0] = 1;

	int s = (int)integers.size() - 2;
	while (s >= 0) {
		if ((integers[s] != integers[s + 1]) || ((s == integers.size() - 2) && (integers.size() > 2))) {
			increment[s].resize(1);
			increment[s][0] = sumOfNumOfCombinations[s + 1] * increment[s + 1][0];
			s--;
		}
		else {
			increment[s].resize(maxFirstMemberOfM[s]);
			for (int i = 0; i <= maxFirstMemberOfM[s] - 1; i++)	{
				increment[s][i] = 0;
				for (int j = i; j < maxFirstMemberOfM[s]; j++)
					increment[s][i] += numOfCombinations[s + 1][j] * increment[s + 1][0];
			}
			s--;

			while ((s >= 0) && (integers[s] == integers[s + 1]))
			{
				increment[s].resize(maxFirstMemberOfM[s]);
				for (int i = 0; i < maxFirstMemberOfM[s]; i++)
				{
					increment[s][i] = 0;
					for (int j = i; j < maxFirstMemberOfM[s]; j++)
						increment[s][i] += numOfCombinations[s + 1][j] * increment[s + 1][j];
				}
				s--;
			}

			if (s >= 0) {
				increment[s].resize(1);
				increment[s][0] = 0;
				for (int j = 0; j < maxFirstMemberOfM[s + 1]; j++) 
					increment[s][0] += numOfCombinations[s + 1][j] * increment[s + 1][j];
				s--;
			}
		}
	}
	return increment;
}

std::vector<long> Subspace::init_indexToStartAt(int numOfIntegers, std::vector<long> numOfRemovedCombinations, std::vector<long> sumOf_numOfCombinations)
{
	std::vector<long> indexToStartAt(numOfIntegers);
	for (int i = 0; i < numOfIntegers; i++)
	{
		indexToStartAt[i] = sumOf_numOfCombinations[i] + numOfRemovedCombinations[i];
	}
	return indexToStartAt;
}

std::vector<long> Subspace::init_indexToStopAt(int numOfIntegers, std::vector<long> numOfRemovedCombinations)
{
	std::vector<long> indexToStopAt(numOfIntegers);
	for (int i = 0; i < numOfIntegers; i++)
	{
		indexToStopAt[i] = numOfRemovedCombinations[i] + 1;
	}
	return indexToStopAt;
}

std::vector<long> Subspace::init_index_of_M(long index, std::vector<int> integers, std::vector<std::vector<long>> increment, std::vector<int> max_first_member_of_M, std::vector<std::vector<long>> numOfCombinations, std::vector<long> numOfRemovedCombinations, std::vector<long> sumOf_numOfCombinations)
{
	long counter1 = 0;
	long counter2 = 1;
	std::vector<long> index_of_M(integers.size());

	index_of_M[integers.size() - 1] = 1;

	int min_first_member_of_M = 0;
	for (int i = 0; i <= integers.size() - 2; i++)
	{
		if (sumOf_numOfCombinations[i] == 1)
		{
			index_of_M[i] = 1;
		}
		else
		{
			if (increment[i].size() == 1)
			{
				counter1 = 0;
				counter2 = 1;
				if (min_first_member_of_M > 0)
					for (int j = 0; j <= min_first_member_of_M - 1; j++)
						counter2 += numOfCombinations[i][j];

				long steps = (long)(ceil(index / (double)increment[i][0]) - 1);
				counter1 += steps * increment[i][0];
				counter2 += steps;

				index_of_M[i] = counter2;
				index -= counter1;

				if ((i >= integers.size() - 1) || (integers[i] != integers[i + 1])) min_first_member_of_M = 0;
			}
			else
			{
				counter1 = 0;
				counter2 = 1;
				if (min_first_member_of_M > 0)
					for (int j = 0; j < min_first_member_of_M; j++)
						counter2 += numOfCombinations[i][j];
				for (int j = min_first_member_of_M; j < max_first_member_of_M[i]; j++)
				{
					if (index <= counter1 + (numOfCombinations[i][j] * increment[i][j]))
					{
						long steps = (long)ceil((index - counter1) / (double)increment[i][j]) - 1;
						counter1 += steps * increment[i][j];
						counter2 += steps;

						index_of_M[i] = counter2;
						index -= counter1;

						if ((i < integers.size() - 1) && (integers[i] == integers[i + 1]))
							min_first_member_of_M = j;
						else
							min_first_member_of_M = 0;

						break;
					}
					else
					{
						long steps = numOfCombinations[i][j];
						counter1 += steps * increment[i][j];
						counter2 += steps;
					}
				}
			}
		}
	}
	for (int i = 0; i <= integers.size() - 1; i++)
		index_of_M[i] = (sumOf_numOfCombinations[i] + numOfRemovedCombinations[i]) - index_of_M[i] + 1;

	return index_of_M;

}

std::vector<std::vector<int>> Subspace::init_M(std::vector<long> index_of_M, std::vector<int> integers, std::vector<int> length_of_A, int numPlayers)
{
	std::vector<std::vector<long>> pascalMatrix = General::initPascalMatrix(numPlayers + 1, numPlayers + 1);

	std::vector<std::vector<int>> M = std::vector<std::vector<int>>(integers.size());
	for (int s = 0; s <= integers.size() - 1; s++)
	{
		M[s] = std::vector<int>(integers[s]);
	}

	for (int i = 0; i <= integers.size() - 1; i++)
	{
		/*1*/int j = 1; long index = index_of_M[i]; int s1 = integers[i];

		bool done = false;
		do
		{
			/*2*/ int x = 1; while (pascalMatrix[s1 - 1][x - 1] < index) x++;

			/*3*/ M[i][j - 1] = (int)((length_of_A[i] - s1 + 1) - x + 1);

			/*4*/ if (pascalMatrix[s1 - 1][x - 1] == index)
			{
				for (int k = j; k <= integers[i] - 1; k++) M[i][k] = (int)(M[i][k - 1] + 1);
				done = true;
			}
			else
			{
				j = j + 1;  index = index - pascalMatrix[s1 - 1][x - 2];  s1 = s1 - 1;
			}
		} while (done == false);
	}
	return M;
}

std::vector<std::vector<int>> Subspace::init_A(int numPlayers, std::vector<int> integers, std::vector<std::vector<int>> M, std::vector<int> length_of_A)
{
	std::vector<std::vector<int>> A = std::vector<std::vector<int>>(integers.size() - 1);
	for (int s = 0; s <= integers.size() - 2; s++)
	{
		A[s] = std::vector<int>(length_of_A[s]);
		if (s == 0)
		{
			for (int j1 = 0; j1 <= numPlayers - 1; j1++)
			{
				A[s][j1] = (int)(j1 + 1);
			}
		}
		else
		{
			int j1 = 0; int j2 = 0;
			for (int j3 = 0; j3 <= length_of_A[s - 1] - 1; j3++)
			{				
				if ((j1 >= M[s-1].size()) || (j3 + 1 != M[s - 1][j1]))
				{
					A[s][j2] = A[s - 1][j3];
					j2++;
				}
				else j1++;
			}
		}
	}
	return A;
}

std::vector<int> Subspace::init_CS(std::vector<std::vector<int>> M, std::vector<std::vector<int>> A, std::vector<int> length_of_A, std::vector<int> bit, int numOfIntegers)
{
	std::vector<int> CS(integers.size());

	for (int s = 0; s <= integers.size() - 2; s++)
	{
		CS[s] = 0;
		for (int j1 = 0; j1 < M[s].size(); j1++)
		{
			CS[s] |= bit[A[s][M[s][j1] - 1]];
		}
	}
	setTheLastTwoCoalitionsInCS(CS, M[numOfIntegers - 2], A, numOfIntegers, bit);
	return CS;
}

void Subspace::setTheLastTwoCoalitionsInCS(std::vector<int> &CS, std::vector<int> M, std::vector<std::vector<int>> A, int numOfIntegers, std::vector<int> bit)
{
	int result1 = 0;
	int result2 = 0;
	int m = integers[numOfIntegers - 2] - 1;
	int a = (int)A[numOfIntegers - 2].size() - 1;
	do
	{
		if (a == M[m] - 1)
		{
			result1 += bit[A[numOfIntegers - 2][a]];
			if (m == 0)
			{
				a--;
				break;
			}
			m--;
		}
		else
			result2 += bit[A[numOfIntegers - 2][a]];

		a--;
	} while (a >= 0);

	while (a >= 0)
	{
		result2 += bit[A[numOfIntegers - 2][a]];
		a--;
	}
	CS[numOfIntegers - 2] = result1;
	CS[numOfIntegers - 1] = result2;
}

std::vector<int> Subspace::init_sumOf_agents(int numOfIntegers, std::vector<int> CS)
{
	std::vector<int> sumOf_agents(numOfIntegers + 1);

	sumOf_agents[0] = 0;

	for (int i = 1; i <= numOfIntegers; i++)
	{
		sumOf_agents[i] = sumOf_agents[i - 1] + CS[i - 1];
	}

	return sumOf_agents;
}

std::vector<double> Subspace::init_sumOf_values(int numOfIntegers, std::vector<int> CS, ODPIP* odpip)
{
	std::vector<double> sumOf_values(numOfIntegers + 1);

	sumOf_values[0] = 0;
	for (int i = 1; i <= numOfIntegers; i++)
	{
		sumOf_values[i] = sumOf_values[i - 1] + odpip->getCoalitionValue(CS[i - 1]);
	}

	return sumOf_values;
}

void Subspace::set_M_and_index_of_M(std::vector<std::vector<int>> &M, std::vector<long> &indexOfM, std::vector<int> indexOfA, std::vector<long> indexToStartAt, int s2)
{
	indexOfM[s2] = indexToStartAt[s2];

	if (integers[s2] == integers[s2 - 1]) {
		if (M[s2 - 1][0] > 1)
			for (int j = 1; j < M[s2 - 1][0]; j++)
				indexOfM[s2] = indexOfM[s2] - General::binomialCoefficient(indexOfA[s2] - j, integers[s2] - 1);

		for (int j1 = 0; j1 < integers[s2]; j1++)
			M[s2][j1] = (int)(M[s2 - 1][0] + j1);
	}
	else
		for (int j1 = 0; j1 < integers[s2]; j1++)
			M[s2][j1] = (int)(1 + j1);
}

bool Subspace::useLocalBranchAndBound(ODPIP* odpip, std::vector<double> sumOf_values, std::vector<int> &sumOf_agents, int s2, int newCoalition, double valueOfNewCoalition)
{
	bool result = false;

	if ((odpip->getValueOfBestPartitionFound(sumOf_agents[s2 + 1]) - sumOf_values[s2 + 1] > 0.00000000005) || (odpip->getValueOfBestPartitionFound(newCoalition) - valueOfNewCoalition > 0.00000000005))
		result = true;

	if (odpip->getValueOfBestPartitionFound(sumOf_agents[s2 + 1]) - sumOf_values[s2 + 1] < -0.00000000005)
		odpip->updateValueOfBestPartitionFound(sumOf_agents[s2 + 1], sumOf_values[s2 + 1]);

	if (odpip->getValueOfBestPartitionFound(newCoalition) - valueOfNewCoalition < -0.00000000005)
		odpip->updateValueOfBestPartitionFound(newCoalition, valueOfNewCoalition);

	return(result);
}

void Subspace::update_A(std::vector<int>& A1, std::vector<int>& A2, int numPlayers, std::vector<int>& M, int lengthOfM)
{
	int j1 = 0;
	int j2 = 0;
	for (int j3 = 0; j3 < A2.size(); j3++)
	{
		if ((j1 >= lengthOfM) || ((j3 + 1) != M[j1]))
		{
			A1[j2] = A2[j3];
			j2++;
		}
		else j1++;
	}
}
