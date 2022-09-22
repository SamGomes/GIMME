#pragma once
#include <vector>
#include "Subspace.h"
#include "General.h"

class ODPIP;
class Node;

class Subspace
{
public:
	long sizeOfSubspace;
	long totalNumOfExpansionsInSubspace;
	std::vector<int> integers;
	std::vector<int> integersSortedAscendingly;
	double UB, Avg, LB;
	bool enabled;
	double priority;
	long numOfSearchedCoalitionsInThisSubspace;
	std::vector<Node*> relevantNodes;
	bool isReachableFromBottomNode;

	Subspace() {}
	Subspace(std::vector<int> integers);
	Subspace(std::vector<int>integers, std::vector<double> avgValueForEachSize, std::vector<std::vector<double>> maxValueForEachSize, int numPlayers);

	long computeNumOfCSInSubspace(std::vector<int> integers);

	int search(ODPIP* odpip, double acceptableValue, std::vector<double> avgValueForEachSize, std::vector<double> sumOfMax, int numOfIntegersToSplit);

	long static computeTotalNumOfExpansionsInSubspace(std::vector<int> integers);

private:

	std::vector<int> init_bit(int numPlayers);

	std::vector<int> init_length_of_A(int numPlayers, std::vector<int> integers);

	std::vector<int> init_max_first_member_of_M(std::vector<int> integers, std::vector<int> lengthOfA);

	std::vector<std::vector<long>> init_numOfCombinations(std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM);

	std::vector<long> init_sumOfNumOfCombinations(std::vector<std::vector<long>> numOfCombinations, std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM);

	std::vector<long> init_numOfRemovedCombinations(std::vector<int> integers, std::vector<int> lengthOfA, std::vector<int> maxFirstMemberOfM);

	std::vector<std::vector<long>> init_increment(std::vector<int> integers, std::vector<std::vector<long>> numOfCombinations, std::vector<long> sumOf_numOfCombinations, std::vector<int> max_first_member_of_M);

	std::vector<long> init_indexToStartAt(int numOfIntegers, std::vector<long> numOfRemovedCombinations, std::vector<long> sumOf_numOfCombinations);
	std::vector<long> init_indexToStopAt(int numOfIntegers, std::vector<long> numOfRemovedCombinations);

	std::vector<long> init_index_of_M(long index, std::vector<int> integers, std::vector<std::vector<long>> increment, std::vector<int> max_first_member_of_M, std::vector<std::vector<long>> numOfCombinations, std::vector<long> numOfRemovedCombinations, std::vector<long> sumOf_numOfCombinations);

	std::vector<std::vector<int>> init_M(std::vector<long> index_of_M, std::vector<int> integers, std::vector<int> length_of_A, int numPlayers);

	std::vector<std::vector<int>> init_A(int numPlayers, std::vector<int> integers, std::vector<std::vector<int>> M, std::vector<int> length_of_A);

	std::vector<int> init_CS(std::vector<std::vector<int>> M, std::vector<std::vector<int>> A, std::vector<int> length_of_A, std::vector<int> bit, int numOfIntegers);

	std::vector<int> init_sumOf_agents(int numOfIntegers, std::vector<int> CS);

	std::vector<double> init_sumOf_values(int numOfIntegers, std::vector<int> CS, ODPIP* odpip);

	void setTheLastTwoCoalitionsInCS(std::vector<int> &CS, std::vector<int> M, std::vector<std::vector<int>> A, int numOfIntegers, std::vector<int> bit);

	void search_useBranchAndBound(ODPIP* odpip, double acceptableValue, std::vector<double> sumOfMax, int numOfIntegersToSplit);

	void searchFirstOrLastLevel(ODPIP* odpip, std::vector<int> integers);

	void set_M_and_index_of_M(std::vector<std::vector<int>> &M, std::vector<long> &indexOfM, std::vector<int> lengthOfA, std::vector<long> indexToStartAt, int s2);

	bool useLocalBranchAndBound(ODPIP* odpip, std::vector<double> sumOf_values, std::vector<int> &sumOf_agents, int s2, int newCoalition, double valueOfNewCoalition);

	void update_A(std::vector<int> &A1, std::vector<int> &A2, int numPlayers, std::vector<int> &M, int lengthOfM);

	bool checkIfLastTwoCoalitionsSatisfyConstraints(std::vector<int> CS, std::vector<bool> feasibleCoalitions);
};

