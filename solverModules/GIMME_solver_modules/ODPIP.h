#pragma once
#include <vector>
#include <cmath>
#include "IntegerPartitionGraph.h"
#include "General.h"
#include "SubsetOfMultiset.h"
#include <iostream>

class ODPIP
{
public:
	unsigned int numPlayers;
	double* coalitionValues;
	unsigned int minNumberOfPlayersPerGroup;
	unsigned int maxNumberOfPlayersPerGroup;
	std::vector<bool> feasibleCoalitions;


	// IP Variables
	int ipNumOfExpansions;
	double ipValueOfBestCSFound;
	std::vector<std::vector<int>> ipBestCSFound;
	int totalNumOfExpansions;
	IntegerPartitionGraph* ipIntegerPartitionGraph;
	double ipUpperBoundOnOptimalValue;
	double ipLowerBoundOnOptimalValue;

	// ODP Variables
	double* max_f;
	std::vector<double> valueOfBestPartitionFound;
	double odpValueOfBestCSFound;
	std::vector<std::vector<int>> odpBestCSFound;

	int odpMaxSizeComputedSoFar = 1;


	std::vector<long> bestCSInBitFormat;


	ODPIP() {}
	~ODPIP();

	ODPIP(int numPlayers, double* coalitionValues, int minNumberOfPlayersPerGroup, int maxNumberOfPlayersPerGroup, std::vector<int> requiredJoinedPlayers, std::vector<int> restrictedJointPlayers);

	int* IP();

	void ODP();

	void updateIpSolution(std::vector<std::vector<int>> CS, double value);
	void updateOdpSolution(std::vector<std::vector<int>> CS, double value);
	void computeAverage(std::vector<double> &avgValueForEachSize);
	
	void initValueOfBestPartition(std::vector<std::vector<double>> maxValueForEachSize);
		
	double getCoalitionValue(int coalitionInBitFormat);
	double getCoalitionValue(std::vector<int> coalitionInByteFormat);
	double getCoalitionStructureValue(std::vector<std::vector<int>> coalitionStructure);

	void updateValueOfBestPartitionFound(int index, double value);
	double getValueOfBestPartitionFound(int index);

	void set_max_f(int index, double value);
	double get_max_f(int index);
	void init_max_f(std::vector<std::vector<double>> maxValueForEachSize);

	long computeTotalNumOfExpansions();

	std::vector<long> getBestCSFoundInBitFormat();
	std::vector<std::vector<int>> getOptimalSplit(std::vector<std::vector<int>> CS);


private:
	void searchFirstAndLastLevel();
	void updateNumOfSearchedAndRemainingCoalitionsAndCoalitionStructures();
	std::vector<std::vector<double>> setMaxValueForEachSize();
	std::vector<std::vector<Subspace>> initSubspaces(std::vector<double> avgValueForEachSize, std::vector<std::vector<double>> maxValueForEachSize);

	long disableSubspacesThatWereSearchedWhileScanningTheInput();
	long disableSubspacesWithUBLowerThanTheHighestLB();
	long disableSubspacesReachableFromBottomNode();
	void setUpperAndLowerBoundsOnOptimalValue();

	std::vector<Node*> getListOfSortedNodes(std::vector<std::vector<Subspace>> subspaces);
	Node* getFirstEnabledNode(std::vector<Node*> sortedNodes);
	std::vector<ElementOfMultiset> getRelevantNodes(Node* node, int numOfIntegersToSplitAtTheEnd);

	void putSubsetAtTheBeginning(Node* node, std::vector<ElementOfMultiset> subset);
	std::vector<double> computeSumOfMax_splitOneInteger(Node* node, std::vector<std::vector<double>> maxValueForEachSize);
	std::vector<double> computeSumOfMax_splitNoIntegers(std::vector<int> integers, std::vector<std::vector<double>> maxValueForEachSize);

	void evaluateSplits(std::vector<int> coalitionInByteFormat, int coalitionSize);
	int evaluateSplitsOfGrandCoalition();
	std::vector<int> getOptimalSplit(int coalitionInBitFormat, int bestHalfOfCoalition);
	int getBestHalf(int coalitionInBitFormat);

	void finalize();

};
