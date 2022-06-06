#pragma once
#include <vector>
#include "General.h"

class CLink
{
public:
	int numPlayers;
	double* coalitionValues;
	int minNumberOfPlayersPerGroup;
	int maxNumberOfPlayersPerGroup;
	int minNumGroups;
	int maxNumGroups;

	std::vector<bool> feasibleCoalitions;

	std::vector<long> optimalCSInBitFormat;
	std::vector<std::vector<double>> PL;

	CLink() {}
	CLink(int numPlayers, double* coalitionValues, int minNumberOfPlayersPerGroup, int maxNumberOfPlayersPerGroup);

	void CLinkAlgorithm();
	void initializePL();
	double lf(int coalition1, int coalition2);

private:
	void finalize();

};

