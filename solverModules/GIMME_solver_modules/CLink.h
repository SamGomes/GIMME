#pragma once
#include <limits>
#include <math.h>
#include <vector>
#include "General.h"

class CLink
{
public:
	unsigned int numPlayers;
	double* coalitionValues;
	unsigned int minNumberOfPlayersPerGroup;
	unsigned int maxNumberOfPlayersPerGroup;
	unsigned int minNumGroups;
	unsigned int maxNumGroups;

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

