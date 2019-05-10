#pragma once
#include "../../Player/Player.h"

class RegressionAlg
{
public:
	PlayerState predict(InteractionsProfile profile, Player* player, void* sortFunc) { return PlayerState(); };
};

class KNNRegression: public RegressionAlg {
private:
	int numberOfNNs;
public:
	KNNRegression(int numberOfNNs);
	PlayerState predict(InteractionsProfile profile, Player* player, void* sortFunc);
};


class NeuralNetworkRegression : public RegressionAlg {
private:
public:
	NeuralNetworkRegression();
	PlayerState predict(InteractionsProfile profile, Player* player, void* sortFunc);
};


class ReinforcementLearningRegression : public RegressionAlg {
private:
	double value;
	//double pMatrix[][];


public:
	ReinforcementLearningRegression();
	PlayerState predict(InteractionsProfile profile, Player* player, void* sortFunc);
};