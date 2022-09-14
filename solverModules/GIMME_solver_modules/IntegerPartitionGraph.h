#pragma once
#include <vector>
#include "Node.h"


using namespace std;



class IntegerPartitionGraph
{
public:
	vector<vector<Node*>> nodes;
	int largestIntegerBeingSplitInThisGraph;

	IntegerPartitionGraph() {}
	~IntegerPartitionGraph();
	IntegerPartitionGraph(vector<vector<Subspace>> subspace, int numPlayers, int largestIntegerBeingSplitInThisGraph);

	vector<Node*> getReachableNodes(Node* node);
	void updateEdges(int numPlayers, int largestIntegerBeingSplitInThisGraph);


private:
	vector<int> getIntegersThatResultedFromTheSplit(vector<IntegerPartition> listOfDirectlyConnectedIntegerPartitions, Node* nodeOnHighLevel);
	void setIntegerRoots(Node* lowerNode, Node* upperNode, vector<int> twoPartsThatResultedFromTheSplit, int partThatWasSplit);

};

