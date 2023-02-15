#include "IntegerPartitionGraph.h"
#include <iostream>

IntegerPartitionGraph::IntegerPartitionGraph(vector<vector<Subspace>> subspaces, int numPlayers, int largestIntegerBeingSplitInThisGraph) {
	this->largestIntegerBeingSplitInThisGraph = largestIntegerBeingSplitInThisGraph;

	nodes.resize(numPlayers);
	for (int level = 0; level < numPlayers; level++) {
		nodes[level] = vector<Node*>(subspaces[level].size());
		for (int i = 0; i < subspaces[level].size(); i++)
			nodes[level][i] = new Node(subspaces[level][i]);
	}


	for (int level = 0; level < numPlayers - 1; level++) {
		for (int i = 0; i < nodes[level].size(); i++) {
			vector<IntegerPartition> listOfDirectlyConnectedIntegerPartitions = nodes[level][i]->integerPartition.getListOfDirectedlyConnectedIntegerPartitions(largestIntegerBeingSplitInThisGraph, 0);
			if (listOfDirectlyConnectedIntegerPartitions.empty())
				nodes[level][i]->edgesFromThisNode = vector<Edge>();
			else {
				nodes[level][i]->edgesFromThisNode = vector<Edge>(listOfDirectlyConnectedIntegerPartitions.size());
				int index = 0;
				for (int j = 0; j < nodes[level + 1].size(); j++) {
					vector<int> integersThatResultedFromTheSplit = getIntegersThatResultedFromTheSplit(listOfDirectlyConnectedIntegerPartitions, nodes[level+1][j]);
					if (!integersThatResultedFromTheSplit.empty()) {
						vector<int> sortedParts1 = nodes[level][i]->integerPartition.partsSortedAscendingly;
						vector<int> sortedParts2 = nodes[level+1][i]->integerPartition.partsSortedAscendingly;
						int partThatWasSplit = -1;
						for (int k = (int)sortedParts1.size() - 1; k >= 0; k--)
							if (sortedParts1[k] != sortedParts2[k + 1]) {
								partThatWasSplit = sortedParts1[k];
								break;
							}
						nodes[level][i]->edgesFromThisNode[index] = Edge(nodes[level + 1][j], partThatWasSplit, integersThatResultedFromTheSplit);
						index++;
						if (index == nodes[level][i]->edgesFromThisNode.size())
							break;
					}
				}
			}
		}
	}

}

IntegerPartitionGraph::~IntegerPartitionGraph() 
{
	for (int i = 0; i < nodes.size(); i++) {
		int nodeSize = (int)nodes[i].size();
		for (int j = 0; j < nodeSize; j++) {
			if (i == 0 && j == 0)
				continue;
			delete nodes[i][j];
		}
	}
}


vector<Node*> IntegerPartitionGraph::getReachableNodes(Node* node)
{
	if (node->edgesFromThisNode.empty()) return vector<Node*>();

	int numOfIntegersInNode = (int)node->integerPartition.partsSortedAscendingly.size();

	node->tempIntegerRoots = vector<int>();
	for (int level = numOfIntegersInNode; level < nodes.size(); level++)
		for (int i = 0; i < nodes[level].size(); i++)
		{
			nodes[level][i]->tempFlag = false;
			nodes[level][i]->tempIntegerRoots = vector<int>();
		}
	for (int i = 0; i < node->edgesFromThisNode.size(); i++)
	{
		node->edgesFromThisNode[i].node->tempFlag = true;
		setIntegerRoots(node, node->edgesFromThisNode[i].node, node->edgesFromThisNode[i].twoPartsThatResultedFromTheSplit, node->edgesFromThisNode[i].partThatWasSplit);
	}

	int numOfReachableNodes = 0;
	for (int level = numOfIntegersInNode; level < nodes.size() - 1; level++)
	{
		for (int i = 0; i < nodes[level].size(); i++)
		{
			if (nodes[level][i]->tempFlag)
			{
				numOfReachableNodes++;
				if (!nodes[level][i]->edgesFromThisNode.empty())
					for (int j = 0; j < nodes[level][i]->edgesFromThisNode.size(); j++) {
						if (nodes[level][i]->edgesFromThisNode[j].node->tempFlag == false)
						{
							nodes[level][i]->edgesFromThisNode[j].node->tempFlag = true;
							setIntegerRoots(nodes[level][i], nodes[level][i]->edgesFromThisNode[j].node, nodes[level][i]->edgesFromThisNode[j].twoPartsThatResultedFromTheSplit, nodes[level][i]->edgesFromThisNode[j].partThatWasSplit);

						}
					}
			}
		}
	}

	int index = 0;
	vector<Node*> listOfReachableNodes(numOfReachableNodes);
	for (int level = numOfIntegersInNode; level < nodes.size() - 1; level++)
	{
		for (int i = 0; i < nodes[level].size(); i++) 
			if (nodes[level][i]->tempFlag)
			{
				listOfReachableNodes[index] = new Node;
				listOfReachableNodes[index] = nodes[level][i];

				index++;
			}
	}

	return listOfReachableNodes;
}

void IntegerPartitionGraph::updateEdges(int numPlayers, int largestIntegerBeingSplitInThisGraph)
{
	int prev_largestIntegerBeingSplitInThisGraph = this->largestIntegerBeingSplitInThisGraph;
	if (prev_largestIntegerBeingSplitInThisGraph >= largestIntegerBeingSplitInThisGraph)
		return;

	for (int level = 1; level < numPlayers - 1; level++)
	{
		for (int i = 0; i < nodes[level].size(); i++)
		{
			std::vector<IntegerPartition> listOfDirectlyConnectedIntegerPartitions = nodes[level][i]->integerPartition.getListOfDirectedlyConnectedIntegerPartitions(largestIntegerBeingSplitInThisGraph, prev_largestIntegerBeingSplitInThisGraph);

			if (!listOfDirectlyConnectedIntegerPartitions.empty())
			{

				int index;
				if (nodes[level][i]->edgesFromThisNode.empty())
				{

					index = 0; 
					nodes[level][i]->edgesFromThisNode.resize(listOfDirectlyConnectedIntegerPartitions.size());
				}
				else
				{

					index = (int)nodes[level][i]->edgesFromThisNode.size();
					std::vector<Edge> tempListOfEdges(nodes[level][i]->edgesFromThisNode.size() + listOfDirectlyConnectedIntegerPartitions.size());
					for (int j = 0; j < nodes[level][i]->edgesFromThisNode.size(); j++)
						tempListOfEdges[j] = nodes[level][i]->edgesFromThisNode[j];
					nodes[level][i]->edgesFromThisNode = tempListOfEdges;
					
				}

				for (int j = 0; j < nodes[level + 1].size(); j++)
				{
					std::vector<int> integersThatResultedFromTheSplit = getIntegersThatResultedFromTheSplit(listOfDirectlyConnectedIntegerPartitions, nodes[level + 1][j]);
					if (!integersThatResultedFromTheSplit.empty())
					{

						std::vector<int> sortedParts1 = nodes[level][i]->integerPartition.partsSortedAscendingly;
						std::vector<int> sortedParts2 = nodes[level+1][j]->integerPartition.partsSortedAscendingly;
						int partThatWasSplit = -1;
						for (int k = (int)sortedParts1.size() - 1; k >= 0; k--)
						{
							if (sortedParts1[k] != sortedParts2[k + 1])
							{
								partThatWasSplit = sortedParts1[k];
								break;
							}
						}
						nodes[level][i]->edgesFromThisNode[index] = Edge(nodes[level + 1][j], partThatWasSplit, integersThatResultedFromTheSplit);
						index++;
						if (index == nodes[level][i]->edgesFromThisNode.size())
							break;
					}
				}
			}
		}
	}
	this->largestIntegerBeingSplitInThisGraph = largestIntegerBeingSplitInThisGraph;
}




vector<int> IntegerPartitionGraph::getIntegersThatResultedFromTheSplit(vector<IntegerPartition> listOfDirectlyConnectedIntegerPartitions, Node* nodeOnHighLevel) {
	vector<int> multiplicity1 = nodeOnHighLevel->integerPartition.sortedMultiplicity;
	int underlyingSet1 = nodeOnHighLevel->integerPartition.sortedUnderlyingSetInBitFormat;

	for (int i = 0; i < listOfDirectlyConnectedIntegerPartitions.size(); i++) {
		vector<int> multiplicity2 = listOfDirectlyConnectedIntegerPartitions[i].sortedMultiplicity;
		int underlyingSet2 = listOfDirectlyConnectedIntegerPartitions[i].sortedUnderlyingSetInBitFormat;

		if (underlyingSet1 != underlyingSet2)
			continue;

		bool notEqual = false;
		for (int j = 0; j < multiplicity1.size(); j++)
			if (multiplicity1[j] != multiplicity2[j]) {
				notEqual = true;
				break;
			}

		if (notEqual)
			continue;
		return listOfDirectlyConnectedIntegerPartitions[i].tempIntegersThatResultedFromASplit;
	}
	return vector<int>();
}

void IntegerPartitionGraph::setIntegerRoots(Node* lowerNode, Node* upperNode, vector<int> twoPartsThatResultedFromTheSplit, int partThatWasSplit)
{
	vector<int> upperIntegers = upperNode->integerPartition.partsSortedAscendingly;
	vector<int> lowerIntegers = lowerNode->integerPartition.partsSortedAscendingly;

	upperNode->tempIntegerRoots = vector<int>(upperIntegers.size(), -1);

	if (lowerNode->tempIntegerRoots.empty())
	{
		for (int k = 0; k < twoPartsThatResultedFromTheSplit.size(); k++)
			for (int j = 0; j < upperIntegers.size(); j++)
				if ((twoPartsThatResultedFromTheSplit[k] == upperIntegers[j]) && (upperNode->tempIntegerRoots[j] == -1))
				{
					upperNode->tempIntegerRoots[j] = partThatWasSplit;
					break;
				}
		for (int j = 0; j < upperIntegers.size(); j++)
			if (upperNode->tempIntegerRoots[j] == -1)
				upperNode->tempIntegerRoots[j] = upperIntegers[j];

	}
	else
	{
		int newRoot = -10;
		int indexOfNewRoot = -10;

		for (int i = 0; i < lowerIntegers.size(); i++)
			if (lowerIntegers[i] == partThatWasSplit)
			{
				indexOfNewRoot = i;
				newRoot = lowerNode->tempIntegerRoots[i];
			}

		for (int i = 0; i < lowerNode->tempIntegerRoots.size(); i++)
		{
			if (i == indexOfNewRoot) continue;

			for (int j = 0; j < upperIntegers.size(); j++)
				if ((upperIntegers[j] == lowerIntegers[i]) && (upperNode->tempIntegerRoots[j] == -1))
				{
					upperNode->tempIntegerRoots[j] = lowerNode->tempIntegerRoots[i];
					break;
				}
		}
		
		for (int j = 0; j < upperIntegers.size(); j++)
			if (upperNode->tempIntegerRoots[j] == -1)
				upperNode->tempIntegerRoots[j] = newRoot;
	}
	
}
