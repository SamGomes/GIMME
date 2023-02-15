#pragma once
#include <vector>

class Node;

class Edge
{
public:
	Node* node;
	int partThatWasSplit;
	std::vector<int> twoPartsThatResultedFromTheSplit;

	Edge(){}
	Edge(Node* node, int partThatWasSplit, std::vector<int> twoPartsThatResultedFromTheSplit);
};

