#pragma once
#include "Subspace.h"
#include "IntegerPartition.h"
#include "Edge.h"

class Node
{
public:
	Subspace subspace;
	IntegerPartition integerPartition;
	std::vector<Edge> edgesFromThisNode;
	bool tempFlag;
	std::vector<int> tempIntegerRoots;

	Node() {}

	Node(Subspace subspace);
};

