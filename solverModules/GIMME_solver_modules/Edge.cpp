#include "Edge.h"
#include "Node.h"

/*
   an obsject of an edge in the integer Partition graph
 */


Edge::Edge(Node* node, int partThatWasSplit, std::vector<int> twoPartsThatResultedFromTheSplit) {
	this->node = node;
	this->partThatWasSplit = partThatWasSplit;
	this->twoPartsThatResultedFromTheSplit = twoPartsThatResultedFromTheSplit;
}