#include "Node.h"


Node::Node(Subspace subspace) {
	this->subspace = subspace;
	IntegerPartition newIntegerPartition = IntegerPartition(subspace.integers);
	integerPartition = newIntegerPartition;
	edgesFromThisNode = std::vector<Edge>();
}