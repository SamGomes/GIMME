
// GIMMEConsoleUI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "../GIMMECore/headers/GIMMECore.h"

int main()
{



	std::vector<AdaptationTask> possibleCollaborativeTasks = std::vector<AdaptationTask>();
	std::vector<AdaptationTask> possibleCompetitiveTasks = std::vector<AdaptationTask>();
	std::vector<AdaptationTask> possibleIndividualTasks = std::vector<AdaptationTask>();

	possibleCollaborativeTasks.clear();
	std::vector<AdaptationTask> taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1Inst5", 0.9f));
	AdaptationTask task1 = AdaptationTask(AdaptationTaskType::COLLABORATION, "collab1", taskInstances);
	possibleCollaborativeTasks.push_back(task1);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2Inst5", 0.9f));
	AdaptationTask task2 = AdaptationTask(AdaptationTaskType::COLLABORATION, "collab2", taskInstances);
	possibleCollaborativeTasks.push_back(task2);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3Inst5", 0.9f));
	AdaptationTask task3 = AdaptationTask(AdaptationTaskType::COLLABORATION, "collab3", taskInstances);
	possibleCollaborativeTasks.push_back(task3);

	possibleCompetitiveTasks.clear();
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp1Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp1Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp1Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp1Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp1Inst5", 0.9f));
	task1 = AdaptationTask(AdaptationTaskType::COMPETITION, "comp1", taskInstances);
	possibleCompetitiveTasks.push_back(task1);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp2Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp2Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp2Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp2Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp2Inst5", 0.9f));
	task2 = AdaptationTask(AdaptationTaskType::COMPETITION, "comp2", taskInstances);
	possibleCompetitiveTasks.push_back(task2);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp3Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp3Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp3Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp3Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::COMPETITION, "comp3Inst5", 0.9f));
	task3 = AdaptationTask(AdaptationTaskType::COMPETITION, "comp3", taskInstances);
	possibleCompetitiveTasks.push_back(task3);

	possibleIndividualTasks.clear();
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1Inst5", 0.9f));
	task1 = AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self1", taskInstances);
	possibleIndividualTasks.push_back(task1);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2Inst5", 0.9f));
	task2 = AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self2", taskInstances);
	possibleIndividualTasks.push_back(task2);
	taskInstances = std::vector<AdaptationTask>();
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3Inst1", 0.1f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3Inst2", 0.3f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3Inst3", 0.5f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3Inst4", 0.7f));
	taskInstances.push_back(AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3Inst5", 0.9f));
	task3 = AdaptationTask(AdaptationTaskType::SELF_INTERACTION, "self3", taskInstances);
	possibleIndividualTasks.push_back(task3);
	
	
	int numStudentsInClass = 23;

	Utilities* utilities = new Utilities();
	utilities->resetRandoms();
	
	//generate all of the students models
	std::vector<Student*> students = std::vector<Student*>();
	for (int i = 0; i < numStudentsInClass; i++) {
		students.push_back(new Student(i, "a", 1, 1, 1, utilities));
	}

	Adaptation adapt = Adaptation(
		students,
		100,
		2, 5,
		10,
		utilities, 5,
		possibleCollaborativeTasks,
		possibleCompetitiveTasks,
		possibleIndividualTasks);


	std::vector<std::pair<AdaptationGroup, std::vector<AdaptationTask>>> groupMechanicPairs = adapt.iterate();
	groupMechanicPairs = adapt.iterate();
	groupMechanicPairs = adapt.iterate();
	groupMechanicPairs = adapt.iterate();
	groupMechanicPairs = adapt.iterate();

	int mechanicsSize = (int)groupMechanicPairs.size();

	//intervene
	for (int j = 0; j < mechanicsSize; j++) {
		std::vector<Student*> currGroup = groupMechanicPairs[j].first.getStudents();
		std::vector<AdaptationTask> currMechanic = groupMechanicPairs[j].second;

		std::cout << "promote on students:" << std::endl;
		for (int k = 0; k < currGroup.size(); k++) {
			std::cout << "Number: " << currGroup[k]->getId() << ", Name: " << currGroup[k]->getName() << std::endl;
		}
		std::cout << ", the following tasks:" << std::endl;
		for (int k = 0; k < currMechanic.size(); k++) {
			std::cout << currMechanic[k].description << std::endl;
		}
		std::cout << "-- -- -- -- -- -- -- -- -- -- -- -- --" << std::endl;
	}
	std::cout << "----------------------End of Iteration--------------------" << std::endl;

	getchar();
}