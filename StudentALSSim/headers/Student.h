#pragma once
#include <algorithm> 

#include "Utilities.h"
#include <iostream>

class Student {
public:
	struct StudentModel {
		Utilities::LearningProfile currProfile;
		double engagement; // on vs off task percentage
		double ability; // score percentage

		double dist;
	};


private:
	int id;
	std::string name;

	//for simulation
	Utilities::LearningProfile inherentPreference;
	double learningRate;

	//Adaptation part
	std::vector<StudentModel> pastModels;
	int maxAmountOfStoredProfiles;

	StudentModel myModel;
	

public:

	Student(int id, std::string name, int maxAmountOfStoredProfiles);
	void reset();

	void setEngagement(double preference);
	double getEngagement();

	void setAbility(double preference);
	double getAbility();

	std::vector<StudentModel> getPastModels();
	void changeCurrProfile(Utilities::LearningProfile currProfile);


	Utilities::LearningProfile getCurrProfile();
	Utilities::LearningProfile getInherentPreference();
	double getLearningRate();

	void simulateReaction(int numberOfAdaptationCycles);
};