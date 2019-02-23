#pragma once
#include <algorithm> 

#include "Utilities.h"
#include <iostream>

class Student {
public:
	struct LearningState {
		Utilities::LearningProfile currProfile;
		double engagement; // on vs off task percentage
		double ability; // score percentage

		double dist;
	};
	
	class StudentModelGrid {
		private:
			std::vector<std::vector<LearningState>> cells;
			int numCells;
			int maxAmountOfStoredProfilesPerCell;

		public:
			StudentModelGrid();
			StudentModelGrid(int numCells, int maxAmountOfStoredProfilesPerCell);
			void pushToGrid(LearningState model);
			std::vector<LearningState> getAllModels();
	};


private:
	int id;
	std::string name;

	//for simulation
	Utilities::LearningProfile inherentPreference;
	double learningRate;
	std::vector<double> iterationReactions;

	//Adaptation part
	int numPastModelIncreasesCells;
	int maxAmountOfStoredProfilesPerCell;
	StudentModelGrid pastModelIncreasesGrid;

	LearningState currModel;



public:

	Student(int id, std::string name, int numPastModelIncreasesCells, int maxAmountOfStoredProfilesPerCell, int numIterations);
	void reset(int numberOfStudentModelCells, int maxAmountOfStoredProfilesPerCell);

	void setEngagement(double preference);
	double getEngagement();

	void setAbility(double preference);
	double getAbility();

	std::vector<LearningState> getPastModelIncreases();
	void changeCurrProfile(Utilities::LearningProfile newProfile);

	int getId();
	std::string getName();

	Utilities::LearningProfile getCurrProfile();
	Utilities::LearningProfile getInherentPreference();
	double getLearningRate();

	void simulateReaction(int currIteration);
	void calcReaction(double* engagement, double* ability, Utilities::LearningProfile* profile, int currIteration);



	LearningState currModelIncreases; //for displaying in the chart
};