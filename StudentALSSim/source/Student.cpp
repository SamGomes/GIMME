#include "../headers/Student.h"


Student::StudentModelGrid::StudentModelGrid(){}
Student::StudentModelGrid::StudentModelGrid(int numCells, int maxAmountOfStoredProfilesPerCell)
{
	this->numCells = numCells;
	this->maxAmountOfStoredProfilesPerCell = maxAmountOfStoredProfilesPerCell;

	cells = std::vector<std::vector<StudentModel>>(numCells);
}
void Student::StudentModelGrid::pushToGrid(Student::StudentModel model) {
	double dimSpan = cbrt((double)numCells-1);
	int currCellInd = dimSpan * dimSpan * floor(dimSpan * model.currProfile.K_cl) + dimSpan * floor(dimSpan * model.currProfile.K_cp) + floor(dimSpan* model.currProfile.K_i);
	/*printf("ind: %d\n", currCellInd);
	printf("profi: (%f,%f,%f)\n", model.currProfile.K_cl, model.currProfile.K_cp, model.currProfile.K_i);*/
	std::vector<StudentModel>* currCell = &(cells[currCellInd]);
	currCell->push_back(model);
	int cellsSize = cells[currCellInd].size();
	if (cellsSize > maxAmountOfStoredProfilesPerCell) {
		currCell->erase(currCell->begin());
	}
}
std::vector<Student::StudentModel> Student::StudentModelGrid::getAllModels() {
	std::vector<StudentModel> allCells = std::vector<StudentModel>();
	for (int i = 0; i < cells.size(); i++) {
		std::vector<StudentModel>* currCell = &cells[i];
		allCells.insert(allCells.end(), currCell->begin(), currCell->end());
	}
	return allCells;
}


Student::Student(int id, std::string name, int numPastModelIncreasesCells, int maxAmountOfStoredProfilesPerCell){
	
	//generate learning profile
	double newRand1 = Utilities::randBetween(0, 1);
	double newRand2 = Utilities::randBetween(0, 1);
	double newRand3 = Utilities::randBetween(0, 1);

	double newRandSum = newRand1 + newRand2 + newRand3;

	this->inherentPreference.K_cl = newRand1 / newRandSum;
	this->inherentPreference.K_cp = newRand2 / newRandSum;
	this->inherentPreference.K_i = newRand3 / newRandSum;

	this->learningRateSeed = Utilities::randBetween(0, 1000);
	//this->learningRate = Utilities::randBetween(0, 1);

	this->currModel.currProfile = { 0,0,0 };
	this->currModel.engagement = 0;
	this->currModel.ability = 0;

	this->id = id;
	this->name = name;
	this->pastModelIncreasesGrid = StudentModelGrid(numPastModelIncreasesCells, maxAmountOfStoredProfilesPerCell);


	this->numPastModelIncreasesCells = numPastModelIncreasesCells;
	this->maxAmountOfStoredProfilesPerCell = maxAmountOfStoredProfilesPerCell;

}

void Student::reset(int numberOfStudentModelCells, int maxAmountOfStoredProfilesPerCell) {
	
	this->currModel.currProfile = { 0,0,0 };

	this->currModel.engagement = 0;
	this->currModel.ability = 0;

	this->pastModelIncreasesGrid = StudentModelGrid(numberOfStudentModelCells, maxAmountOfStoredProfilesPerCell);
}

void Student::setEngagement(double engagement) {
	this->currModel.engagement = engagement;
}
double Student::getEngagement() {
	return this->currModel.engagement;
}

std::vector<Student::StudentModel> Student::getPastModelIncreases() {
	return this->pastModelIncreasesGrid.getAllModels();
}


void Student::setAbility(double ability) {
	this->currModel.ability = ability;
}
double Student::getAbility() {
	return this->currModel.ability;
}


void Student::changeCurrProfile(Utilities::LearningProfile newProfile) {
	Student::StudentModel currModel = this->currModel;
	this->currModel.currProfile = newProfile;
}
Utilities::LearningProfile Student::getCurrProfile() {
	return this->currModel.currProfile;
}
Utilities::LearningProfile Student::getInherentPreference() {
	return this->inherentPreference;
}
double Student::getLearningRate() {
	return this->learningRate;
}

void Student::simulateReaction()
{
	StudentModel increases = StudentModel(currModel);
	this->calcReaction(&currModel.engagement, &currModel.ability, &currModel.currProfile);
	
	increases.ability = currModel.ability - increases.ability;
	increases.engagement = currModel.engagement; // -increases.engagement;
	
	this->pastModelIncreasesGrid.pushToGrid(increases);
}

void Student::calcReaction(double* engagement, double* ability, Utilities::LearningProfile* profile)
{
	Utilities::LearningProfile currProfile = this->currModel.currProfile;

	*engagement = 1.0 - inherentPreference.distanceBetween(*profile);

	this->learningRate = Utilities::normalRandom(0.5, 0.1);
	double abilityIncreaseSim = (learningRate * *engagement); //between 0 and 1
	*ability += abilityIncreaseSim;

}
