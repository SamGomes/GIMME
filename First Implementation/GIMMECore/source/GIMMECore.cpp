#include "..//headers//GIMMECore.h"


#if defined(_MSC_VER)
	#ifndef GIMME_EXTERNAL_API
		#define GIMME_EXTERNAL_API __declspec(dllexport) (if using windows to export uncomment)
	#endif
#elif defined(__GNUC__)
	#ifndef GIMME_EXTERNAL_API
		#define GIMME_EXTERNAL_API __attribute__((visibility("default")))
	#endif
#endif

extern "C"
{
	RandomGen* randomGen;

	RegressionAlg* regAlg;
	ConfigsGenAlg* configGenAlg;
	FitnessAlg* fitnessAlg;

	std::vector<Player*>* players;
	Adaptation* adapt;

	const int MAX_NUM_GROUPS = 50;
	const int MAX_GROUPS_SIZE = 5;

struct ExportedAdaptationGroup {
public:
	InteractionsProfile interactionsProfile;

	int numPlayers;
	int playerIDs[MAX_GROUPS_SIZE];

	//AdaptationMechanic tailoredMechanic;
	PlayerCharacteristics avgPlayerCharacteristics;
};

struct ExportedAdaptationConfiguration {
public:
	int numGroups;
	ExportedAdaptationGroup groups[MAX_NUM_GROUPS];
};


	void GIMME_EXTERNAL_API addPlayer(int id, char* name, int numPastModelIncreasesCells, int maxAmountOfStoredProfilesPerCell, int numStoredPastIterations) {
		Player* player = new Player(id, name, numPastModelIncreasesCells, maxAmountOfStoredProfilesPerCell, numStoredPastIterations, randomGen);
		players->push_back(player);
	}
	void GIMME_EXTERNAL_API removePlayer(int id) {
		//players.erase(players.begin() + players);
	}

	PlayerCharacteristics GIMME_EXTERNAL_API getPlayerCharacteristics(int id) {
		std::string playerCharacteristicsStr = std::string();
		for (int i = 0; i < players->size(); i++) {
			Player* currPlayer = (*players)[i];
			if (currPlayer->getId() == id) {
				PlayerCharacteristics charact = currPlayer->getCurrState().characteristics;
				return charact;
			}
		}
		return PlayerCharacteristics();
	}

	void GIMME_EXTERNAL_API setPlayerCharacteristics(int id, PlayerCharacteristics characteristics) {
		std::string playerCharacteristicsStr = std::string();
		for (int i = 0; i < players->size(); i++) {
			Player* currPlayer = (*players)[i];
			if (currPlayer->getId() == id) {
				currPlayer->setCharacteristics(characteristics);
			}
		}
	}

	void deleteAdatationData() {
		if (randomGen != NULL) {
			delete randomGen;
		}
		
		if (regAlg != NULL) {
			delete regAlg;
		}
		if (configGenAlg != NULL) {
			delete configGenAlg;
		}
		if (fitnessAlg != NULL) {
			delete fitnessAlg;
		}

		if (adapt != NULL) {
			delete adapt;
		}

		if (players != NULL) {
			for (Player* player : *players) {
				delete player;
			}
			delete players;
		}

	}

	void GIMME_EXTERNAL_API initAdaptation() {
		deleteAdatationData();
		
		randomGen = new RandomGen();
		
		regAlg = new KNNRegression(5);
		configGenAlg = new RandomConfigsGen();
		fitnessAlg = new FundamentedFitness();

		players = new std::vector<Player*>();
		adapt = new Adaptation(
			"test",
			players,
			100,
			2, MAX_GROUPS_SIZE,
			regAlg,
			configGenAlg,
			fitnessAlg,
			randomGen,
			5);
	}

	ExportedAdaptationConfiguration GIMME_EXTERNAL_API iterate() {
		AdaptationConfiguration groupMechanicPairs = adapt->iterate();

		ExportedAdaptationConfiguration exportedConfig;
		exportedConfig.numGroups = (groupMechanicPairs.groups).size();
		//exportedConfig.groups = (ExportedAdaptationGroup*) malloc(sizeof(ExportedAdaptationGroup)*players->size());
		for (int i = 0; i < MAX_NUM_GROUPS; i++) {
			if (i >= (groupMechanicPairs.groups).size()) {
				exportedConfig.groups[i] = ExportedAdaptationGroup();
				continue;
			}
			AdaptationGroup currGroup = (groupMechanicPairs.groups)[i];
			int size = currGroup.players.size();

			ExportedAdaptationGroup exportedGroup;
			
			exportedGroup.interactionsProfile = currGroup.interactionsProfile;
			exportedGroup.avgPlayerCharacteristics = currGroup.avgPlayerState.characteristics;
			exportedGroup.numPlayers = size;
			for (int i = 0; i < MAX_GROUPS_SIZE; i++) {
				if (i >= size) {
					exportedGroup.playerIDs[i] = -1;
					continue;
				}
				Player currPlayer = *(currGroup.players[i]);
				exportedGroup.playerIDs[i] = currPlayer.getId();
			}

			exportedConfig.groups[i] = exportedGroup;
		}

		return exportedConfig;
	}

	void GIMME_EXTERNAL_API installCheck() {
		std::cout << "GIMME successfully added!" << std::endl;
	}
}