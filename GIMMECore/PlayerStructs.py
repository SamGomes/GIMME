import time
from .InteractionsProfile import InteractionsProfile

from abc import ABC, abstractmethod


class PlayerCharacteristics(object):
    # public members
    ability = None
    engagement = None
    group_diversity = None

    def __init__(self, ability=None, engagement=None, group_diversity=None):
        self.ability = 0 if ability is None else ability
        self.engagement = 0 if engagement is None else engagement
        self.group_diversity = 0 if group_diversity is None else group_diversity

    def reset(self):
        self.ability = 0
        self.engagement = 0
        return self


class PlayerState(object):
    # public members
    creation_time = None

    type = None
    profile = None
    characteristics = None
    dist = None
    quality = None

    group = None
    tasks = None

    def __init__(self, type=None, profile=None, characteristics=None, dist=None, quality=None, group=None,
                 tasks=None):
        self.creation_time = time.time()

        self.type = 1 if type is None else type
        self.profile = InteractionsProfile() if profile is None else profile
        self.characteristics = PlayerCharacteristics() if characteristics is None else characteristics
        self.dist = -1 if dist is None else dist
        self.quality = -1 if quality is None else quality

        self.group = [] if group is None else group
        self.tasks = [] if tasks is None else tasks

    def reset(self):
        self.characteristics.reset()
        self.profile.reset()
        self.creation_time = time.time()
        self.type = 1
        self.quality = -1
        self.dist = -1

        self.group = []
        self.tasks = []
        return self


class PlayerPersonality(ABC):
    # protected members
    _max_difference_value = None

    def __init__(self):
        self._max_difference_value = 1

    @abstractmethod
    def get_personality_string(self):
        pass

    @abstractmethod
    def get_pair_personality_diversity(self, other):
        pass

    @abstractmethod
    def get_team_personality_diversity(self, players):
        pass


class PersonalityMBTI(PlayerPersonality):
    # private members
    __letter1 = None
    __letter2 = None
    __letter3 = None
    __letter4 = None

    def __init__(self):
        super()
        self.__letter1 = None
        self.__letter2 = None
        self.__letter3 = None
        self.__letter4 = None

    def __init__(self, letter1, letter2, letter3, letter4):
        super()
        self.__letter1 = letter1.upper()
        self.__letter2 = letter2.upper()
        self.__letter3 = letter3.upper()
        self.__letter4 = letter4.upper()

    def get_personality_string(self):
        return self.__letter1 + self.__letter2 + self.__letter3 + self.__letter4

    def get_letters_list(self):
        return [self.__letter1, self.__letter2, self.__letter3, self.__letter4]

    # Determine the difference between 2 personalities. The value ranges from 0 (no difference) to 1 (max difference).
    def get_pair_personality_diversity(self, other):
        if not isinstance(other, PersonalityMBTI):
            raise Exception("[ERROR] Comparison between different personality models not allowed.")

        difference = 0
        other_letters = other.get_letters_list()
        self_letters = self.get_letters_list()

        for i in range(0, len(self_letters)):
            difference += 0 if self_letters[i] == other_letters[i] else self._max_difference_value / 4

        return difference

    # Determine the group personality difference. Using a formula proposed by Pieterse, Kourie and Sonnekus.
    # players is a list of PlayerPersonalties
    def get_team_personality_diversity(self, players):
        # Difference is 0 if there's only one player
        if len(players) == 1:
            return 0

        letters_list = [[], [], [], []]

        # Populate letters_list with all the players' letters
        for player in players:
            for letters, letter in zip(letters_list, player.get_letters_list()):
                letters.append(letter)

        difference = 0.0

        for letters in letters_list:
            letters.sort()
            length = len(letters)

            if (letters[0] != letters[1]) or (letters[length - 2] != letters[length - 1]):
                # The first/last two letters are different -> means all but one letters are the same (difference = 1)
                difference += 1.0
                continue
            elif length <= 3:
                # The first/last two letters are the same and len =< 3, means all the letters are the same (difference = 0)
                continue

            for letter in letters:
                # If not all the letters are the same, then difference = 2
                if letter != letters[1]:
                    difference += 2.0
                    break

        # Otherwise, all the letters are the same (difference = 0)

        # Max value for difference is 8. Divide by 8 in order to normalize it.
        return difference / 8.0


class PlayerStatesDataFrame(object):
    # public members
    interactions_profile_template = None
    trim_alg = None
    states = None
    flat_profiles = None
    flat_abilities = None
    flat_engagements = None

    def __init__(self, interactions_profile_template, trim_alg, states=None):
        self.interactions_profile_template = interactions_profile_template
        self.trim_alg = trim_alg

        self.states = [] if states is None else states

        # auxiliary stuff
        self.flat_profiles = []
        self.flat_abilities = []
        self.flat_engagements = []
        if states is not None:
            for state in self.states:
                self.flat_profiles.append(state.profile.flattened())
                self.flat_abilities.append(state.characteristics.ability)
                self.flat_engagements.append(state.characteristics.engagement)

    def reset(self):
        self.states = []

        # auxiliary stuff
        self.flat_profiles = []
        self.flat_abilities = []
        self.flat_engagements = []

    def push_to_data_frame(self, player_state):
        self.states.append(player_state)

        # update tuple representation
        self.flat_profiles.append(player_state.profile.flattened())
        self.flat_abilities.append(player_state.characteristics.ability)
        self.flat_engagements.append(player_state.characteristics.engagement)

        trimmed_list_and_remainder = self.trim_alg.trimmed_list(self.states)
        trimmed_list = trimmed_list_and_remainder[0]
        remainder_indexes = trimmed_list_and_remainder[1]

        self.states = trimmed_list

        # update tuple representation
        for i in remainder_indexes:
            self.flat_profiles.pop(i)
            self.flat_abilities.pop(i)
            self.flat_engagements.pop(i)

    def get_all_states(self):
        return self.states

    def get_all_states_flatten(self):
        return {'profiles': self.flat_profiles, 'abilities': self.flat_abilities, 'engagements': self.flat_engagements}

    def get_num_states(self):
        return len(self.states)
