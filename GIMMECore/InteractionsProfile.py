import copy
import random
import traceback


class InteractionsProfile(object):
    # public members
    dimensions = None

    # private members
    __dimensionality = None

    def __init__(self, dimensions=None):
        self.dimensions = {} if dimensions is None else dimensions
        self.__dimensionality = len(self.dimensions)

    def reset(self):
        for key in self.dimensions:
            self.dimensions[key] = 0
        return self

    def generate_copy(self):
        keys = list(self.dimensions.keys())
        newVar = type(self)(copy.copy(self.dimensions))
        for key in keys:
            newVar.dimensions[key] = self.dimensions[key]
        return newVar

    def normalize(self):
        return self.__normalization(self)

    def normalized(self):
        clone = self.generate_copy()
        return self.__normalization(clone)

    def __normalization(self, profile):
        if len(profile.dimensions) > 1:
            total = 0
            for key in profile.dimensions:
                total += profile.dimensions[key]
            if total == 0:
                for key in profile.dimensions:
                    profile.dimensions[key] = 1 / len(profile.dimensions)
            else:
                for key in profile.dimensions:
                    profile.dimensions[key] = profile.dimensions[key] / total
        return profile

    def randomize(self):
        return self.__randomization(self)

    def randomized(self):
        clone = self.generate_copy()
        return self.__randomization(clone)

    def __randomization(self, profile):
        profile.reset()
        for key in profile.dimensions:
            profile.dimensions[key] = random.uniform(0.0, 1.0)
        return profile

    def sqr_distance_between(self, profile_to_test):
        cost = self.generate_copy()
        cost.reset()
        if len(cost.dimensions) != len(profile_to_test.dimensions):
            traceback.print_stack()
            print(cost.dimensions)
            print(profile_to_test.dimensions)
            raise Exception(
                "[ERROR] Could not compute distance between profiles in different sized spaces. Execution aborted.")

        for key in cost.dimensions:
            cost.dimensions[key] = abs(self.dimensions[key] - profile_to_test.dimensions[key])

        total = 0
        for key in cost.dimensions:
            cost.dimensions[key] = pow(cost.dimensions[key], 2)
            total += cost.dimensions[key]

        return total

    def distance_between(self, profile_to_test):
        num_dims = len(profile_to_test.dimensions)
        return self.sqr_distance_between(profile_to_test) ** (1 / float(num_dims))

    def flattened(self):
        return [dim for dim in self.dimensions.values()]

    def __unflatten_func(self, profile, array):
        i = 0
        for key in profile.dimensions.keys():
            profile.dimensions[key] = array[i]
            i += 1
        return profile

    def unflatten(self, array):
        return self.__unflatten_func(self, array)

    def unflattened(self, array):
        clone = self.generate_copy()
        return self.__unflatten_func(clone, array)
