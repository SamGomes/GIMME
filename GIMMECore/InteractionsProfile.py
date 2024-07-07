import copy
import random
import traceback

class InteractionsProfile(object):
    def __init__(self, dimensions=None):
        self.dimensions = {} if dimensions is None else dimensions
        self.dimensionality = len(self.dimensions)

    def reset(self):
        for key in self.dimensions:
            self.dimensions[key] = 0
        return self

    def init(self):
        return self.reset()

    def generate_copy(self):
        keys = list(self.dimensions.keys())
        newVar = type(self)(copy.copy(self.dimensions))
        for key in keys:
            newVar.dimensions[key] = self.dimensions[key]
        return newVar

    def normalize(self):
        return self.normalization(self)

    def normalized(self):
        clone = self.generate_copy()
        return self.normalization(clone)

    def normalization(self, profile):
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
        return self.randomization(self)

    def randomized(self):
        clone = self.generate_copy()
        return self.randomization(clone)

    def randomization(self, profile):
        profile.reset()
        for key in profile.dimensions:
            profile.dimensions[key] = random.uniform(0.0, 1.0)
        return profile

    def sqr_distance_between(self, profileToTest):
        cost = self.generate_copy()
        cost.reset()
        if len(cost.dimensions) != len(profileToTest.dimensions):
            traceback.print_stack()
            print(cost.dimensions)
            print(profileToTest.dimensions)
            raise Exception(
                "[ERROR] Could not compute distance between profiles in different sized spaces. Execution aborted.")

        for key in cost.dimensions:
            cost.dimensions[key] = abs(self.dimensions[key] - profileToTest.dimensions[key])

        total = 0
        for key in cost.dimensions:
            cost.dimensions[key] = pow(cost.dimensions[key], 2)
            total += cost.dimensions[key]

        return total

    def distance_between(self, profileToTest):
        numDims = len(profileToTest.dimensions)
        return self.sqr_distance_between(profileToTest) ** (1 / float(numDims))

    def flattened(self):
        return [dim for dim in self.dimensions.values()]

    def unflatten_func(self, profile, array):
        i = 0
        for key in profile.dimensions.keys():
            profile.dimensions[key] = array[i]
            i += 1
        return profile

    def unflatten(self, array):
        return self.unflatten_func(self, array)

    def unflattened(self, array):
        clone = self.generate_copy()
        return self.unflatten_func(clone, array)
