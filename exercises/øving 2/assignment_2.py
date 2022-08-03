import numpy as np
import matplotlib as mpl


class HiddenMarkowModel:
    def __init__(self):
        self.transformation = [[0.8, 0.3], [0.7, 0.2]]
        # ^ ??? possibly wrong column/row combination
        self.observation_1 = [[0.75, 0], [0, 0.2]]
        self.observation_2 = [[0.25, 0], [0, 0.8]]
        self.pi = [0.5, 0.5]

    # help-method for normalizing vectors (i.e. the values in the vector summarized equals 1)
    def normalize(self, v):
        summ = 0
        for el in v:
            summ += el
        return v/summ

    def backward(self, observations):
        output = []
        evi = [1.0, 1.0]
        for birds_nearby in observations.reverse():
            evi = self.backward_by_one(birds_nearby, evi)
            output.insert(0, evi)
        return output

    def backward_by_one(self, birds_nearby, e_prob):
        if birds_nearby:
            one_step_backward = np.matmul(
                self.transformation, self.observation_1)
        else:
            one_step_backward = np.matmul(
                self.transformation, self.observation_2)
        normalized_vec = self.normalize(np.matmul(one_step_backward, e_prob))
        return normalized_vec

    # method for filtering and prediciting, using the help method and taking into account all provided evidence
    def forward(self, observations):
        output = []
        birds_nearby = self.pi
        for obs in observations:
            birds_nearby = self.forward_by_one(obs, birds_nearby)
            output.append(birds_nearby)
        return output

    # help-method for one step in the forward-method
    def forward_by_one(self, birds_nearby, fish_prob):
        one_step_forward = np.matmul(self.transformation, fish_prob)
        if birds_nearby:
            return self.normalize(np.matmul(self.observation_1, one_step_forward))
        return self.normalize(np.matmul(self.observation_2, one_step_forward))

    def for_back(self, observations):
        forward_vals = self.forward(observations)
        backward_vals = self.backward(observations)
        backward_vals.append([1.0, 1.0])
        smooth = []
        for i, obs in enumerate(observations):
            smooth.append(self.normalize(
                forward_vals[i]*backward_vals[i+1]))
        return smooth


if __name__ == '__main__':
    observations = [True, True, False, True, False, True]
    # task 1 B
    hhm = HiddenMarkowModel()
    print(observations)
    for t in range(len(observations)):
        obs_t = observations[0:t]
        print(obs_t)
        print(hhm.forward(obs_t))
