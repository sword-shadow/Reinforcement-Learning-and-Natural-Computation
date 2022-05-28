#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from scipy.optimize import linprog

class MinimaxQPlayer:
    
    def __init__(self, numStates, numActionsA, numActionsB, decay, expl, gamma):
        # goal bonus decays with steps
        self.decay = decay
        # epsilon-greedy action policy during learning
        self.expl = expl
        self.learning = True
        # the discount constant
        self.gamma = gamma
        # the update percentage
        self.alpha = 1
        # the value function
        self.V = np.ones(numStates)
        # the Q function
        self.Q = np.ones((numStates, numActionsA, numActionsB))
        # the action policy
        self.pi = np.ones((numStates, numActionsA)) / numActionsA
        # the numbers of states and actions
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB

    def chooseAction(self, state, restrict=None):
        # epsilon-greedy 
        if self.learning and np.random.rand() < self.expl:
            action = np.random.randint(self.numActionsA)
        # choose action with policy probability
        else:
            action = self.weightedActionChoice(state)
        return action

    def weightedActionChoice(self, state):
        rand = np.random.rand()
        cumSumProb = np.cumsum(self.pi[state])
        action = 0
        while rand > cumSumProb[action]:
            action += 1
        return action
    
    def getReward(self, initialState, finalState, actions, reward, restrictActions=None):
        if not self.learning:
            return
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * self.Q[initialState, actionA, actionB] +             self.alpha * (reward + self.gamma * self.V[finalState])
        self.V[initialState] = self.updatePolicy(initialState)  # EQUIVALENT TO : min(np.sum(self.Q[initialState].T * self.pi[initialState], axis=1))
        self.alpha *= self.decay

    def updatePolicy(self, state, retry=False):
        c = np.zeros(self.numActionsA + 1)
        c[0] = -1
        A_ub = np.ones((self.numActionsB, self.numActionsA + 1))
        A_ub[:, 1:] = -self.Q[state].T
        b_ub = np.zeros(self.numActionsB)
        A_eq = np.ones((1, self.numActionsA + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.numActionsA
        
        # to solve min c^T x, with A_ub x <= b_ub, A_eq x = b_eq and the (min, max) bound for each param in c
        '''
         this function is to min the x[0] which is positive, where:
             x[0] is the expected min sum
             x means the action probability
             ub means the sum of Q.T * pi >= x[0], for each action state of the opponent
             eq means the sum of action probability equals to one
        '''
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if res.success:
            self.pi[state] = res.x[1:]
        elif not retry:
            return self.updatePolicy(state, retry=True)
        else:
            print("Alert : %s" % res.message)
            return self.V[state]

        return res.x[0]

    def policyForState(self, state):
        for i in range(self.numActionsA):
            print("Actions %d : %f" % (i, self.pi[state, i]))

class RandomPlayer:

    def __init__(self, numActions, p=None):
        self.numActions = numActions
        self.p = p

    def chooseAction(self, state, maxAction=None):
        maxAction = self.numActions if maxAction is None else maxAction
        if self.p == None:
            return np.random.randint(maxAction)
        else:
            return np.random.choice(maxAction, p=self.p)

    def getReward(self, initialState, finalState, actions, reward, maxActions=None):
        pass