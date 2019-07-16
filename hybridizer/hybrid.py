#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:51:16 2019

@author: jwouters
"""

from collections import deque

class HybridCluster:
    """ A class modeling a hybrid cluster and the operations performed on it
    """
    def __init__(self, idx, spike_train):
        self.idx = idx
        self.root = spike_train
        self.__ops = deque()

    def is_hybrid(self):
        """ Check whether the cluster is a hybrid cluster
        """
        # The cluster is only a hybrid cluster if the root has been relocated
        # in the binary data.
        if len(self.__ops) == 0:
            return False
        else:
            return True

    def get_actual_spike_train(self):
        """ Return the spike times of the hybrid cluster
        """
        if self.is_hybrid():
            return self.__ops[-1].spike_train
        else:
            return self.root

    def apply_operator(self, op):
        """ Apply a given operator to hybridize this cluster
        """
        # execute operator
        op.execute()
        # keep track of operator
        self.__ops.append(op)

    def undo_last_operator(self):
        """ Undo the last applied operator for this cluster
        """
        self.__ops.pop().undo()


""" Hybridization operators
"""
class HybridizationOperator:
    """ Abstract hybridization operator
    """
    def __init__(self, spike_train):
        self.spike_train = spike_train

class Insert(HybridizationOperator):
    """ Operator that defines a spike train addition
    """
    def __init__(self, spike_train):
        HybridizationOperator.__init__(self, spike_train)

    def execute(self):
        """ apply operation to data
        """
        self.spike_train.insert()

    def undo(self):
        """ undo this operator
        """
        Subtract(self.spike_train).execute()

class Subtract(HybridizationOperator):
    """ Operator that defines a spike train subtraction
    """
    def __init__(self, spike_train):
        HybridizationOperator.__init__(self, spike_train)

    def execute(self):
        """ apply operation to data
        """
        self.spike_train.subtract()

    def undo(self):
        """ Undo this operator
        """
        Insert(self.spike_train).execute()
