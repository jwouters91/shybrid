#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHYBRID
Copyright (C) 2018  Jasper Wouters

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

    def forget_recording(self):
        """ Forget all recording references in this hybrid cluster
        """
        if self.root is not None:
            self.root.forget_recording()

        for ops in self.__ops:
            ops.spike_train.forget_recording()

    def add_recording(self, recording):
        """ Add the recording references to all spike trains in this cluster
        """
        if self.root is not None:
            self.root.add_recording(recording)

        for ops in self.__ops:
            ops.spike_train.add_recording(recording)

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
