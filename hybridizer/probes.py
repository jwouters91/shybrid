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

import numpy as np

# TODO merge probe models
class Probe:
    """ Class exposing the probe file. Supporting only single-shank 
    with shank id **1**.

    TODO: automatically determine shank ID and raise error if not single-shank

    Parameters
    ----------
        probe_fn (string): full path and filename to the probe file
    """

    def __init__(self, probe_fn):
        variables = {}
        # execute the probe file
        exec(open(probe_fn).read(), variables)

        channel_groups = list(variables['channel_groups'].keys())

        if len(channel_groups) != 1:
            raise ValueError("Probe file can only have one channel group")
        channel_group = channel_groups[0]

        # extract channels from probe
        self.channels = variables['channel_groups'][channel_group]['channels']
        self.channels = np.array(self.channels)

        # extract total number of channels
        self.total_nb_channels = variables['total_nb_channels']

        # extract geometry
        self.geometry = variables['channel_groups'][channel_group]['geometry']

        # assuming rectangular probes with equal x spacing and equal y spacing
        self.x_between = self.get_x_between()
        self.y_between = self.get_y_between()

    def get_min_geometry(self):
        """ Return the minimum geometry value for each dimension in a single
        ndarray
        """
        return np.array(list(self.geometry.values())).min(axis=0)

    def get_max_geometry(self):
        """ Return the minimum geometry value for each dimension in a single
        ndarray
        """
        return np.array(list(self.geometry.values())).max(axis=0)

    def get_x_between(self):
        """ Return the electrode pitch in the x direction
        """
        X = 0

        x_locs = np.array(list(self.geometry.values()))[:,X]

        # TODO refactor
        self.x_max = x_locs.max()
        self.x_min = x_locs.min()

        # init at the maximum possible difference (possibly zero)
        x_between = self.get_max_geometry()[X] - self.get_min_geometry()[X]

        # choose x between as the shortest non-zero difference
        for x_tmp_1 in x_locs:
            for x_tmp_2 in x_locs:
                x_diff = abs(x_tmp_1 - x_tmp_2)
                if x_diff > 0 and x_diff < x_between:
                    x_between = x_diff

        if x_between == 0:
            x_between = 1

        return x_between

    def get_y_between(self):
        """ Return the electrode pitch in the y direction
        """
        Y = 1

        y_locs = np.array(list(self.geometry.values()))[:,Y]

        # TODO refactor
        self.y_max = y_locs.max()
        self.y_min = y_locs.min()

        # init at the maximum possible difference (possibly zero)
        y_between = self.get_max_geometry()[Y] - self.get_min_geometry()[Y]

        # choose x between as the shortest non-zero difference
        for y_tmp_1 in y_locs:
            for y_tmp_2 in y_locs:
                y_diff = abs(y_tmp_1 - y_tmp_2)
                if y_diff > 0 and y_diff < y_between:
                    y_between = y_diff

        if y_between == 0:
            y_between = 1

        return y_between

    def chans_to_good_chans(self, chans):
        """ Convert given channels to good channels, preserves the order

        Parameters
        ----------
        chans (array_like) : array containing channels in the original domain

        Returns
        -------
        good_chans (ndarray) : array containing the given channels in the good
        channels domain
        """
        good_chans = []
        for chan in chans:
            good_chans += [np.where(self.channels == chan)[0][0]]

        return np.array(good_chans)

class RectangularProbe:
    """ A model for probes that have a rectangular electrode grid. Automatic
    probe graph generation supported.
    """

    def __init__(self):
        self.channels = np.array([], dtype=Channel)
        self.origin = None

    def fill_channels(self, channel_geometries):
        """ Fill the channels attribute with channel objects generated from
        the given channels geometries dictionary

        Parameters
        ----------
            channel_geometries (dict) : dictionary containing the coordinates
            for every channel
        """
        # extract channels
        channels = channel_geometries.keys()

        # loop over the channels
        for channel in channels:
            x, y = channel_geometries[channel]
            channel_obj = Channel(x, y, channel=channel)
            self.channels = np.append(self.channels, channel_obj)

    def connect_channels(self):
        """ Calculate the channel graph from its channels and add broken/
        missing channels to the channels list
        """
        # extract the in-between-channels geometry assuming a rectangular grid
        self._calculate_interchannel_stats()

        # add broken / missing channels
        x = self.x_min
        y = self.y_min

        more_x = True
        more_y = True

        while more_y:
            while more_x:
                # Find channel with x,y coordinate
                tmp_channel = self.get_channel_from_xy(x, y)

                # Insert stub channel if missing
                if tmp_channel is None:
                    broken_channel = Channel(x, y)
                    broken_channel.set_broken(True)

                    self.channels = np.append(self.channels, broken_channel)

                # update x
                x = x + self.x_between
                if x > self.x_max:
                    x = self.x_min
                    # exit while loop
                    more_x = False

            # update y
            y = y + self.y_between
            # reenable inner loop
            more_x = True
            if y > self.y_max:
                # exit while loop
                more_y = False

        # for every channel start looking for its neighbors
        for channel in self.channels:
            for candidate in self.channels:
                rel = self._is_neighbor(channel, candidate)

                if rel == 'left':
                    channel.left = candidate
                elif rel == 'upper':
                    channel.upper = candidate
                elif rel == 'right':
                    channel.right = candidate
                elif rel == 'lower':
                    channel.lower = candidate

        # set the origin channel
        self.origin = self.get_channel_from_xy(self.x_min, self.y_min)

    def _calculate_interchannel_stats(self):
        """ Calculate the interchannel statistics
        """        
        self.x_between = -1
        self.y_between = -1

        self.x_min = self.channels[0].x
        self.x_max = self.channels[0].x
        self.y_min = self.channels[0].y
        self.y_max = self.channels[0].y

        for channel_i in self.channels:
            # extract min/max
            if channel_i.x < self.x_min:
                self.x_min = channel_i.x

            if channel_i.x > self.x_max:
                self.x_max = channel_i.x

            if channel_i.y < self.y_min:
                self.y_min = channel_i.y

            if channel_i.y > self.y_max:
                self.y_max = channel_i.y

            # double loop over channels for comparison
            for channel_j in self.channels:
                # compare interchannel distance for the current pair
                tmp_x_between = abs(channel_i.x - channel_j.x)
                tmp_y_between = abs(channel_i.y - channel_j.y)

                # assign if smaller but non-zero or if the initial is detected
                # and non-zero
                if tmp_x_between < self.x_between or self.x_between < 0:
                    if tmp_x_between > 0:
                        self.x_between = tmp_x_between

                if tmp_y_between < self.y_between or self.y_between < 0:
                    if tmp_y_between > 0:
                        self.y_between = tmp_y_between

        if self.x_between == -1:
            self.x_between = 1 # arbitary positive
            self.nb_cols = 1
        else:
            self.nb_cols = int((self.x_max - self.x_min) / self.x_between + 1)

        if self.y_between == -1:
            self.y_between = 1
            self.nb_rows = 1
        else:
            self.nb_rows = int((self.y_max - self.y_min) / self.y_between + 1)

    def _is_neighbor(self, channel, candidate):
        """ Check whether the candidate is a neighbor of the given channel
        and return the relation or None
        """
        x_diff = channel.x - candidate.x
        y_diff = channel.y - candidate.y

        if abs(x_diff) == self.x_between and y_diff == 0:
            if x_diff > 0:
                return 'left'
            else:
                return 'right'

        if abs(y_diff) == self.y_between and x_diff == 0:
            if y_diff > 0:
                return 'lower'
            else:
                return 'upper'

        # no neighbor == return None
        return None

    def validate_probe_graph(self):
        """ If the graph is valid it should be possible to back reference from
        every neighbor
        """
        error_found = False

        # loop over all channels and try to reference back from a neighbor
        for channel in self.channels:
            if not channel.left is None:
                if not channel.left.right is channel:
                    print('broken graph left {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.right is None:
                if not channel.right.left is channel:
                    print('broken graph right {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.upper is None:
                if not channel.upper.lower is channel:
                    print('broken graph upper {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.lower is None:
                if not channel.lower.upper is channel:
                    print('broken graph lower {} {}'.format(channel.x, channel.y))
                    error_found = True

        if not error_found:
            print('-- probe graph is found to be healthy')

    def get_channel_from_xy(self, x, y):
        """ Return the channel on the probe with the given x, y coordinate,
        or None if no channel was found.
        """
        for channel in self.channels:
            if channel.x == x and channel.y == y:
                return channel

        # if no match is found return None
        return None

    def sort_given_channel_idx(self, channel_idxs):
        """ Sort the given channel idxs from left to right (fastest changing)
        and bottom to top
        """
        output_idxs = np.array([]) # initialise return variable
        traversing = True # true as long as the probe has not been traversed

        # start traversing from the origin of the probe
        current_channel = self.origin
        next_row = self.origin.upper

        # traverse the probe
        while traversing:
            if not current_channel.is_broken():
                # extract the index of the current channel
                channel_idx = current_channel.channel
                if channel_idx in channel_idxs:
                    output_idxs = np.append(output_idxs, current_channel.channel)

            # update current channel
            current_channel = current_channel.right
            
            # if the next channel is None go to the next row
            if current_channel is None:
                current_channel = next_row

                # if this next row turns out to be None we return the result
                if current_channel is None:
                    # the traversing doesn't have to changed explicitly because
                    # the return statement breaks the while loop
                    return output_idxs

                # update next row
                next_row = current_channel.upper

    def get_channels_from_zone(self, nb_channels, x_reach, x_offset):
        """ Return a list channels idxs generated from the given geometrical
        zone. A zone is always created from the origin.

        Broken channels are represented by Nones
        """
        output_idxs = np.array([], dtype=np.int)

        if x_offset >= x_reach:
            raise Exception('Given starting index has to smaller than the given width')

        if x_reach > self.nb_cols:
            raise Exception('Given width cannot exceed the probe dimensions')

        # go to start
        current_channel = self.origin
        next_row = current_channel.upper

        # apply offset
        if x_offset > 0:
            # walk to right
            for _ in range(x_offset):
                current_channel = current_channel.right

        # walk right to left in this row
        walk_right = x_reach - x_offset

        keep_walking = True
        while keep_walking:
            # walk to the right on current row
            while walk_right > 0:
                # only add real channel
                if current_channel.channel is not None:
                    output_idxs = np.append(output_idxs, current_channel.channel)
                if output_idxs.size == nb_channels:
                    keep_walking = False
                    # break from inner loop
                    break

                current_channel = current_channel.right
                walk_right -= 1

            # update to next_row
            current_channel = next_row

            if current_channel is not None:
                next_row = current_channel.upper
                walk_right = x_reach

            # no more channels left
            else:
                break

        # return the raw data channels for this zone
        return output_idxs


class Channel:
    """ A channel class with neighbor support

    Parameters
    ----------
        x (float) : x-coordinate

        y (float) : y-coordinate

        channel (int) : corresponding channel id in the recording data

    Attributes
    ----------
        x (float) : x-coordinate

        y (float) : y-coordinate

        channel (int) : corresponding raw data channel index

        left (Channel) : left neighboring channel (negative x direction)

        right (Channel) : right neighboring channel (positive x direction)

        lower (Channel) : lower neighboring channel (negative y direction)

        upper (Channel) : upper neighboring channel (positive x direction)
    """

    def __init__(self, x, y, channel=None, broken=False):
        self.x = x
        self.y = y
        self.channel = channel

        self.left = None
        self.right = None
        self.lower = None
        self.upper = None

        self.broken = broken

    def set_broken(self, broken):
        """ Set the broken attribute to the given boolean broken
        """
        self.broken = broken

    def is_broken(self):
        """ Return whether this channel is broken or not
        """
        return self.broken
