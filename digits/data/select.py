# -*- coding: utf-8 -*-
"""
Methods to (sub)-select an imported dataset given some criteria.
"""
import numpy as np


# Selectors based on ranges
#
# workaround for this slicing issue
#   https://bpaste.net/show/bb7601ca91a8
# as suggested by papna in #pydata is to use
# swaplevel().sort().loc().swaplevel().sort()

def fromtimerange(samples, tmin, tmax):
    """
    Sub-select time window from samples data frame.
    Expects 2nd layer sample label to be named 't_NNNN'.
    """
    timepoints = np.unique(samples.columns.get_level_values(level='sample'))
    if timepoints[0][0] != 't':
        raise ValueError("Did not find time-values, are we still in time domain?")

    if isinstance(tmin, int) and isinstance(tmax, int):
        zlen = len(timepoints[0]) - 2
        tmin = 't_' + str(tmin).zfill(zlen)
        tmax = 't_' + str(tmax).zfill(zlen)

    for param in [tmin, tmax]:
        if param not in timepoints:
            raise ValueError("Invalid value for parameter: {0}".format(param))

    return samples.swaplevel(0,1,axis=1).sort_index(axis=1).loc[:,(slice(tmin,tmax),)].swaplevel(0,1,axis=1).sort_index(axis=1)

def fromfreqrange(samples, fmin, fmax):
    """
    Sub-select frequency window from samples data frame.
    Expects 2nd layer sample label to be named 'f_NNNN'.
    """
    freqpoints = np.unique(samples.columns.get_level_values(level='sample'))
    if freqpoints[0][0] != 'f':
        raise ValueError("Did not find freq-values, did you transform to freq domain?")

    freqfloats = np.array([ float(f.split("_")[1]) for f in freqpoints ])
    if fmax=='max':
        fmax = freqfloats[-1]
    if fmin=='min':
        fmin = freqfloats[0]
    try:
        fmax = freqpoints[ freqfloats >= float(fmax) ][0]
    except IndexError:
        raise ValueError("fmax({}) invalid, must be <= {}".format(freqfloats[-1]))
    try:
        fmin = freqpoints[ freqfloats <= float(fmin) ][-1]
    except IndexError:
        raise ValueError("fmin({}) invalid, must be >= {}".format(freqfloats[0]))
    #print("fmin: {}, fmax: {}".format(fmin, fmax))

    return samples.swaplevel(0,1,axis=1).sort_index(axis=1).loc[:,(slice(fmin,fmax),)].swaplevel(0,1,axis=1).sort_index(axis=1)

def fromchannelrange(samples, chanmin, chanmax):
    """
    Sub-select from a range of channels. Usually fromchannellist or
    fromchannelblacklist is more useful.
    """
    channels = np.unique(samples.columns.get_level_values(level='channel'))
    for param in [chanmin, chanmax]:
        if param not in channels:
            raise ValueError("Invalid value for parameter: {0}".format(param))

    return samples.swaplevel(0,1,axis=1).sort_index(axis=1).loc[:, (slice(chanmin,chanmax),)].swaplevel(0,1,axis=1).sort_index(axis=1)



def fromchannellist(samples, channels):
    """
    Returns a sample data frame that only contains channels according to the
    channels input list.
    """
    if not isinstance(channels, list):
        raise ValueError("Channel argument must be a list.")

    return samples.loc[:, (channels, slice(None))]


def fromchannelblacklist(samples, channels):
    """
    Inverse function to fromchannellist.
    """
    if not isinstance(channels, list):
        raise ValueError("Channel argument must be a list.")
    all_channels = np.unique(samples.columns.get_level_values(level='channel'))

    return fromchannellist(samples, list(set(all_channels) - set(channels)))


# Dual selectors (need to reshape/filter targets as well!)

def fromsessionlist(samples, targets, sessions):
    """
    Return a samples and targets data frame filtered by a list of session names.
    """
    if not isinstance(sessions, list):
        raise ValueError("Session argument must be a list.")

    if isinstance(sessions[0], int):
        sessions = [str(x).zfill(2) for x in sessions]

    # this is very digits package specific
    if len(sessions[0]) < 2:
        sessions = [x.zfill(2) for x in sessions]

    samples = samples.loc[(slice(None), (sessions)), :]
    targets = targets.loc[(slice(None), (sessions)), :]

    return samples, targets

def fromsessionblacklist(samples, targets, sessions):
    """
    Inverse function to fromsessionlist.
    """
    if not isinstance(sessions, list):
        raise ValueError("Session argument must be a list.")

    all_sess = np.unique(samples.index.get_level_values(level='session'))

    return fromsessionlist(samples, targets,list(set(all_sess) - set(sessions)))


def fromtriallist(samples, targets, trials):
    """
    Return a samples and targets data frame filtered by a list of trial names.
    """
    if not isinstance(trials, list):
        raise ValueError("Trials argument must be a list.")

    samples = samples.swaplevel(0,2,axis=0).sort_index(axis=0).loc[(trials,),:].swaplevel(0,2,axis=0).sort_index(axis=0)
    targets = targets.swaplevel(0,2,axis=0).sort_index(axis=0).loc[(trials,),:].swaplevel(0,2,axis=0).sort_index(axis=0)

    return samples, targets


def frompresentationlist(samples, targets, presentations):
    """
    Return a samples and targets data frame filtered by a list of presentation names.
    """
    if not isinstance(presentations, list):
        raise ValueError("Presentation argument must be a list.")

    if isinstance(presentations[0], int):
        presentations = [str(x) for x in presentations]

    samples = samples.swaplevel(0,3,axis=0).sort_index(axis=0).loc[(presentations,),:].swaplevel(0,3,axis=0).sort_index(axis=0)
    targets = targets.swaplevel(0,3,axis=0).sort_index(axis=0).loc[(presentations,),:].swaplevel(0,3,axis=0).sort_index(axis=0)

    return samples, targets


def fromtargetlist(samples, targets, targetlist):
    """
    Return a samples and targets data frame filtered by target values
    (needs at least 2 targets in the list).
    """
    if len(targetlist)<2:
        raise ValueError("Targetlist argument must be a list with size >1.")

    mask = targets.label == targetlist.pop()
    for target in targetlist:
        mask = (mask) | (targets.label == target)
    return samples[mask], targets[mask]

def jointargets(targets, groups):
    """
    Creates two labels and maps all old labels into the two categories.
    Combine with fromtargetlist() for arbitrary groups.
    Group A,B will be assigned 0,1 respectively.
    """
    (groupa, groupb) = groups
    targets_new = targets.copy()
    for ix,target in targets.iterrows():
        if target.label in groupa:
            targets_new.loc[ix] = 0
        else:
            targets_new.loc[ix] = 1
    return targets_new


# Helpers

def getchannelnames(samples):
    return __getlevelnames(samples, 'channel', 1)

def getsamplingnames(samples):
    return __getlevelnames(samples, 'sample', 1)

def getsubjectnames(samples):
    return __getlevelnames(samples, 'subject', 0)

def getsessionnames(samples):
    return __getlevelnames(samples, 'session', 0)

def gettrialnames(samples):
    return __getlevelnames(samples, 'trial', 0)

def getpresentationnames(samples):
    return __getlevelnames(samples, 'presentation', 0)

def __getlevelnames(samples, level, axis):
    if axis == 0:
        names = samples.index.get_level_values(level=level)
    else:
        names = samples.columns.get_level_values(level=level)
    return np.unique(names).tolist()
