# -*- coding: utf-8 -*-
"""
This module is used for importing study specific .mat files.

Since all trials will be joined to a single dataset we cannot easily handle
single electrodes/channels from certain trials.
Thus, data loaded with this module is expected to be artifact free already.
"""
from os import path, mkdir, unlink
from glob import glob
from scipy import io
import numpy as np
import pandas as pd
from pandas.io.pytables import HDFStore
import warnings
import re
from digits.utils import dotdict


class UnmatchedDimensions(Exception): pass
class UnmatchedSubjects(Exception): pass
class InconsistentElectrodes(Exception): pass

class Session():
    """Simple Class for a session object holding 2 pandas dataframes: samples and targets"""
    def __init__(self, subject, sessionid, samples, digits, trials, channels):
        if len(digits) != len(samples):
            raise UnmatchedDimensions("sample and target length doesn't match")

        samplenames = ["t_"+str.zfill(x,4) for x in np.arange(0,1401).astype('str')]
        colix = pd.MultiIndex.from_product([channels, samplenames],
                                            names=['channel', 'sample'])

        rowsubjects = np.repeat(subject, len(trials))
        rowsessions = np.repeat(sessionid, len(trials))
        trials = trials.astype('str')
        rowruns = np.arange(len(trials)).astype('str')
        rowix = pd.MultiIndex.from_arrays([rowsubjects, rowsessions, trials, rowruns],
                                          names=['subject', 'session', 'trial', 'presentation'])

        samples = pd.DataFrame(samples, index=rowix, columns=colix)
        targets = pd.DataFrame(digits, index=rowix, columns=['label'])
        self.ds = dotdict({'samples': samples, 'targets': targets})


class Importer:
    """Main Class for importing mat files.

    Data will be stored in the samples and targets of the ds dictionary
    attribute and can be loaded or saved from and to a compressed hdf5 file.

    Attributes:
        dataroot: directory that contains the mat files
        ds: trivial dictionary containing:
            samples: a pandas dataframe with a MultiIndex composed of the session data
            targets: a pandas dataframe with the targets/labels
    """
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.ds = None
        self.store = None
        self.importpath = path.join(self.dataroot, 'imported')

    def __append(self, session):
        """append session to current object"""
        if (self.ds.samples.columns.get_level_values('channel') !=
            session.ds.samples.columns.get_level_values('channel')).any():
            print(self.ds.samples.columns.get_level_values('channel'))
            print(session.ds.samples.columns.get_level_values('channel'))
            raise InconsistentElectrodes('electrode labels do not match when merging datasets')

        self.ds.samples = self.ds.samples.append(session.ds.samples,
                                                 verify_integrity=True)
        self.ds.targets = self.ds.targets.append(session.ds.targets,
                                                 verify_integrity=True)

    def __sort(self):
        """MultiIndex Slicing operations require we sort all indices"""
        self.ds.samples.sort_index(level='channel', axis=1, inplace=True)
        #self.ds.samples.sort_index(axis=1, inplace=True)

    def get_session(self, subject, sessionid):
        """Add a single trial as target/samples pair from a mat file.

        Args:
            param1: (string): subject ID
            param2: (string): session ID

        Returns:
            A Session object containing samples and target data.
        """

        trialpath = glob(path.join(self.dataroot,
                                   subject + '-' + sessionid + '-*.mat'))
        if not trialpath or not path.exists(trialpath[0]):
            raise FileNotFoundError("no file for subject '{0}' and trial '{1}'".format(subject, sessionid))

        session = io.loadmat(trialpath[0])
        """
        >> session
        session =
            data: [1x1 struct]
        """

        data = session['data'][0,0]
        """
        >> session.data
        ans =
                    label: {64x1 cell}
                     time: {1x638 cell}
                    trial: {1x638 cell}
                     elec: [1x1 struct]
                      cfg: [1x1 struct]
                  TrlInfo: {638x16 cell}
            TrlInfoLabels: {16x1 cell}
        """
        channels = data[0][:,0] # label
        channels = np.array(channels.tolist()).flatten() # unify dtype
        samples = data[2][0, :] # trial
        samples = np.array([ x[:].flatten() for x in samples ], dtype='float32')
        cfg = data[4][0][0]
        trlinfo = data[5][:,:]
        trlinfolabels = data[6][:,0]

        """
        >> session.data.cfg
        ans =
                       method: 'spline'
                   badchannel: {2x1 cell}
                       trials: 'all'
                       lambda: 1.0000e-05
                        order: 4
                         elec: [1x1 struct]
            outputfilepresent: 'overwrite'
                     callinfo: [1x1 struct]
                      version: [1x1 struct]
                  trackconfig: 'off'
                  checkconfig: 'loose'
                    checksize: 100000
                 showcallinfo: 'yes'
                        debug: 'no'
                trackcallinfo: 'yes'
                trackdatainfo: 'no'
               missingchannel: {0x1 cell}
                     previous: [1x1 struct]
        """
        try:
            badchannels = cfg[1][0,0]
        except IndexError:
            badchannels = []


        """
        >> session.data.TrlInfoLabels
        ans =
            'time stamp original (EEG)'
            'time stamp new (EEG)'
            'task'
            'data part #'
            'trial #'
            'stimulus type'
            'EEG trigger'
            'encoding digit #'
            'time stamp original (E-Prime)'
            'set size'
            'probe type'
            'response'
            'ACC'
            'RT'
            'digit/probe presented'
            'probe position'
        """
        trials = trlinfo[:,4].astype('uint8')
        digits = trlinfo[:,14].astype('uint8')

        return Session(subject, sessionid, samples, digits, trials, channels)

    def add_session(self, subject, sessionid):
        """Concatenate a single Session to the current importer instance
        implicitly using __append().

        Args:
            param1: (string): subject ID
            param2: (string): session ID
        """
        session = self.get_session(subject, sessionid)
        if not self.ds:
            self.ds = dotdict({'samples': session.ds.samples,
                               'targets': session.ds.targets})
            return

        if sessionid in self.ds.samples.index.get_level_values('session'):
            warnings.warn("Session already added, doing nothing.")
            return

        if subject not in self.ds.samples.index.get_level_values('subject'):
            raise UnmatchedSubjects("Subjects don't match, will not add current session")
        # TODO: other checks ?

        self.__append(session)

    def import_all(self, subject):
        """Import all .mat files for a subject ID.

        Args:
            param1: (string): subject ID
        """
        trialpath = path.join(self.dataroot, '*' + subject + '*mat')
        trialfiles = sorted(glob(trialpath))
        if not trialfiles:
            raise FileNotFoundError(trialpath)

        sessionid_re = re.compile('.*' + subject + '-([0-9]+)-.*mat')
        sessionids = [sessionid_re.match(file).groups()[0] for file in trialfiles]
        for id in sessionids:
            self.add_session(subject, id)

        self.__sort()

    def save(self, filename, force=False):
        """Save the trials and samples arrays from the current importer
        instance to a dataset inside a lzf compressed hdf5 file for later use.

        Args:
            param1: (string): filename, will be stored in self.importpath

        Optional Args:
            force: (boolean) Wether or not to overwrite an existing file
                             (default: False)
        """
        try:
            mkdir(self.importpath)
        except FileExistsError:
            pass

        filename = path.join(self.importpath, filename)
        if path.exists(filename):
            if force:
                unlink(filename)
            else:
                 raise FileExistsError('Import file "' + filename + '" already exists.')

        self.__sort()
        self.store = HDFStore(filename, complib='lzo')
        self.store['samples'] = self.ds.samples
        self.store['targets'] = self.ds.targets
        self.store.close()

    def load(self, name):
        """Load a hdf5 file created with save() and attach the targets and
        samples array to the current importer instance.

        Args:
            param1: (string): a name for the dataset and the hdf5 file name
        """
        self.open(name)
        self.ds = dotdict({'samples': None, 'targets': None})
        self.ds.samples = self.store['samples']
        self.ds.targets = self.store['targets']
        self.store.close()

    def open(self, name):
        if not path.exists(self.importpath):
            raise FileNotFoundError(path.join(self.dataroot, 'imported'))
        filename = path.join(self.importpath, name)
        if not path.exists(filename):
            raise FileExistsError(filename)
        self.store = HDFStore(filename)

    def close(self, name):
        self.store.close()
