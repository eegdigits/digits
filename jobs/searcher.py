# digits related code
from digits.data import matimport
from digits.data import select
from digits.metrics.cfm import Confusion, ConfusionGrid
from digits.metrics import scores
from digits.transform.dimreduction import SubsampleTransform, FFTransform
from digits.inspect.plot import normhist
from digits.utils import dotdict, getoutname
from digits.transform.shaper import CSPFlatten
from digits.metrics.rdm import RDM

# system libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV

from mne.decoding import CSP

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import yaml
from os import path
from glob import glob
from subprocess import Popen, STDOUT, PIPE
from shutil import which
from pprint import pprint
from tempfile import NamedTemporaryFile


class Search(object):

    jobtemplate = '''
#PBS -m n
#PBS -o $HOME/logs/
#PBS -j oe
#PBS -l walltime={time},nodes=1:ppn=8,mem={mem}
#PBS -N {search}_{subject}
#PBS -d .

source ~/.venvs/digits/bin/activate
python -c "from searcher import {search}; {search}(subject='{subject}', config_file='{config}').run()"
'''


    def __init__(self, subject=None, config_file=None,
                 cores=8, time='36:0:0', mem='36gb'):
        with open(config_file, 'r') as f:
            self.config = dotdict(yaml.load(f)['config'])
        self.configpipeline = self._set_pipeline()
        self.config.subject = subject
        self.config.cores = cores
        self.results = dotdict()
        self.results.data = {}
        self.results.params = {}
        self.outname = getoutname(self.config)
        self.jobfile = self.jobtemplate.format(time=time, mem=mem,
                                               config=config_file,
                                               subject=subject,
                                               search=self.__class__.__name__)


    def exists(self):
        return path.exists('results/'+self.outname+'.npz')

    def __transform(self):
        config = self.config
        imp = matimport.Importer(dataroot='../data/thomas/artcorr/')
        imp.open(config.subject+'.h5')

        samples = imp.store.samples
        targets = imp.store.targets
        samples = select.fromtimerange(samples, config.t0, config.t1)
        samples, targets = select.fromsessionblacklist(samples, targets, ['01'])
        samples = select.fromchannelblacklist(samples, ['LHEOG', 'RHEOG', 'IOL'])
        samples = SubsampleTransform(width=config.subsample_width, verbose=True).transform(samples)
        if config.fft:
            samples = FFTransform(verbose=True, bins=config.size, fmin=config.fmin, fmax=config.fmax,
                                  power=config.power, rate=config.subsample_width/1000.).transform(samples)
        self.samples = samples
        self.targets = targets

    def _shape(self, X):
        return X

    def __save(self):
        outfile = path.join('results', getoutname(self.config))
        np.savez(outfile, results=dict(self.results), config=dict(self.config))

    def _set_pipeline(self):
        self.config.pipeline = Pipeline([(None, None)])
        self.config.params = []

    def submit(self):
        #print(self.jobfile)
        if which('qsub'):
            p = Popen(['qsub'], shell=True, stdout=PIPE, stderr=STDOUT, stdin=PIPE)
            # torque qsub *needs* Ascii
            (pout, perr) = p.communicate(input=self.jobfile.encode('ascii'))
            print(pout.decode('utf-8'))
        else:
            self.run()

    def run(self):
        pprint(self.config)
        self.__transform()
        results = self.results
        config = self.config
        samples = self.samples
        targets = self.targets
        for dix,(d1,d2) in enumerate(combinations(np.arange(10), 2)):
            print("running gridsearch for [{},{}]".format(d1,d2))

            tmp_samples, tmp_targets = select.fromtargetlist(samples, targets, [d1, d2])

            #mmscaler = MinMaxScaler(feature_range=(-1,1))
            split = train_test_split(tmp_samples, tmp_targets.values.flatten(),
                                     test_size=0.1, stratify=tmp_targets.values.flatten())
            X_train, X_test, y_train, y_test = split

            grid = GridSearchCV(config.pipeline, config.params, n_jobs=config.cores,
                                cv=10, verbose=1, error_score=0.5)
            _ = grid.fit(self._shape(X_train), y_train)
            params = grid.grid_scores_

            results.data[(d1, d2)] = [(grid.score(self._shape(X_test), y_test), grid.best_params_)]
            results.params[(d1, d2)] = sorted(params, reverse=True, key=lambda x: x.mean_validation_score)
            print(results.params[(d1, d2)])
        self.__save()


class LDASearch(Search):

    def __init__(self, **kwargs):
        super(LDASearch, self).__init__(**kwargs, cores=1, mem='36gb', time='72:0:0')

    def _set_pipeline(self):
        self.config.pipeline = Pipeline([
            ('lda', LinearDiscriminantAnalysis())
        ])
        self.config.params = [
            {
                'lda__shrinkage': ['auto'],
                #'lda__solver': ['lsqr', 'eigen'],
                'lda__solver': ['lsqr'],
            },
            #{
            #    'lda__shrinkage': np.linspace(0, 0.2, 10),
            #    'lda__solver': ['lsqr'],
            #},
        ]


class CSPSearch(Search):

    def __init__(self, **kwargs):
        super(CSPSearch, self).__init__(**kwargs, cores=1, mem='9gb', time='6:0:0')

    def _set_pipeline(self):
        self.config.pipeline = Pipeline([
                    ('csp', CSP()),
                    ('flat', CSPFlatten()),
                    ('lda', LinearDiscriminantAnalysis())
        ])

        self.config.params = [{
            'csp__n_components': np.arange(2,7),
            'csp__reg': ['ledoit_wolf'],
            'csp__transform_into': ['csp_space'],
            'lda__shrinkage': ['auto'],
            'lda__solver': ['lsqr'],
        },
        {
            'csp__n_components': np.arange(2,7),
            'csp__reg': ['ledoit_wolf'],
            'csp__transform_into': ['csp_space'],
            'lda__shrinkage': np.linspace(0.01, 1, 20),
            'lda__solver': ['lsqr'],
        }]

    def _shape(self, X):
        samplen = len(X)
        chlen = len(select.getchannelnames(X))
        return X.values.reshape(samplen, chlen, -1).astype('float32')


class KNNSearch(Search):

    def __init__(self, **kwargs):
        super(KNNSearch, self).__init__(**kwargs)

    def _set_pipeline(self):
        self.config.pipeline = Pipeline([
                    ('knn', KNeighborsClassifier())
        ])

        self.config.params = [
            {
                'knn__n_neighbors': np.arange(5,15),
                'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            },
        ]


class SVCSearch(Search):

    def __init__(self, **kwargs):
        super(SVCSearch, self).__init__(**kwargs)

    def _set_pipeline(self):
        self.config.pipeline = Pipeline([
                ('svc', SVC())
        ])
        self.config.params = [{
            'svc__kernel': ['linear'],
            'svc__C': np.logspace(-8,2,20),
        }]

    def run(self):
        pprint(self.config)
        self.__transform()
        results = self.results
        config = self.config
        samples = self.samples
        targets = self.targets
        for dix,(d1,d2) in enumerate(combinations(np.arange(10), 2)):
            print("running gridsearch for [{},{}]".format(d1,d2))

            tmp_samples, tmp_targets = select.fromtargetlist(samples, targets, [d1, d2])

            split = train_test_split(tmp_samples, tmp_targets.values.flatten(),
                                     test_size=0.1, stratify=tmp_targets.values.flatten())
            X_train, X_test, y_train, y_test = split

            grid = GridSearchCV(config.pipeline, config.params, n_jobs=config.cores,
                                pre_dispatch=config.cores, cv=10, verbose=1)
            _ = grid.fit(X_train, y_train)
            params = grid.grid_scores_
            results.data[(d1, d2)] = [(grid.score(X_test, y_test), grid.best_params_)]

            C_lin = grid.best_params_['svc__C']
            ssqs = np.logspace(-8,8,20)
            Cs = ssqs * C_lin
            gammas = 1/(2 * ssqs)

            newparams = [
                {
                    'svc__kernel': ['rbf'],
                    'svc__C': [c],
                    'svc__gamma': [gamma]
                } for c,gamma in zip(Cs, gammas)
            ]
            grid = GridSearchCV(config.pipeline, newparams, n_jobs=config.cores,
                                pre_dispatch=config.cores, cv=10, verbose=1)
            _ = grid.fit(X_train, y_train)
            results.data[(d1, d2)].append((grid.score(X_test, y_test), grid.best_params_))
            params.extend(grid.grid_scores_)
            results.params[(d1, d2)] = sorted(params, reverse=True, key=lambda x: x.mean_validation_score)
            print(results.params[(d1, d2)])
            print(results.data[d1, d2])
        self.__save()


if __name__ == '__main__':

    subjects = [3130, 3131, 3132, 3134, 3135, 3136, 3138, 3146, 3147, 3149,
                3154, 3156, 3157, 3158, 3159, 3161, 3162, 3233, 3237, 3239,
                3240, 3241, 3242, 3243, 3245, 3248, 3250, 3251, 3252, 3253,
                3255, 3260]

    models = {
        LDASearch: glob('configs/*.yaml'),
        CSPSearch: glob('configs/*.yaml'),
        SVCSearch: glob('configs/*.yaml'),
        KNNSearch: glob('configs/*lda_4.yaml') +  glob('configs/*nofft20.yaml')
    }

    for subject in subjects:
        for model, configs in models.items():
            for config in configs:
                s = model(subject=str(subject), config_file=config)
                if not s.exists():
                    print('submitting "{}"'.format(s.outname))
                    s.submit()
                else:
                    print('skipping "{}"'.format(s.outname))
