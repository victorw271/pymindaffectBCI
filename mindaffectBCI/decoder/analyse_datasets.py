#  Copyright (c) 2019 MindAffect B.V. 
<<<<<<< HEAD
#  Author: Jason Farquhar <jadref@gmail.com>
=======
#  Author: Jason Farquhar <jason@mindaffect.nl>
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

<<<<<<< HEAD
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
import numpy as np
from mindaffectBCI.decoder.offline.datasets import get_dataset
from mindaffectBCI.decoder.model_fitting import  MultiCCA, LinearSklearn, init_clsfr, BaseSequence2Sequence
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, plot_factoredmodel, plot_trial
=======
import numpy as np
import sklearn
from mindaffectBCI.decoder.datasets import get_dataset
from mindaffectBCI.decoder.model_fitting import BaseSequence2Sequence, MultiCCA, FwdLinearRegression, BwdLinearRegression, LinearSklearn
try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import LinearSVR, LinearSVC
    from sklearn.model_selection import GridSearchCV
except:
    pass
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, plot_factoredmodel
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
from mindaffectBCI.decoder.scoreStimulus import factored2full, plot_Fe
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, print_decoding_curve, plot_decoding_curve, flatten_decoding_curves, score_decoding_curve
from mindaffectBCI.decoder.scoreOutput import plot_Fy
from mindaffectBCI.decoder.preprocess import preprocess, plot_grand_average_spectrum
<<<<<<< HEAD
from mindaffectBCI.decoder.utils import askloadsavefile, block_permute
from mindaffectBCI.decoder.preprocess_transforms import make_preprocess_pipeline
from mindaffectBCI.decoder.stim2event import plot_stim_encoding
=======
from mindaffectBCI.decoder.trigger_check import triggerPlot
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
import matplotlib.pyplot as plt
import gc
import re
import traceback

<<<<<<< HEAD
try:
    from sklearn.model_selection import GridSearchCV
except:
    pass

def get_train_test_indicators(X,Y,train_idx=None,test_idx=None):
    train_ind=None
    test_ind=None

    # convert idx to indicators
    if test_idx is not None:
        test_ind = np.zeros((X.shape[0],),dtype=bool)
        test_ind[test_idx] = True
    if train_idx is not None:
        train_ind = np.zeros((X.shape[0],),dtype=bool)
        train_ind[train_idx] = True

    # compute missing train/test indicators
    if train_ind is None and test_ind is None:
        # if nothing set use everything
        train_ind = np.ones((X.shape[0],),dtype=bool)
        test_ind  = np.ones((X.shape[0],),dtype=bool)
    elif train_ind is None: # test but no train
        train_ind = np.logical_not(test_ind)
    elif test_ind is None: # train but no test
        test_ind = np.logical_not(train_ind)
=======
def analyse_dataset(X:np.ndarray, Y:np.ndarray, coords, outfile, model:str='cca', test_idx=None, cv=True, tau_ms:float=300, fs:float=None,  rank:int=1, evtlabs=None, offset_ms=0, center=True, tuned_parameters=None, ranks=None, retrain_on_all=True, **kwargs):
    """ cross-validated training on a single datasets and decoing curve estimation

    Args:
        X (np.ndarray): the X (EEG) sequence
        Y (np.ndarray): the Y (stimulus) sequence
        coords ([type]): array of dicts of meta-info describing the structure of X and Y
        fs (float): the data sample rate (if coords is not given)
        model (str, optional): The type of model to fit, as in `model_fitting.py`. Defaults to 'cca'.
        cv (bool, optional): flag if we should train with cross-validation using the cv_fit method. Defaults to True.
        test_idx (list-of-int, optional): indexs of test-set trials which are *not* passed to fit/cv_fit. Defaults to True.
        tau_ms (float, optional): length of the stimulus-response in milliseconds. Defaults to 300.
        rank (int, optional): rank of the decomposition in factored models such as cca. Defaults to 1.
        evtlabs ([type], optional): The types of events to used to model the brain response, as used in `stim2event.py`. Defaults to None.
        offset_ms ((2,):float, optional): Offset for analysis window from start/end of the event response. Defaults to 0.

    Raises:
        NotImplementedError: if you use for a model which isn't implemented

    Returns:
        score (float): the cv score for this dataset
        dc (tuple): the information about the decoding curve as returned by `decodingCurveSupervised.py`
        Fy (np.ndarray): the raw cv'd output-scores for this dataset as returned by `decodingCurveSupervised.py` 
        clsfr (BaseSequence2Sequence): the trained classifier
    """
    # extract dataset info
    if coords is not None:
        fs = coords[1]['fs'] 
        print("X({})={}, Y={} @{}hz".format([c['name'] for c in coords], X.shape, Y.shape, fs))
        # Write metrics to file
        with open('metrics.txt', 'a') as outfile:
            outfile.write("X({})={}, Y={} @{}hz \n".format([c['name'] for c in coords], X.shape, Y.shape, fs))
    else:
        print("X={}, Y={} @{}hz".format(X.shape, Y.shape, fs))
        # Write metrics to file
        with open('metrics.txt', 'a') as outfile:
            outfile.write("X={}, Y={} @{}hz \n".format(X.shape, Y.shape, fs))
    tau = int(tau_ms*fs/1000)
    offset=int(offset_ms*fs/1000)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    return train_ind, test_ind

<<<<<<< HEAD
def load_and_fit_dataset(loader, filename, model:str='cca', train_idx:slice=None, test_idx:slice=None, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.analyse_datasets import init_clsfr, get_train_test_indicators
    X, Y, coords = loader(filename, **loader_args)

    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
    fs = coords[1]['fs'] 
    print("X({})={}, Y={} @{}hz".format([c['name'] for c in coords], X.shape, Y.shape, fs))
    clsfr = init_clsfr(model=model, fs=fs, **clsfr_args)
 
     # do train/test split
    if test_idx is None and train_idx is None:
        X_train = X
        Y_train = Y
    else:
        train_ind, test_ind = get_train_test_indicators(X,Y,train_idx,test_idx)
        print("Training Idx: {}\nTesting Idx :{}\n".format(np.flatnonzero(train_ind),np.flatnonzero(test_ind)))
        X_train = X[train_ind,...]
        Y_train = Y[train_ind,...]

    clsfr.fit(X_train,Y_train)
    return clsfr, filename, X, Y, coords

def load_and_score_dataset(loader, filename, clsfrs:list, train_idx:slice=None, test_idx:slice=None, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.analyse_datasets import get_train_test_indicators
    X, Y, coords = loader(filename, **loader_args)
    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    train_ind, test_ind = get_train_test_indicators(X,Y,train_idx,test_idx)
    X = X[test_ind,...]
    Y = Y[test_ind,...]

    scores = [ None for _ in clsfrs ]
    for i,c in enumerate(clsfrs):
        print('.',end='')
        #try:
        scores[i] = c.score(X,Y) 
        #except:
        #    scores[i] = -1
    print(flush=True)
    return scores, filename


def load_and_decode_dataset(loader, filename, clsfrs:list, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    X, Y, coords = loader(filename, **loader_args)
    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
    dcs = [ None for _ in clsfrs ]
    for i,c in enumerate(clsfrs):
        try:
            Fy = c.predict(X,Y)
            (dc) = decodingCurveSupervised(Fy, marginalizedecis=True, minDecisLen=clsfr.minDecisLen, bwdAccumulate=clsfr.bwdAccumulate, priorsigma=(clsfr.sigma0_,clsfr.priorweight), softmaxscale=clsfr.softmaxscale_, nEpochCorrection=clsfr.startup_correction)
            dcs[i] = dc 
        except:
            dcs[i] = None
    return dcs, filename

def print_decoding_curves(decoding_curves):
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    return print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))

def plot_decoding_curves(decoding_curves, labels=None):
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    plot_decoding_curve(int_len,prob_err,labels=labels)

def plot_trial_summary(X, Y, Fy, Fe=None, Py=None, fs=None, label=None, evtlabs=None, centerx=True, xspacing=10, sumFy=True, Yerr=None, block=False):
    """generate a plot summarizing the inputs (X,Y) and outputs (Fe,Fe) for every trial in a dataset for debugging purposes
=======
    # Write metrics to file
    with open('metrics.txt', 'a') as outfile:
        outfile.write('Cscale={} \n'.format(Cscale))

    # create the model if not provided
    if isinstance(model,BaseSequence2Sequence):
        clsfr = model
    elif model=='cca' or model is None:
        clsfr = MultiCCA(tau=tau, offset=offset, rank=rank, evtlabs=evtlabs, center=center, **kwargs)
    elif model=='bwd':
        clsfr = BwdLinearRegression(tau=tau, offset=offset, evtlabs=evtlabs, center=center, **kwargs)
    elif model=='fwd':
        clsfr = FwdLinearRegression(tau=tau, offset=offset, evtlabs=evtlabs, center=center, **kwargs)
    elif model == 'ridge': # should be equivalent to BwdLinearRegression
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=Ridge(alpha=0,fit_intercept=center), **kwargs)
    elif model == 'lr':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LogisticRegression(C=C,fit_intercept=center), labelizeY=True, **kwargs)
    elif model == 'svr':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LinearSVR(C=C), **kwargs)
    elif model == 'svc':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LinearSVC(C=C), labelizeY=True, **kwargs)
    elif isinstance(model,sklearn.linear_model) or isinstance(model,sklearn.svm):
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=model, labelizeY=True, **kwargs)
    elif model=='linearsklearn':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, **kwargs)
    else:
        raise NotImplementedError("don't  know this model: {}".format(model))

    # do train/test split
    if test_idx is None:
        X_train = X
        Y_train = Y
    else:
        test_ind = np.zeros((X.shape[0],),dtype=bool)
        test_ind[test_idx] = True
        train_ind = np.logical_not(test_ind)
        X_train = X[train_ind,...]
        Y_train = Y[train_ind,...]
        retrain_on_all = True

    # fit the model
    if cv:
        # hyper-parameter optimization by cross-validation
        if tuned_parameters is not None:
            # hyper-parameter optimization with cross-validation

            cv_clsfr = GridSearchCV(clsfr, tuned_parameters)
            print('HyperParameter search: {}'.format(tuned_parameters))
            cv_clsfr.fit(X_train, Y_train)
            means = cv_clsfr.cv_results_['mean_test_score']
            stds = cv_clsfr.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, cv_clsfr.cv_results_['params']):
                print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))

            clsfr.set_params(**cv_clsfr.best_params_) # Note: **dict -> k,v argument array
        
        if ranks is not None and isinstance(clsfr,MultiCCA):
            # cross-validated rank optimization
            res = clsfr.cv_fit(X_train, Y_train, cv=cv, ranks=ranks, retrain_on_all=retrain_on_all)
        else:
            # cross-validated performance estimation
            res = clsfr.cv_fit(X_train, Y_train, cv=cv, retrain_on_all=retrain_on_all)

        Fy = res['estimator']

    else:
        print("Warning! overfitting...")
        clsfr.fit(X_train,Y_train)
        Fy = clsfr.predict(X, Y, dedup0=True)

    # use the raw scores, i.e. inc model dim, in computing the decoding curve
    rawFy = res['rawestimator'] if 'rawestimator' in res else Fy

    if test_idx is not None:
        # predict on the hold-out test set
        Fy_test = clsfr.predict(X[test_idx,...],Y[test_idx,...])
        # insert into the full results set
        tmp = list(rawFy.shape); tmp[-3]=X.shape[0]
        allFy = np.zeros(tmp,dtype=Fy.dtype)
        allFy[...,train_ind,:,:] = rawFy
        allFy[...,test_ind,:,:] = Fy_test
        rawFy = allFy
        res['rawestimator']=rawFy

    # assess model performance
    score=clsfr.audc_score(rawFy)
    print(clsfr)
    print("score={}".format(score))

    with open('metrics.txt', 'a') as outfile:
        outfile.write("score={} \n".format(score))

    # compute decoding curve
    (dc) = decodingCurveSupervised(rawFy, marginalizedecis=True, minDecisLen=clsfr.minDecisLen, bwdAccumulate=clsfr.bwdAccumulate, priorsigma=(clsfr.sigma0_,clsfr.priorweight), softmaxscale=clsfr.softmaxscale_, nEpochCorrection=clsfr.startup_correction)

    return score, dc, Fy, clsfr, res


def analyse_datasets(dataset:str, model:str='cca', dataset_args:dict=None, loader_args:dict=None, preprocess_args:dict=None, clsfr_args:dict=None, tuned_parameters:dict=None, **kwargs):
    """analyse a set of datasets (multiple subject) and generate a summary decoding plot.

    Args:
        dataset ([str]): the name of the dataset to load
        model (str, optional): The type of model to fit. Defaults to 'cca'.
        dataset_args ([dict], optional): additional arguments for get_dataset. Defaults to None.
        loader_args ([dict], optional): additional arguments for the dataset loader. Defaults to None.
        clsfr_args ([dict], optional): additional aguments for the model_fitter. Defaults to None.
        tuned_parameters ([dict], optional): sets of hyper-parameters to tune by GridCVSearch
    """    
    if dataset_args is None: dataset_args = dict()
    if loader_args is None: loader_args = dict()
    if clsfr_args is None: clsfr_args = dict()
    loader, filenames, _ = get_dataset(dataset,**dataset_args)
    scores=[]
    decoding_curves=[]
    nout=[]
    for i, fi in enumerate(filenames):
        print("{}) {}".format(i, fi))
        with open('metrics.txt', 'a') as outfile:
            outfile.write("\n\n {}) {} \n".format(i, fi))
        try:
            X, Y, coords = loader(fi, **loader_args)
            if preprocess_args is not None:
                X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
            score, decoding_curve, _, _, _ = analyse_dataset(X, Y, coords, outfile, model, tuned_parameters=tuned_parameters, **clsfr_args, **kwargs)
            nout.append(Y.shape[-1] if Y.ndim<=3 else Y.shape[-2])
            scores.append(score)
            decoding_curves.append(decoding_curve)
            del X, Y
            gc.collect()
        except Exception as ex:
            print("Error: {}\nSKIPPED".format(ex))
    avescore=sum(scores)/len(scores)
    avenout=sum(nout)/len(nout)
    print("\n--------\n\n Ave-score={}\n".format(avescore))
    # extract averaged decoding curve info
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    print("Ave-DC\n{}\n".format(print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))))
    plot_decoding_curve(int_len,prob_err)
    plt.suptitle("{} ({}) AUDC={:3.2f}(n={} ncls={})\nloader={}\nclsfr={}({})".format(dataset,dataset_args,avescore,len(scores),avenout-1,loader_args,model,clsfr_args))
    plt.savefig("{}_decoding_curve.png".format(dataset))
    plt.show()

    with open('metrics.txt', 'a') as outfile:
        outfile.write("\n--------\n\n Ave-score={}\n".format(avescore))
        outfile.write("Ave-DC\n{}\n".format(print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))))

def analyse_train_test(X:np.ndarray, Y:np.ndarray, coords, splits=1, label:str='', model:str='cca', tau_ms:float=300, fs:float=None,  rank:int=1, evtlabs=None, preprocess_args=None, clsfr_args:dict=None,  **kwargs):    
    """analyse effect of different train/test splits on performance and generate a summary decoding plot.

    Args:
        splits (): list of list of train-test split pairs.  
        dataset ([str]): the name of the dataset to load
        model (str, optional): The type of model to fit. Defaults to 'cca'.
        dataset_args ([dict], optional): additional arguments for get_dataset. Defaults to None.
        loader_args ([dict], optional): additional arguments for the dataset loader. Defaults to None.
        clsfr_args ([dict], optional): additional aguments for the model_fitter. Defaults to None.
        tuned_parameters ([dict], optional): sets of hyper-parameters to tune by GridCVSearch
    """    
    fs = coords[1]['fs'] if coords is not None else fs

    if isinstance(splits,int): 
        # step size for increasing training data split size
        splitsize=splits
        maxsize = X.shape[0]
        splits=[]
        for tsti in range(splitsize, maxsize, splitsize):
            # N.B. triple nest, so list, of lists of train/test pairs
            splits.append( ( (slice(tsti),slice(tsti,None)), ) )


    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    # run the train/test splits
    decoding_curves=[]
    labels=[]
    scores=[]
    Ws=[]
    Rs=[]
    for i, cv in enumerate(splits):
        # label describing the folding
        trnIdx = np.arange(X.shape[0])[cv[0][0]]
        tstIdx = np.arange(X.shape[0])[cv[0][1]]
        lab = "Trn {} ({}) / Tst {} ({})".format(len(trnIdx), (trnIdx[0],trnIdx[-1]), len(tstIdx), (tstIdx[0],tstIdx[-1]))
        labels.append( lab )

        print("{}) {}".format(i, lab))
        score, decoding_curve, Fy, clsfr = analyse_dataset(X, Y, coords, model, cv=cv, retrain_on_all=False, **clsfr_args, **kwargs)
        decoding_curves.append(decoding_curve)
        scores.append(score)
        Ws.append(clsfr.W_)
        Rs.append(clsfr.R_)

    # plot the model for each folding
    plt.figure(1)
    for w,r,l in zip(Ws,Rs,labels):
        plt.subplot(211)
        tmp = w[0,0,:]
        sgn = np.sign(tmp[np.argmax(np.abs(tmp))])
        plt.plot(tmp*sgn,label=l)
        plt.subplot(212)
        tmp=r[0,0,:,:].reshape((-1,))
        plt.plot(np.arange(tmp.size)*1000/fs, tmp*sgn, label=l)
    plt.subplot(211)
    plt.grid()
    plt.title('Spatial filter')
    plt.legend()
    plt.subplot(212)
    plt.grid()
    plt.title('Impulse response')
    plt.xlabel('time (ms)')

    plt.figure(2)
    # collate the results and visualize
    avescore=sum(scores)/len(scores)
    print("\n--------\n\n Ave-score={}\n".format(avescore))
    # extract averaged decoding curve info
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    print("Ave-DC\n{}\n".format(print_decoding_curve(np.mean(int_len,0),np.mean(prob_err,0),np.mean(prob_err_est,0),np.mean(se,0),np.mean(st,0))))
    plot_decoding_curve(int_len,prob_err)
    plt.legend( labels + ['mean'] )
    plt.suptitle("{} AUDC={:3.2f}(n={})\nclsfr={}({})".format(label,avescore,len(scores),model,clsfr_args))
    try:
        plt.savefig("{}_decoding_curve.png".format(label))
    except:
        pass
    plt.show()


def flatten_decoding_curves(decoding_curves):
    ''' take list of (potentially variable length) decoding curves and make into a single array '''
    il = np.zeros((len(decoding_curves),decoding_curves[0][0].size))
    pe = np.zeros(il.shape)
    pee = np.zeros(il.shape)
    se = np.zeros(il.shape)
    st = np.zeros(il.shape)
    # TODO [] : insert according to the actual int-len
    for di,dc in enumerate(decoding_curves):
        il_di = dc[0]
        ll = min(il.shape[1],il_di.size)
        il[di,:ll] = dc[0][:ll]
        pe[di,:ll] = dc[1][:ll]
        pee[di,:ll] = dc[2][:ll]
        se[di,:ll] = dc[3][:ll]
        st[di,:ll] = dc[4][:ll] 
    return il,pe,pee,se,st


def debug_test_dataset(X, Y, coords=None, label=None, tau_ms=300, fs=None, offset_ms=0, evtlabs=None, rank=1, model='cca', preprocess_args:dict=None, clsfr_args:dict=None, plotnormFy=False, triggerPlot=False, **kwargs):
    """Debug a data set, by pre-processing, model-fitting and generating various visualizations

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        coords ([type], optional): meta-info about the dimensions of X and Y. Defaults to None.
        label ([type], optional): textual name for this dataset, used for titles and save-file names. Defaults to None.
        tau_ms (int, optional): stimulus-response length in milliseconds. Defaults to 300.
        fs ([type], optional): sample rate of X and Y. Defaults to None.
        offset_ms (int, optional): offset for start of stimulus response w.r.t. stimulus time. Defaults to 0.
        evtlabs ([type], optional): list of types of stimulus even to fit the model to. Defaults to None.
        rank (int, optional): the rank of the model to fit. Defaults to 1.
        model (str, optional): the type of model to fit. Defaults to 'cca'.
        preprocess_args (dict, optional): additional arguments to send to the data pre-processor. Defaults to None.
        clsfr_args (dict, optional): additional arguments to pass to the model fitting. Defaults to None.

    Returns:
        score (float): the cv score for this dataset
        dc (tuple): the information about the decoding curve as returned by `decodingCurveSupervised.py`
        Fy (np.ndarray): the raw cv'd output-scores for this dataset as returned by `decodingCurveSupervised.py` 
        clsfr (BaseSequence2Sequence): the trained classifier
    """    
    fs = coords[1]['fs'] if coords is not None else fs
    if clsfr_args is not None:
        if 'tau_ms' in clsfr_args and clsfr_args['tau_ms'] is not None:
            tau_ms = clsfr_args['tau_ms']
        if 'offset_ms' in clsfr_args and clsfr_args['offset_ms'] is not None:
            offset_ms = clsfr_args['offset_ms']
        if 'evtlabs' in clsfr_args and clsfr_args['evtlabs'] is not None:
            evtlabs = clsfr_args['evtlabs']
        if 'rank' in clsfr_args and clsfr_args['rank'] is not None:
            rank = clsfr_args['rank']
    if evtlabs is None:
        evtlabs = ('re','fe')

    # work on copy of X,Y just in case
    X = X.copy()
    Y = Y.copy()

    tau = int(fs*tau_ms/1000)
    offset=int(offset_ms*fs/1000)    
    times = np.arange(offset,tau+offset)/fs
    
    if coords is not None:
        print("X({}){}".format([c['name'] for c in coords], X.shape))
    else:
        print("X={}".format(X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    ch_names = coords[2]['coords'] if coords is not None else None
    ch_pos = None
    if coords is not None and 'pos2d' in coords[2]:
        ch_pos = coords[2]['pos2d']
    elif not ch_names is None and len(ch_names) > 0:
        from mindaffectBCI.decoder.readCapInf import getPosInfo
        cnames, xy, xyz, iseeg =getPosInfo(ch_names)
        ch_pos=xy
    if ch_pos is not None:
        print('ch_pos={}'.format(ch_pos.shape))

    # visualize the dataset
    from mindaffectBCI.decoder.stim2event import stim2event
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, idOutliers
    import matplotlib.pyplot as plt

    print("Plot summary stats")
    if Y.ndim == 4: # already transformed
        Yevt = Y
    else: # convert to event
        Yevt = stim2event(Y, axis=-2, evtypes=evtlabs)
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Yevt[..., 0:1, :], tau=tau, offset=offset)
    plt.figure(11); plt.clf()
    plot_summary_statistics(Cxx, Cxy, Cyy, evtlabs, times, ch_names)
    plt.show(block=False)

    print('Plot global spectral properties')
    plot_grand_average_spectrum(X,axis=-2,fs=fs, ch_names=ch_names)
    plt.show(block=False)

    print("Plot ERP")
    plt.figure(12);plt.clf()
    plot_erp(Cxy, ch_names=ch_names, evtlabs=evtlabs, times=times)
    plt.suptitle("ERP")
    plt.show(block=False)
    plt.pause(.5)
    plt.savefig("{}_ERP".format(label)+".pdf",format='pdf')
    
    # fit the model
    # override with direct keyword arguments
    if clsfr_args is None: clsfr_args=dict()
    clsfr_args['evtlabs']=evtlabs
    clsfr_args['tau_ms']=tau_ms
    clsfr_args['fs']=fs
    clsfr_args['offset_ms']=offset_ms
    clsfr_args['rank']=rank
    #clsfr_args['retrain_on_all']=False # respect the folding, don't retrain on all at the end
    score, res, Fy, clsfr, cvres = analyse_dataset(X, Y, coords, model, **clsfr_args, **kwargs)
    Fe = clsfr.transform(X)


    # get the prob scores, per-sample
    if 'rawestimator' in cvres:   
        rawFy = cvres['rawestimator'] 
        Py = clsfr.decode_proba(rawFy, marginalizemodels=True, minDecisLen=-1)
    else:
        rawFy = cvres['estimator']
        Py = clsfr.decode_proba(Fy, marginalizemodels=True, minDecisLen=-1)

    Yerr = res[5] # (nTrl,nSamp)
    Perr = res[6] # (nTrl,nSamp)

    plot_trial_summary(X, Y, rawFy, fs=fs, Yerr=Yerr[:,-1], Py=Py, Fe=Fe, label=label)
    plt.show(block=False)
    plt.gcf().set_size_inches((15,9))
    plt.savefig("{}_trial_summary".format(label)+".pdf")
    plt.pause(.5)

    plt.figure(14)
    plot_decoding_curve(*res)
    plt.show(block=False)

    plt.figure(19)
    plt.subplot(211)
    plt.imshow(res[5], origin='lower', aspect='auto',cmap='gray', extent=[0,res[0][-1],0,res[5].shape[0]])
    plt.clim(0,1)
    plt.colorbar()
    plt.title('Yerr - correct-prediction (0=correct, 1=incorrect)?')
    plt.ylabel('Trial#')
    plt.grid()
    plt.subplot(212)
    plt.imshow(res[6], origin='lower', aspect='auto', cmap='gray', extent=[0,res[0][-1],0,res[5].shape[0]])
    plt.clim(0,1)
    plt.colorbar()
    plt.title('Perr - Prob of prediction error (0=correct, 1=incorrect)')
    plt.xlabel('time (samples)')
    plt.ylabel('Trial#')
    plt.grid()
    plt.show(block=False)

    if triggerPlot:
        plt.figure(20)
        triggerPlot(X,Y,fs, clsfr=clsfr, evtlabs=clsfr.evtlabs, tau_ms=tau_ms, offset_ms=offset_ms, max_samp=10000, trntrl=None, plot_model=False, plot_trial=True)
        plt.show(block=False)
        plt.savefig("{}_triggerplot".format(label)+".pdf",format='pdf')

    print("Plot Model")
    plt.figure(15)
    if hasattr(clsfr,'A_'):
        plot_erp(factored2full(clsfr.A_, clsfr.R_), ch_names=ch_names, evtlabs=evtlabs, times=times)
        plt.suptitle("fwd-model")
    else:
        plot_erp(factored2full(clsfr.W_, clsfr.R_), ch_names=ch_names, evtlabs=evtlabs, times=times)
        plt.suptitle("bwd-model")
    plt.show(block=False)

    print("Plot Factored Model")
    plt.figure(18)
    plt.clf()
    clsfr.plot_model(fs=fs,ch_names=ch_names)
    plt.savefig("{}_model".format(label)+".pdf")
    plt.show(block=False)
    
    # print("plot Fe")
    # plt.figure(16);plt.clf()
    # plot_Fe(Fe)
    # plt.suptitle("Fe")
    # plt.show()

    # print("plot Fy")
    # plt.figure(17);plt.clf()
    # plot_Fy(Fy,cumsum=True)
    # plt.suptitle("Fy")
    # plt.show()

    if plotnormFy:
        from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores, plot_normalizedScores
        print("normalized Fy")
        plt.figure(20);plt.clf()
        # normalize every sample
        ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1)
        plot_Fy(ssFy,label=label,cumsum=False)
        plt.show(block=False)

        plt.figure(21)
        plot_normalizedScores(Fy[4,:,:],ssFy[4,:,:],scale_sFy[4,:],decisIdx)
    
    plt.show()

    return score, res, Fy, clsfr, rawFy

def plot_trial_summary(X, Y, Fy, Fe=None, Py=None, fs=None, label=None, evtlabs=None, centerx=True, xspacing=10, sumFy=True, Yerr=None):
    """generate a plot summarizing the inputs (X,Y) and outputs (Fe,Fe) for every trial in a dataset for debugging purposes

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        Fy (nTrl,nSamp,nY): The output scores obtained by comping the stimulus-scores (Fe) with the stimulus information (Y)
        Fe ((nTrl,nSamp,nY,nE), optional): The stimulus scores, for the different event types, obtained by combining X with the decoding model. Defaults to None.
        Py ((nTrl,nSamp,nY), optional): The target probabilities for each output, derived from Fy. Defaults to None.
        fs (float, optional): sample rate of X, Y, used to set the time-axis. Defaults to None.
        label (str, optional): A textual label for this dataset, used for titles & save-files. Defaults to None.
        centerx (bool, optional): Center (zero-mean over samples) X for plotting. Defaults to True.
        xspacing (int, optional): Gap in X units between different channel lines. Defaults to 10.
        sumFy (bool, optional): accumulate the output scores before plotting. Defaults to True.
        Yerr (bool (nTrl,), optional): indicator for which trials the model made a correct prediction. Defaults to None.
    """    
    times = np.arange(X.shape[1])
    if fs is not None:
        times = times/fs
        xunit='s'
    else:
        xunit='samp'

    if centerx:
        X = X.copy() - np.mean(X,1,keepdims=True)
    if xspacing is None: 
        xspacing=np.median(np.diff(X,axis=-2).ravel())

    if sumFy:
        Fy = np.cumsum(Fy,axis=-2)

    Xlim = (np.min(X[...,0].ravel()),np.max(X[...,-1].ravel()))

    Fylim = (np.min(Fy.ravel()),np.max(Fy.ravel()))
    if Fe is not None:
        Felim = (np.min(Fe.ravel()),np.max(Fe.ravel()))

    if Py is not None:
        if Py.ndim>3 :
            print("Py: Multiple models? accumulated away")
            Py = np.sum(Py,0)

    if Fy is not None:
        if Fy.ndim>3 :
            print("Fy: Multiple models? accumulated away")
            Fy = np.mean(Fy,0)

    nTrl = X.shape[0]; w = int(np.ceil(np.sqrt(nTrl)*1.8)); h = int(np.ceil(nTrl/w))
    fig = plt.figure(figsize=(20,10))
    trial_grid = fig.add_gridspec( nrows=h, ncols=w, figure=fig, hspace=.05, wspace=.05) # per-trial grid
    nrows= 5 + (0 if Fe is None else 1) + (0 if Py is None else 1)
    ti=0
    for hi in range(h):
        for wi in range(w):
            if ti>=X.shape[0]:
                break

            gs = trial_grid[ti].subgridspec( nrows=nrows, ncols=1, hspace=0 )

            # pre-make bottom plot
            botax = fig.add_subplot(gs[-1,0])

            # plot X (0-3)
            fig.add_subplot(gs[:3,:], sharex=botax)
            plt.plot(times,X[ti,:,:] + np.arange(X.shape[-1])*xspacing)
            plt.gca().set_xticklabels(())
            plt.grid(True)
            plt.ylim((Xlim[0],Xlim[1]+(X.shape[-1]-1)*xspacing))
            if wi==0: # only left-most-plots
                plt.ylabel('X')
            plt.gca().set_yticklabels(())
            # group 'title'
            plt.text(.5,1,'{}{}'.format(ti,'*' if Yerr is not None and Yerr[ti]==False else ''), ha='center', va='top', fontweight='bold', transform=plt.gca().transAxes)

            # imagesc Y
            fig.add_subplot(gs[3,:], sharex=botax)
            plt.imshow(Y[ti,:,:].T, origin='upper', aspect='auto', cmap='gray', extent=[times[0],times[-1],0,Y.shape[-1]], interpolation=None)
            plt.gca().set_xticklabels(())
            if wi==0: # only left-most-plots
                plt.ylabel('Y')
            plt.gca().set_yticklabels(())

            # Fe (if given)
            if Fe is not None:
                fig.add_subplot(gs[4,:], sharex=botax)
                plt.plot(times,Fe[ti,:,:] + np.arange(Fe.shape[-1])[np.newaxis,:])
                plt.gca().set_xticklabels(())
                plt.grid(True)
                if wi==0: # only left-most-plots
                    plt.ylabel('Fe')
                plt.gca().set_yticklabels(())
                plt.ylim((Felim[0],Felim[1]+Fe.shape[-1]-1))

            # Fy
            if Py is None:
                plt.axes(botax) # no Py, Fy is bottom axis
            else:
                row = 4 if Fe is None else 5
                fig.add_subplot(gs[row,:], sharex=botax)
            plt.plot(times,Fy[ti,:,:], color='.5')
            plt.plot(times,Fy[ti,:,0],'k-')
            if hi==h-1: # only bottom plots
                plt.xlabel('time ({})'.format(xunit))
            else:
                plt.gca().set_xticklabels(())
            if wi==0: # only left most plots
                plt.ylabel("Fy")
            plt.grid(True)
            plt.gca().set_yticklabels(())
            plt.ylim(Fylim)

            # Py (if given)
            if Py is not None:
                plt.axes(botax)
                plt.plot(times[:Py.shape[-2]],Py[ti,:,:], color='.5')
                plt.plot(times[:Py.shape[-2]],Py[ti,:,0],'k-')
                if hi==h-1: # only bottom plots
                    plt.xlabel('time ({})'.format(xunit))
                else:
                    plt.gca().set_xticklabels(())
                if wi==0: # only left most plots
                    plt.ylabel("Py")
                plt.grid(True)
                plt.gca().set_yticklabels(())
                plt.ylim((0,1))

            ti=ti+1

    if label is not None:
        if Yerr is not None:
            plt.suptitle("{} {}/{} correct".format(label,sum(np.logical_not(Yerr)),len(Yerr)))
        else:
            plt.suptitle("{}".format(label))
    fig.set_tight_layout(True)
    plt.show(block=False)




def debug_test_single_dataset(dataset:str,filename:str=None,dataset_args=None, loader_args=None, *args,**kwargs):
    """run the debug_test_dataset for a single subject from dataset
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        Fy (nTrl,nSamp,nY): The output scores obtained by comping the stimulus-scores (Fe) with the stimulus information (Y)
        Fe ((nTrl,nSamp,nE), optional): The stimulus scores, for the different event types, obtained by combining X with the decoding model. Defaults to None.
        Py ((nTrl,nSamp,nY), optional): The target probabilities for each output, derived from Fy. Defaults to None.
        fs (float, optional): sample rate of X, Y, used to set the time-axis. Defaults to None.
        label (str, optional): A textual label for this dataset, used for titles & save-files. Defaults to None.
        centerx (bool, optional): Center (zero-mean over samples) X for plotting. Defaults to True.
        xspacing (int, optional): Gap in X units between different channel lines. Defaults to 10.
        sumFy (bool, optional): accumulate the output scores before plotting. Defaults to True.
        Yerr (bool (nTrl,), optional): indicator for which trials the model made a correct prediction. Defaults to None.
    """    
    times = np.arange(X.shape[1])
    if fs is not None:
        times = times/fs
        xunit='s'
    else:
        xunit='samp'

    if centerx:
        X = X.copy() - np.mean(X,1,keepdims=True)
    if xspacing is None: 
        xspacing=np.median(np.diff(X,axis=-2).ravel())

    if sumFy:
        Fy = np.cumsum(Fy,axis=-2)

    if Fe.ndim>3 and Fe.shape[0]==1:# strip model dim
        Fe = Fe[0,...]

    Xlim = (np.min(X[...,0].ravel()),np.max(X[...,-1].ravel()))

    Fylim = (np.min(Fy.ravel()),np.max(Fy.ravel()))
    if Fe is not None:
        Felim = (np.min(Fe.ravel()),np.max(Fe.ravel()))

    if Py is not None:
        if Py.ndim>3 :
            print("Py: Multiple models? accumulated away")
            Py = np.sum(Py,0)

    if Fy is not None:
        if Fy.ndim>3 :
            print("Fy: Multiple models? accumulated away")
            Fy = np.mean(Fy,0)

    nTrl = X.shape[0]; w = int(np.ceil(np.sqrt(nTrl)*1.8)); h = int(np.ceil(nTrl/w))
    fig=plt.gcf()
    if nTrl>3 : fig.set_size_inches(20,10,forward=True)
    trial_grid = fig.add_gridspec( nrows=h, ncols=w, figure=fig, hspace=.05, wspace=.05) # per-trial grid
    nrows= 5 + (0 if Fe is None else 1) + (0 if Py is None else 1)
    ti=0
    for hi in range(h):
        for wi in range(w):
            if ti>=X.shape[0]:
                break

            gs = trial_grid[ti].subgridspec( nrows=nrows, ncols=1, hspace=0 )

            # pre-make bottom plot
            botax = fig.add_subplot(gs[-1,0])

            # plot X (0-3)
            fig.add_subplot(gs[:3,:], sharex=botax)
            plt.plot(times,X[ti,:,:] + np.arange(X.shape[-1])*xspacing)
            plt.gca().set_xticklabels(())
            plt.grid(True)
            plt.ylim((Xlim[0],Xlim[1]+(X.shape[-1]-1)*xspacing))
            if wi==0: # only left-most-plots
                plt.ylabel('X')
            plt.gca().set_yticklabels(())
            # group 'title'
            plt.text(.5,1,'{}{}'.format(ti,'*' if Yerr is not None and Yerr[ti]==False else ''), ha='center', va='top', fontweight='bold', transform=plt.gca().transAxes)

            # imagesc Y
            fig.add_subplot(gs[3,:], sharex=botax)
            plt.imshow(Y[ti,:,:].T, origin='upper', aspect='auto', cmap='gray', extent=[times[0],times[-1],0,Y.shape[-1]], interpolation=None)
            plt.gca().set_xticklabels(())
            if wi==0: # only left-most-plots
                plt.ylabel('Y')
            plt.gca().set_yticklabels(())

            # Fe (if given)
            if Fe is not None:
                fig.add_subplot(gs[4,:], sharex=botax)
                plt.plot(times,Fe[ti,:,:] + np.arange(Fe.shape[-1])[np.newaxis,:])
                plt.gca().set_xticklabels(())
                plt.grid(True)
                if wi==0: # only left-most-plots
                    plt.ylabel('Fe')
                plt.gca().set_yticklabels(())
                try:
                    plt.ylim((Felim[0],Felim[1]+Fe.shape[-1]-1))
                except:
                    pass

            # Fy
            if Py is None:
                plt.axes(botax) # no Py, Fy is bottom axis
            else:
                row = 4 if Fe is None else 5
                fig.add_subplot(gs[row,:], sharex=botax)
            plt.plot(times,Fy[ti,:,:], color='.5')
            plt.plot(times,Fy[ti,:,0],'k-')
            if hi==h-1 and Py is None: # only bottom plots
                plt.xlabel('time ({})'.format(xunit))
            else:
                plt.gca().set_xticklabels(())
            if wi==0: # only left most plots
                plt.ylabel("Fy")
            plt.grid(True)
            plt.gca().set_yticklabels(())
            try:
                plt.ylim(Fylim)
            except:
                pass

            # Py (if given)
            if Py is not None:
                plt.axes(botax)
                plt.plot(times[:Py.shape[-2]],Py[ti,:,:], color='.5')
                plt.plot(times[:Py.shape[-2]],Py[ti,:,0],'k-')
                if hi==h-1: # only bottom plots
                    plt.xlabel('time ({})'.format(xunit))
                else:
                    plt.gca().set_xticklabels(())
                if wi==0: # only left most plots
                    plt.ylabel("Py")
                plt.grid(True)
                plt.gca().set_yticklabels(())
                plt.ylim((0,1))

            ti=ti+1

    if label is not None:
        if Yerr is not None:
            plt.suptitle("{} {}/{} correct".format(label,sum(np.logical_not(Yerr)),len(Yerr)))
        else:
            plt.suptitle("{}".format(label))
    fig.set_tight_layout(True)
    plt.show(block=block)

def plot_stimseq(Y_TSy,fs=None,show=None):
    if fs is not None:
        plt.plot(np.arange(Y_TSy.shape[1])/fs, Y_TSy[0,...]/np.max(Y_TSy)*.75+np.arange(Y_TSy.shape[-1])[np.newaxis,:],'.')
        plt.xlabel('time (s)')
        plt.ylabel('output + level')
    else:
        plt.plot(Y_TSy[0,...]/np.max(Y_TSy)*.75+np.arange(Y_TSy.shape[-1])[np.newaxis,:],'.')
        plt.xlabel('time (samp)')
        plt.ylabel('output + level')
    plt.grid(True)
    plt.title('Y_TSy')
    if show is not None: plt.show(block=show)


def print_hyperparam_summary(res):
    fn = res[0].get('filenames',None)
    s = "N={}\n".format(len(fn))
    if fn:
        s = s + "fn={}\n".format([f[-30:] if f else None for f in fn])
    for ri in res:
        s += "\n{}\n".format(ri['config'])
        s += print_decoding_curves(ri['decoding_curves'])
    return s


def cv_fit(clsfr, X, Y, cv, fit_params=dict(), verbose:int=0, cv_clsfr_only:bool=False, score_fn=None, **kwargs):
    """cross-validated fit a classifier and compute it's scores on the validation set

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        np.ndarray: Fy the predictions on the validation examples
    """
    if fit_params is None: fit_params=dict()

    # FAST PATH use the classifiers optimized CV_fit routine 
    if hasattr(clsfr,'cv_fit'):
        res = clsfr.cv_fit(X, Y, cv=cv, fit_params=fit_params, **kwargs)

    else:
        if cv_clsfr_only and hasattr(clsfr,'stages') and hasattr(clsfr.stages[-1][1],'cv_fit'):
            # BODGE: only cv the final stage
            Xpp, Ypp = clsfr.fit_modify(X, Y, until_stage=-1)
            res = clsfr.stages[-1][1].cv_fit(Xpp, Ypp, cv=cv, fit_params=fit_params, score_fn=score_fn, **kwargs)

        else:
            # manually run the folds and collect the results
            # setup the folding
            if cv == True:  cv = 5
            if isinstance(cv, int):
                if X.shape[0] == 1 or cv <= 1:
                    # single trial, train/test on all...
                    cv = [(slice(X.shape[0]), slice(X.shape[0]))] # N.B. use slice to preserve dims..
                else:
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))

            scores=[]
            for i, (train_idx, valid_idx) in enumerate(cv):
                if verbose > 0:
                    print(".", end='', flush=True)
                
                clsfr.fit(X[train_idx, ...].copy(), Y[train_idx, ...].copy(), **fit_params, **kwargs)

                # predict, forcing removal of copies of  tgt=0 so can score
                if X[valid_idx,...].size==0:
                    print("Warning: no-validation trials!!! using all data!")
                    valid_idx = slice(X.shape[0])
                if hasattr(clsfr,'stages'):
                    Xpp, Ypp = clsfr.modify(X[valid_idx,...],Y[valid_idx,...],until_stage=-1) # pre-process
                    if score_fn is not None:
                        score = score_fn(clsfr,Xpp,Ypp)
                    elif hasattr(clsfr.stages[-1][1],'score'):
                        score = clsfr.stages[-1][1].score(Xpp, Ypp) # score
                else:
                    if score_fn is not None:
                        score = score_fn(clsfr,X[valid_idx,...].copy(), Y[valid_idx,...].copy())
                    else:
                        score = clsfr.score(X[valid_idx, ...].copy(), Y[valid_idx, ...].copy())
                scores.append(score)
            res=dict(scores_cv=scores)

    return res


def cv_fit_predict(clsfr, X, Y, cv, fit_params=dict(), verbose:int=0, cv_clsfr_only:bool=False, **kwargs):
    """cross-validated fit a classifier and compute it's predictions on the validation set

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        np.ndarray: Fy the predictions on the validation examples
    """
    if fit_params is None: fit_params=dict()

    # FAST PATH use the classifiers optimized CV_fit routine 
    if hasattr(clsfr,'cv_fit'):
        res = clsfr.cv_fit(X, Y, cv=cv, fit_params=fit_params, **kwargs)
        Fy = res['estimator'] if not 'rawestimator' in res else res['rawestimator']

    else:
        if cv_clsfr_only and hasattr(clsfr,'stages') and  hasattr(clsfr.stages[-1][1],'cv_fit'):
            # BODGE: only cv the final stage
            Xpp, Ypp = clsfr.fit_modify(X, Y, until_stage=-1)
            res = clsfr.stages[-1][1].cv_fit(Xpp, Ypp, cv=cv, fit_params=fit_params, **kwargs)
            Fy = res['estimator'] if not 'rawestimator' in res else res['rawestimator']

        else:
            # manually run the folds and collect the results
            # setup the folding
            if cv == True:  cv = 5
            if isinstance(cv, int):
                if X.shape[0] == 1 or cv <= 1:
                    # single trial, train/test on all...
                    cv = [(slice(X.shape[0]), slice(X.shape[0]))] # N.B. use slice to preserve dims..
                else:
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))

            for i, (train_idx, valid_idx) in enumerate(cv):
                if verbose > 0:
                    print(".", end='', flush=True)
                
                clsfr.fit(X[train_idx, ...].copy(), Y[train_idx, ...].copy(), **fit_params, **kwargs)

                # predict, forcing removal of copies of  tgt=0 so can score
                if X[valid_idx,...].size==0:
                    print("Warning: no-validation trials!!! using all data!")
                    valid_idx = slice(X.shape[0])
                Fyi = clsfr.predict(X[valid_idx, ...].copy(), Y[valid_idx, ...].copy())

                if i==0: # reshape Fy to include the extra model dim
                    Fy = np.zeros((Y.shape[0],)+Fyi.shape[1:], dtype=X.dtype)       
                Fy[valid_idx,...]=Fyi

    return dict(estimator_cv=Fy)


def decoding_curve_cv(clsfr:BaseSequence2Sequence, X, Y, cv, fit_params=dict(), cv_clsfr_only:bool=False):
    """cross-validated fit a classifier and compute it's decoding curve and associated scores

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        [type]: [description]
    """
    if cv is not None:
        res = cv_fit_predict(clsfr, X, Y, cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
        Fy = res['estimator_cv'] if isinstance(res,dict) else res
    else: # predict only
        #print("Predict only!")
        Fy = clsfr.predict(X,Y)

    # BODGE: get the proba calibration info!
    try:
        est = clsfr.stages[-1][1] if hasattr(clsfr,'stages') else clsfr
        est.calibrate_proba(Fy)
        softmaxscale, startup_correction, sigma0, priorweight, nvirt_out=\
             (est.softmaxscale_, est.startup_correction, est.sigma0_, est.priorweight, est.nvirt_out)
    except:
        softmaxscale, startup_correction, sigma0, priorweight, nvirt_out = (3,None,None,0, 0)

    nvirt_out = 0

    # extract the predictions and compute the decoding curve and scores
    (dc) = decodingCurveSupervised(Fy, marginalizedecis=True, priorsigma=(sigma0,priorweight), softmaxscale=softmaxscale, nEpochCorrection=startup_correction, nvirt_out=nvirt_out, verb=-1)
    scores = score_decoding_curve(*dc)
    scores['decoding_curve']=dc
    scores['Fy']=Fy
    return scores

def set_params_decoding_curve_cv(clsfr:BaseSequence2Sequence, X, Y, cv, config:dict=dict(), fit_params:dict=dict(), cv_clsfr_only:bool=False, extra_config:dict=None):
    """set parameters on classifier and then cv-fit and compute it's decoding curve

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        config (dict): parameters to set on the estimator with set_params(**config)
        fit_params (dict): additional parameters to pass to cv_fit
        extra_config (dict): extra info about X/Y to store in this runs results

    Returns:
        [type]: [description]
    """    
    from sklearn import clone
    from copy import deepcopy
    clsfr = clone(clsfr) if cv is not None and cv is not False else deepcopy(clsfr)
    clsfr.set_params(**config)
    scores = decoding_curve_cv(clsfr, X, Y, cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
    #print("Scores: {}".format(scores['score']))
    scores['config']=config
    scores['clsfr']=clsfr
    if extra_config : 
        scores.update(extra_config)
    return scores

def collate_futures_results(futures:list, verb:int=1):
    """combine the results from `decoding_curve_gridsearchCV` into a single dict

    Args:
        futures (list-of-dicts or list-of-futures): the future results to combine

    Returns:
        [type]: [description]
    """    
    import concurrent.futures
    import time
    list_of_futures = hasattr(futures[0],'result')
    # collect the results as the jobs finish
    res={}
    # get iterator over the futures (or list of results)
    futures_iter = concurrent.futures.as_completed(futures) if list_of_futures else futures
    n_to_collect=len(futures)
    n_collected =0
    t0 = time.time()
    tlog = t0
    for future in futures_iter:
        n_collected = n_collected+1
        if verb>0 and time.time()-tlog > 3:
            elapsed=time.time()-t0
            done = n_collected / n_to_collect
            print("\r{:d}% {:d} / {:d} collected in {:4.1f}s  est {:4.1f}s total {:4.1f}s remaining".format(int(100*done), n_collected,n_to_collect,elapsed,elapsed/done,elapsed/done - elapsed))
            tlog = time.time()
        if list_of_futures:
            try:
                scores = future.result()
            except :
                print("fail in getting result!")
                import traceback
                traceback.print_exc()
                continue
        else: # list of dicts
            scores=future

        # merge into the set of all results
        # work with list of dicts as output
        scores = [scores] if not isinstance(scores,list) else scores
        for score in scores:
            if score is None: continue
            if res :
                for k,v in score.items(): 
                    res[k].append(v.tolist() if isinstance(v,np.ndarray) else v)
            else:
                res = {k:[v.tolist() if isinstance(v,np.ndarray) else v] for k,v in score.items()}
    return res

def get_results_per_config(res):
    # get the unique configurations and the rows for each
    configs = []
    configrows = []
    for rowi, cf in enumerate(res['config']):
        key = str(cf)
        #print("key={}".format(key))
        try:
            fi = configs.index(key)
            configrows[fi].append(rowi)
        except ValueError: # new config
            #print("newkey={}".format(key))
            configs.append(key)
            configrows.append([rowi])

    # for each config, make a new row with the average of this configs values
    configres = dict() # config results in dict with key str config
    for ci,(cf,rows) in enumerate(zip(configs,configrows)):
        configinfo = dict()
        for k,v in res.items():
            vs = [v[r] for r in rows]
            configinfo[k]=vs
        configres[str(cf)] = configinfo
    return configres, configs, configrows


<<<<<<< HEAD
def average_results_per_config(res):

    _, configs, configrows = get_results_per_config(res)
=======


def run_analysis():    
    #analyse_datasets("plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(30,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3))
    #"plos_one",loader_args=dict(fs_out=120,stopband=((0,3),(45,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(45,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:674
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,2),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:61
    #"plos_one",tau_ms=350,evtlabs=('re','fe'),rank=1 : ave-score=72  -- should be 83!!!
    # C: slightly larger freq range helps. rank doesn't.

    #analyse_datasets("lowlands",loader_args=dict(fs_out=60,stopband=((0,5),(25,-1))),
    #                  model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe')))#,badEpThresh=6))
    #"lowlands",clsfr_args=dict(tau_ms=550,evtlabs=('re','fe'),rank=1,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=56
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=56
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=51
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,3),(25,-1))): ave-score=53
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=42
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=45
    #analyse_datasets("lowlands",loader_args=dict(passband=None,stopband=((0,5),(25,-1))),clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1,badEpThresh=4)): ave-score=.47
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    # for each config, make a new row with the average of this configs values
    newres = { k:[] for k in res.keys() }
    for ci,(cf,rows) in enumerate(zip(configs,configrows)):
        for k,v in res.items():
            vs = [v[r] for r in rows]
            if k.lower() == 'decoding_curve': # BODGE: special reducer for decoding-curves
                newres[k].append( [np.nanmean(v,0) for v in flatten_decoding_curves(vs)] )
            else:
                try:
                    newres[k].append(np.nanmean(vs,0))
                except:
                    newres[k].append(vs)
        # ensure config is right
        newres['config'][ci] = cf
    return newres

<<<<<<< HEAD

def decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, X, Y, cv, n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, fit_params:dict=dict(), cv_clsfr_only:bool=False, verb:int=1):
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    parameter_grid = ParameterGrid(tuned_parameters)
    print("Submitting {} jobs:".format(len(parameter_grid)),end='')
    futures=[]
    t0 = time.time()
    if n_jobs==0 or n_jobs==1: # in main thread
        for ci,fit_config in enumerate(parameter_grid):
            print(".",end='')
            future = set_params_decoding_curve_cv(clsfr, X, Y, cv, config=fit_config, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
            futures.append(future)
        # collect the results as the jobs finish
        res= collate_futures_results(futures)

    else:
        # to job pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks".format(n_jobs))

            # loop over analysis settings
            for ci,fit_config in enumerate(parameter_grid):
                print(".",end='')
                future = executor.submit(set_params_decoding_curve_cv, clsfr, X, Y, cv, config=fit_config, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
                futures.append(future)
            print("submitted. in {}s".format(time.time()-t0))

            # collect the results as the jobs finish
            res= collate_futures_results(futures)

    print("Completed {} jobs in {} s".format(len(parameter_grid),time.time()-t0))
    return res

def load_and_decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, filename, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), loader_args:dict=dict(), shortfilename:str=None, **kwargs):
    """ load filename with given loader and then do gridsearch CV

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        list-of-dict: list of cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """
    if tuned_parameters is None : tuned_parameters=dict()
    if fit_params is None : fit_params=dict()
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    try:
        X, Y, coords = loader(filename,**loader_args)
    except:
        import traceback
        traceback.print_exc()
        print("Problem loading file: {}.    SKIPPED".format(filename))
        return []
    if shortfilename is None: shortfilename=filename
    
    # BODGE: manually add the meta-info to the pipeline
    #  TODO[]: Make so meta-info is added to X in the loader, and preserved over multi-threading
    try:
        fs = coords[1]['fs']
        ch_names = coords[2]['coords']
        clsfr.set_params(metainfoadder__info=dict(fs=fs, ch_names=ch_names))
    except:
        print("Warning -- cant add meta-info to clsfr pipeline!")

    # record the shortened filename as extra_config
    extra_config = { 'filename':shortfilename }
    if loader_args:
        extra_config['loader_args']=loader_args
=======
    # N.B. ram limits the  tau size...
    # analyse_datasets("brainsonfire",
    #                 loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=20))
    #"brainsonfire",loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5)) : score=.46
    #"brainsonfire",loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=10)) : score=.53

    #analyse_datasets("twofinger",
    #                 model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5), 
    #                 loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1))))
    #"twofinger",'cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5),loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1)))): ave-score=.78
    # "twofinger",tau_ms=600, offset_ms=-300, rank=5,subtriallen=10, stopband=((0,1),(25,-1)))): ave-score: .85
    # C: slight benefit from pre-movement data
    
    # Note: max tau=500 due to memory limitation
    #analyse_datasets("cocktail",
    #                 clsfr_args=dict(tau_ms=500, evtlabs=None, rank=5, rcond=1e-4, center=False),
    #                 loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1))))
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':60, 'subtriallen':15,'passband':(5,25)}) : .78
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':60, 'subtriallen':15,'passband':(1,25)}) : .765
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,25)}) : .765
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .77
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=8,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .818
    #analyse_datasets("cocktail",tau_ms=700,evtlabs=None,rank=8,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .826
    #analyse_datasets("cocktail",tau_ms=700,evtlabs=None,rank=16,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .854
    #analyse_datasets("cocktail",tau_ms=500, evtlabs=None, rank=15,fs_out=60, subtriallen=10, stopband=((0,1),(25,-1)) : ave-score:.80 (6-subtrials)
    # C: longer analysis window + higher rank is better.  Sample rate isn't too important

    #analyse_datasets("openBMI_ERP",clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5),loader_args=dict(fs_out=30,stopband=((0,1),(12,-1)),offset_ms=(-500,1000)))
    # "openBMI_ERP",tau_ms=700,evtlabs=('re'),rank=1,loader_args=dict(offset_ms=(-500,1000) Ave-score=0.758
    # "openBMI_ERP",tau_ms=700,evtlabs=('re','ntre'),rank=1,loader_args={'offset_ms':(-500,1000)}) Ave-score=0.822
    # "openBMI_ERP",tau_ms=700,evtlabs=('re','ntre'),rank=5,loader_args={'offset_ms':(-500,1000)}) Ave-score=0.894
    #"openBMI_ERP",clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5),loader_args=dict(offset_ms=(-500,1000))): Ave-score=0.894
    # C: large-window, tgt-vs-ntgt  + rank>1 : gives best fit?
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    # loop over analysis settings
    futures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for fit_config in ParameterGrid(tuned_parameters):
            if n_jobs==0 or n_jobs==1:
                future = set_params_decoding_curve_cv(clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            else:
                future = executor.submit(set_params_decoding_curve_cv, clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            futures.append(future)
    return futures


def datasets_decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, filenames, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), cv_clsfr_only:bool=False, loader_args:dict=dict(), job_per_file:bool=True):
    """[summary]

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        dict: cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """    
    import concurrent.futures
    import os.path
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    common_path = os.path.commonpath(filenames)

    futures = []
    t0 = time.time()
    tlog = t0
    if n_jobs > 1 and job_per_file: # each file in it's own job
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks, one-per-filename".format(n_jobs))
            print("Submitting {} jobs:",end='')
            for fi,fn in enumerate(filenames):
                print('.',end='')
                future = executor.submit(load_and_decoding_curve_GridSearchCV, clsfr, fn, loader, cv, n_jobs=1, tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
                futures.append(future)
            print("{} jobs submitted in {:4.0f}s. Waiting results.\n".format(len(filenames),time.time()-t0))
        
            # collect the results as the jobs finish
            res = collate_futures_results(futures)
    else: # each config in it's own job
        for fi,fn in enumerate(filenames):
            if time.time()-tlog > 3:
                print("\r{} of {}  in {}s".format(fi,len(filenames),time.time()-t0))
                tlog=time.time()
            future = load_and_decoding_curve_GridSearchCV(clsfr,fn,loader,cv,n_jobs=n_jobs,tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
            futures.append(future)
        # collect the results as the jobs finish
        res = collate_futures_results(futures)

    return res


def set_params_cv(clsfr:BaseSequence2Sequence, X, Y, cv, config:dict=dict(), score_fn=None,
                  fit_params:dict=dict(), cv_clsfr_only:bool=False, extra_config:dict=None):
    """set parameters on classifier and then cv-fit and compute it's decoding curve

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        config (dict): parameters to set on the estimator with set_params(**config)
        fit_params (dict): additional parameters to pass to cv_fit
        extra_config (dict): extra info about X/Y to store in this runs results

    Returns:
        [type]: [description]
    """
    from sklearn import clone
    from copy import deepcopy
    clsfr = clone(clsfr) if cv is not None and cv is not False else deepcopy(clsfr)
    if config is not None:
        clsfr.set_params(**config)
    scores = cv_fit(clsfr, X.copy(), Y.copy(), cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
    #print("Scores: {}".format(scores['scores_cv']))
    scores['config']=config
    scores['clsfr']=clsfr
    if extra_config : 
        scores.update(extra_config)
    return scores


def load_and_GridSearchCV(clsfr:BaseSequence2Sequence, filename, loader, cv, 
            n_jobs:int=1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), loader_args:dict=dict(), shortfilename:str=None, **kwargs):
    """ load filename with given loader and then do gridsearch CV

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        list-of-dict: list of cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """
    if tuned_parameters is None : tuned_parameters=dict()
    if fit_params is None : fit_params=dict()
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    try:
        X, Y, coords = loader(filename,**loader_args)
    except:
        import traceback
        traceback.print_exc()
        print("Problem loading file: {}.    SKIPPED".format(filename))
        return []
    if shortfilename is None: shortfilename=filename
    
<<<<<<< HEAD
    # BODGE: manually add the meta-info to the pipeline
    #  TODO[]: Make so meta-info is added to X in the loader, and preserved over multi-threading
    try:
        fs = coords[1]['fs']
        ch_names = coords[2]['coords']
        clsfr.set_params(metainfoadder__info=dict(fs=fs, ch_names=ch_names))
    except:
        print("Warning -- cant add meta-info to clsfr pipeline!")

    # record the shortened filename as extra_config
    extra_config = { 'filename':shortfilename }
    if loader_args:
        extra_config['loader_args']=loader_args
=======
    #analyse_datasets("p300_prn",loader_args=dict(fs_out=30,stopband=((0,1),(25,-1)),subtriallen=10),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5))
    #"p300_prn",model='cca',loader_args=dict(fs_out=30,stopband=((0,2),(12,-1)),subtriallen=10),clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=15)) : score=.43
    #"p300_prn",model='cca',loader_args=dict(fs_out=60,stopband=((0,2),(25,-1)),subtriallen=10),clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=15)) : score=.47

    #analyse_datasets("mTRF_audio", tau_ms=600, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)})
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 15)}) : score=.86
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=2, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 15)}) : score=.85
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score = .89
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 25)}) : score = .86
    #analyse_datasets("mTRF_audio", tau_ms=100, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score= .85
    #analyse_datasets("mTRF_audio", tau_ms=20, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score=.88
    #analyse_datasets("mTRF_audio", tau_ms=600, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score=.91
    
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':64, 'passband':(.5, 15)}) : score=.77
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':64, 'passband':(5, 25)}) : score=.77
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.78
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=2, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.76
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=1, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.69
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    futures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # loop over analysis settings
        for fit_config in ParameterGrid(tuned_parameters):
            if n_jobs>1: # multi-thread
                future = executor.submit(set_params_cv, clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            else:# single thread
                future = set_params_cv(clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            futures.append(future)
    return futures

<<<<<<< HEAD

def datasets_GridSearchCV(clsfr:BaseSequence2Sequence, filenames, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), cv_clsfr_only:bool=False, loader_args:dict=dict(), job_per_file:bool=True):
    """run a complete dataset with different parameter settings expanding the grid of tuned_parameters
=======
    #analyse_datasets("tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5))
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=10) : ave-score:51
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=3) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=3) : ave-score:52
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=10) : ave-score:49
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=10) : ave-score:50
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re'),rank=5) : ave-score:44
    # C: above chance for 8/9, low rank~3, slow response
    
    #analyse_datasets("tactile_PatientStudy",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5))
    #"tactile_PatientStudy",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5) : ave-score:44

    #analyse_datasets("ninapro_db2",loader_args=dict(stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=20, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),
    #                 model='cca',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=6))
    #"ninapro_db2",loader_args=dict(subtrllen=10, stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=40, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),model='cca',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=20)): ave-score=65 (but dont' believe it)
    #"ninapro_db2",loader_args=dict(subtrllen=10, stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=40, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),model='ridge',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=20)): ave-score=26 (but dont' believe it)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().
    """    
    import concurrent.futures
    import os.path
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    common_path = os.path.commonpath(filenames)

    futures = []
    t0 = time.time()
    tlog = t0

    if n_jobs > 1 and job_per_file: # each file in it's own job
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks, one-per-filename".format(n_jobs))
            print("Submitting {} jobs:",end='')
            for fi,fn in enumerate(filenames):
                print('.',end='')
                future = executor.submit(load_and_GridSearchCV, clsfr, fn, loader, cv, n_jobs=1, tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
                futures.append(future)
            print("{} jobs submitted in {:4.0f}s. Waiting results.\n".format(len(filenames),time.time()-t0))
        
            # collect the results as the jobs finish
            res = collate_futures_results(futures)
    else: # each config in it's own job
        for fi,fn in enumerate(filenames):
            if time.time()-tlog > 3:
                print("\r{} of {}  in {}s".format(fi,len(filenames),time.time()-t0))
                tlog=time.time()
            future = load_and_GridSearchCV(clsfr,fn,loader,cv,n_jobs=n_jobs,tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
            futures.append(future)

        # collect the results as the jobs finish
        res = collate_futures_results(futures)

    return res


from datetime import datetime
import json
import pickle

if __name__=="__main__":
    from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
    import glob
    import os

<<<<<<< HEAD
    #loader_args = dict(fs_out=100,filterband=(3,25,'bandpass'))#((0,3),(45,65)))#,))
    #loader_args = dict()
    loader_args = dict(filterband=((.5,45,'bandpass')), fs_out = 62.5)

    # Noisetag pipeline
=======
    #debug_test_single_dataset('p300_prn',dataset_args=dict(label='rc_5_flash'),
    #              loader_args=dict(fs_out=32,stopband=((0,1),(12,-1)),subtriallen=None),
    #              model='cca',tau_ms=750,evtlabs=('re','anyre'),rank=3,reg=.02)

    savefile = None
    savefile = '~/Desktop/mark/mindaffectBCI_*decoder_off.txt'
    savefile = '~/Desktop/mark/mindaffectBCI_*201020_1148.txt'
    #savefile = '~/Desktop/mark/mindaffectBCI_*201014*0940*.txt'
    savefile = "~/Downloads/mindaffectBCI*.txt"
    if savefile is None:
        savefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')
    
    # default to last log file if not given
    files = glob.glob(os.path.expanduser(savefile))
    savefile = max(files, key=os.path.getctime)

    X, Y, coords = load_mindaffectBCI(savefile, stopband=((45,65),(5.5,25,'bandpass')), fs_out=100)
    label = os.path.splitext(os.path.basename(savefile))[0]

    #cv=[(slice(0,10),slice(10,None))]
    test_idx = slice(10,None) # hold-out test set

    #analyse_dataset(X, Y, coords, tau_ms=400, evtlabs=('re','fe'), rank=1, model='cca', tuned_parameters=dict(rank=[1,2,3,5]))
    #analyse_dataset(X, Y, coords, tau_ms=450, evtlabs=('re','fe'), 
    #                model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=1, priorweight=200)#, tuned_parameters=dict(startup_correction=(0,10,100), priorweight=(0,50,500)))

    debug_test_dataset(X, Y, coords, tau_ms=450, evtlabs=('re','fe'), 
                      model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=1, priorweight=200)#, prediction_offsets=(-1,0,1))

    quit()

    # strip weird trials..
    # keep = np.ones((X.shape[0],),dtype=bool)
    # keep[10:20]=False
    # X = X[keep,...]
    # Y = Y[keep,...]
    # coords[0]['coords']=coords[0]['coords'][keep]

    # set of splits were we train on non-overlapping subsets of trnsize.
    if False:
        trnsize=10
        splits=[]
        for i in range(0,X.shape[0],trnsize):
            trn_ind=np.zeros((X.shape[0]), dtype=bool)
            trn_ind[slice(i,i+trnsize)]=True
            tst_ind= np.logical_not(trn_ind)
            splits.append( ( (trn_ind, tst_ind), ) ) # N.B. ensure list-of-lists-of-trn/tst-splits
        #splits=5
        # compute learning curves
        analyse_train_test(X,Y,coords, label='decoder-on. train-test split', splits=splits, tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca', ranks=(1,2,3,5) )

    else:
        debug_test_dataset(X, Y, coords, label=label, tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=100, priorweight=1e6)#, prediction_offsets=(-2,-1,0,1) )
        #debug_test_dataset(X, Y, coords, label=label, tau_ms=400, evtlabs=('re','fe'), rank=1, model='lr', ignore_unlabelled=True)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    # ask for a directory to lood all datafiles from
    exptdir = askloadsavefile(filetypes='dir')
    loader, filenames, exptdir = get_dataset('mindaffectBCI',exptdir=exptdir)
    print("Got {} files".format(len(filenames)))

    # other testing datasets
    # loader, filenames, exptdir = get_dataset('lowlands')

    # c-VEP pipeline
    fs_out = 60
    tau_ms = 850
    offset_ms = 0
    filterband = None
    temporal_basis = 'wf2,10'
    perlevelweight=True
    evtlabs=('re')
    pipeline=[
        'MetaInfoAdder',
        'BadChannelRemover',
        ['NoiseSubspaceDecorrelator', {'ch_names':['Fp1', 'Fp2'], 'noise_idx':'idOutliers2', 'filterband':(.5,8,'bandpass')}],
        ['ButterFilterAndResampler', {'filterband':filterband, 'fs_out':fs_out}],
        #['TargetEncoder',{'evtlabs':('re')}],
        #['MultiCCA:clsfr',{'tau_ms':950, "rank":1, "temporal_basis":"winfourier10"}],
        #['MultiCCACV:clsfr',{'tau_ms':450, "inner_cv_params":{"rank":(1,3,5),"reg":[(1e-1,1e-1),(1e-3,1e-3),(1e-4,1e-4)]} }],
        #['MultiCCACV:clsfr',{'tau_ms':450, "inner_cv_params":{"rank":(1,3,5)}, "temporal_basis":"winfourier10"}],
        #['FwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-4,1e-3,1e-2,1e-1,.2,.3,.5,.9]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-1,.1,.2,.3,.5]}}]
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-1,.1,.2,.3,.5,.8]}}] #"reg":.1}]#
        ["TimeShifter", {"timeshift_ms":offset_ms}],
        ['mindaffectBCI.decoder.model_fitting.MultiCCACV:clsfr', {'evtlabs':evtlabs, 'tau_ms':tau_ms, "rank":1, 'temporal_basis':temporal_basis}]
    ]
    tuned_parameters={'clsfr__tau_ms':[450,650,850]}
    fit_params=None
    #tuned_parameters = dict(clsfr__reg=[1e-4,1e-3,1e-2,1e-1,.3])
    ppp = make_preprocess_pipeline(pipeline)

    res = datasets_GridSearchCV(ppp, filenames[:5], loader, loader_args=loader_args,
                 cv=10, n_jobs=5,
                 job_per_file=True, tuned_parameters=tuned_parameters,
                 cv_clsfr_only=True, fit_params=fit_params)


    # WARNING: cv_clsfr_only is currently broken -- ends up not allowing a classifiers inner-cv to run!
    res = datasets_decoding_curve_GridSearchCV(ppp, filenames, loader, loader_args=loader_args,
                 cv=5, n_jobs=1, tuned_parameters=tuned_parameters, 
                 job_per_file=True, 
                 cv_clsfr_only=False, fit_params=fit_params)

    # plot the x-dataset curves per config:
    label = 'mark_cca'
    configresults, configs, configrows = get_results_per_config(res)
    for config,configres in configresults.items():
        print("\n\n{}\n".format(config))
        plt.figure()
        print(print_decoding_curves(configres['decoding_curve']))
        plot_decoding_curves(configres['decoding_curve'])#, labels=configres['config'])

    # save the per-config results to a json file..
    UNAME = datetime.now().strftime("%y%m%d_%H%M")
    with open('{}_{}'.format(label,UNAME),'wb') as f:
        pickle.dump(res,f)
        #json.dump(res,f)
    plt.show(block=False)

    # get the per-config summary
    res = average_results_per_config(res)
    # report the per-config summary
    for dc,conf in zip(res['decoding_curve'],res['config']):
        print("\n\n{}\n".format(conf))
        print(print_decoding_curve(*dc))
    plt.figure()
    plot_decoding_curves(res['decoding_curve'],labels=res['config'])
    plt.show(block=True)

    quit()

    # # EMG pipeline
    # dataset_args = dict(exptdir='~/Desktop/emg')
    # loader_args = dict(fs_out=100,filterband=((45,55),(95,105),(145,155),(195,205),(2,220,'bandpass')))#((0,3),(45,65)))#,))
    # pipeline = [ 
    #     ["MetaInfoAdder", {"info":{"fs":-1}}],
    #     ["AdaptiveSpatialWhitener:wht1", {"halflife_s":10}],
    #     #"SpatialWhitener:wht1",
    #     ["FFTfilter:filt1", {"filterbank":[[30,35,60,65,"hilbert"],[60,65,110,115,"hilbert"],[110,115,145,145,"hilbert"]], "blksz":100}],
    #     "Log",
    #     #["AdaptiveSpatialWhitener", {"halflife_s":10}],
    #     "SpatialWhitener:wht2",
    #     ["ButterFilterAndResampler:filt2", {"filterband":(8,-1), "fs_out":25}],
    #     ["TargetEncoder", { "evtlabs":"hoton"}],
    #     ["TimeShifter",{"timeshift_ms":0}],
    #     #"FeatureDimCompressor", # make back to 3d Trial,Sample,Features
    #     ["MultiCCA_cv", {"tau_ms":200, "rank":10, "center":True, "startup_correction":0}]
    # ]

    tuned_parameters=dict(#multicca__rank=(5,10), 
    #            multicca__tau_ms=[200, 300, 350, 450], 
    #            multicca__offset_ms=[0, 75, 100],
                        # filt1__filterbank=[[[30,35,60,65,"hilbert"],[60,65,110,115,"hilbert"],[110,115,145,145,"hilbert"]],
                        #                     (30,35,140,145,'hilbert'),
                        #                     (20,25,140,145,'hilbert'),
                        #                     (10,15,140,145,'hilbert'),
                        #                     (20,25,190,195,'hilbert'),
                        #                     (20,25,230,245,'hilbert'),
                        #                     (30,35,120,125,'hilbert'),
                        # ]
                        )


    #fit_params={"rank":(1,3,5),"reg":[(1e-1,1e-1),(1e-4,1e-4)]}

    # pipeline= [ 
    #         ["MetaInfoAdder", {"info":{"fs":-1}, "force":False}],  #N.B. NEED this for GridSearchCV!!
    #         #"SpatialWhitener:wht1",
    #         ["ButterFilterAndResampler:filt2", {"filterband":(45,125,'bandpass'), 'order':6}],
    #         "SpatialWhitener:wht2",
    #         #["FFTfilter:filt1", {"filterbank":[30,35,145,150,"bandpass"], "blksz":100, "squeeze_feature_dim":True}],
    #         ["BlockCovarianceMatrixizer", {"blksz_ms":200, "window":1, "overlap":.5}],
    #         ["ButterFilterAndResampler:filt2", {"filterband":(8,-1)}],
    #         ["TimeShifter", {"timeshift_ms":0}],
    #         ["TargetEncoder", {"evtlabs":"hoton"}],
    #         "DiagonalExtractor",
    #         #"Log",
    #         "FeatureDimCompressor", # make back to 3d Trial,Sample,Features
    #         ["MultiCCA", {"tau_ms":200, "rank":10, "center":True, "startup_correction":0}]
    #     ]

    # tuned_parameters=dict(
    #     diagonalextractor=[None,'skip'],
    #                     )

    ppp = make_preprocess_pipeline(pipeline)
    print(ppp)

    fit_params=dict() #dict(rank=(1,3,5),reg=[(x,y) for x in (1e-3,1e-1) for y in (1e-4,1e-2)])

    loader, filenames, _ = get_dataset(dataset,**dataset_args)
    res = datasets_decoding_curve_GridSearchCV(ppp, filenames, loader, loader_args={"fs_out":500},
                 cv=5, n_jobs=1, tuned_parameters=tuned_parameters, 
                 job_per_file=True, 
                 cv_clsfr_only=False, fit_params=fit_params)
    # get the per-config summary
    res = average_results_per_config(res)
    # report the per-config summary
    for dc,conf in zip(res['decoding_curve'],res['config']):
        print("\n\n{}\n".format(conf))
        print(print_decoding_curve(*dc))
    plt.figure()
    plot_decoding_curves(res['decoding_curve'],labels=res['config'])
    plt.show(block=True)