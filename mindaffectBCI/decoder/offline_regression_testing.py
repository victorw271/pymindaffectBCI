<<<<<<< HEAD
from joblib import PrintTime
=======
>>>>>>> ed6a7426202fb59d2c518e9aef6a1957ef06f690
import numpy as np
import matplotlib.pyplot as plt
# force random seed for reproducibility
seed=0
from mindaffectBCI.decoder.analyse_datasets import decoding_curve_GridSearchCV, datasets_decoding_curve_GridSearchCV, average_results_per_config, plot_decoding_curves
from mindaffectBCI.decoder.preprocess_transforms import make_preprocess_pipeline
from mindaffectBCI.decoder.offline.datasets import get_dataset
from mindaffectBCI.decoder.decodingCurveSupervised import print_decoding_curve
import random
random.seed(seed)
np.random.seed(seed)

import csv

def setup_plos_one():
    dataset = "plos_one"
    dataset_args = dict()

    # default pipeline -- filter for slow-drift and line-noise at load time so no startup artifacts
#    loader_args={'fs_out':60, 'filterband':((0,.5),(45,65),(95,105),(145,155),(195,205),(3,25,'bandpass'))}
    #loader_args={'fs_out':60, 'filterband':((0,3),(30,-1))}
    loader_args = dict(fs_out=100,filterband=((45,65),(3,25,'bandpass'))) 
    pipeline=[
        ['MetaInfoAdder',{'info':{'fs':-1}}],
        #['ButterFilterAndResampler',{'filterband':[(45,65),(3,25,'bandpass')], 'fs_out':100}],
        ['TargetEncoder',{'evtlabs':('re','fe')}],
        #['MultiCCACV:clsfr',{'tau_ms':450, 'offset_ms':0, "inner_cv_params":{"rank":(1,2,3,5),"reg":[(x,y) for x in (1e-8,1e-6,1e-4,1e-2,1e-1) for y in (1e-8,1e-6,1e-4,1e-2,1e-1)]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450, "inner_cv_params":{"reg":(1e-4, 1e-3, .01, .1, .5, 1-.1, 1-.01)}}]
        ['MultiCCA:clsfr',{'tau_ms':450, 'offset_ms':0}]
        #['MultiCCA2:clsfr',{'tau_ms':450, 'offset_ms':0}]
    ]

    cv=[(slice(10),slice(10,None))]

    return dataset, dataset_args, loader_args, pipeline, cv


def setup_kaggle():
    dataset="kaggle"
    dataset_args = dict()

    # default pipeline -- filter for slow-drift and line-noise at load time so no startup artifacts
    #loader_args={'fs_out':500, 'filterband':((0,.5),(45,65),(95,105),(145,155),(195,205))}
    loader_args={'fs_out':100, 'filterband':((45,65),(3,25,'bandpass'))}
    pipeline=[
        ['MetaInfoAdder',{'info':{'fs':-1}}],
        #['ButterFilterAndResampler',{'filterband':[(3,25,'bandpass')], 'fs_out':100}],
        ['TargetEncoder',{'evtlabs':('re','fe')}],
        #['MultiCCACV:clsfr',{'tau_ms':450, 'offset_ms':0, "inner_cv_params":{"rank":(1,2,3,5),"reg":[(x,y) for x in (1e-6,1e-2,1e-1) for y in (1e-6,1e-2,1e-1)]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450, "inner_cv_params":{"reg":(1e-4, 1e-3, .01, .1, .5, 1-.1, 1-.01)}}]
        ['MultiCCA:clsfr',{'tau_ms':450, 'offset_ms':0}]
        #['MultiCCA2:clsfr',{'tau_ms':450, 'offset_ms':0}]
    ]
    # run the search
    cv=[(slice(10),slice(10,None))]
    
    return dataset, dataset_args, loader_args, pipeline, cv
    

def setup_mindaffectBCI():
    dataset="mindaffectBCI" 
    dataset_args=dict(exptdir='~/Desktop/mark', regexp='.*noisetag.*')

    # default pipeline -- filter for slow-drift and line-noise at load time so no startup artifacts
    #loader_args={'fs_out':500, 'filterband':((0,.5),(45,65),(95,105),(145,155),(195,205))}
    loader_args={'fs_out':100, 'filterband':((45,65),(3,25,'bandpass'))}
    pipeline=[
        ['MetaInfoAdder',{'info':{'fs':-1}}],
        #['ButterFilterAndResampler',{'filterband':[(3,25,'bandpass')], 'fs_out':100}],
        ['TargetEncoder',{'evtlabs':('re','fe')}],
        #['MultiCCACV:clsfr',{'tau_ms':450, 'offset_ms':0, "inner_cv_params":{"rank":(1,2,3,5),"reg":[(x,y) for x in (1e-6,1e-2,1e-1) for y in (1e-6,1e-2,1e-1)]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450, "inner_cv_params":{"reg":(1e-4, 1e-3, .01, .1, .5, 1-.1, 1-.01)}}]
        ['MultiCCA:clsfr',{'tau_ms':450, 'offset_ms':0}]
        #['MultiCCA2:clsfr',{'tau_ms':450, 'offset_ms':0}]
    ]
    # run the search

    cv=[(slice(10),slice(10,None))]
    
    return dataset, dataset_args, loader_args, pipeline, cv



def setup_lowlands():
    dataset='lowlands'
    dataset_args=dict()

    # default pipeline
    # filter for slow-drift and line-noise at load time so no startup artifacts
    loader_args={'fs_out':100, 'filterband':((45,65),(3,25,'bandpass'))}
    pipeline=[
        ['MetaInfoAdder',{'info':{'fs':-1}}],
        #['ButterFilterAndResampler',{'filterband':[(3,25,'bandpass')], 'fs_out':100}],
        ['TargetEncoder',{'evtlabs':('re','fe')}],
        #['MultiCCACV:clsfr',{'tau_ms':450, 'offset_ms':0, "inner_cv_params":{"rank":(1,2,3,5),"reg":[(x,y) for x in (1e-8,1e-2,1e-1) for y in (1e-8,1e-2,1e-1)]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450, "inner_cv_params":{"reg":(1e-4, 1e-3, .01, .1, .5, 1-.1, 1-.01)}}]
        ['MultiCCA:clsfr',{'tau_ms':450, 'offset_ms':0}]
        #['MultiCCA2:clsfr',{'tau_ms':450, 'offset_ms':0}]
    ]

    cv=[(slice(10),slice(10,None))]
    
    return dataset, dataset_args, loader_args, pipeline, cv


















def pipeline_test(dataset:str, dataset_args:dict, loader_args:dict, pipeline, cv):

    loader, filenames, _ = get_dataset(dataset,**dataset_args)

    print("\n{} Files\n: {}".format(len(filenames),filenames))
<<<<<<< HEAD
=======

>>>>>>> ed6a7426202fb59d2c518e9aef6a1957ef06f690

    # first make the base pipeline to run
    clsfr = make_preprocess_pipeline(pipeline)
    print(clsfr)

    # run this pipeline with all the settings.
    # N.B. set n_jobs=1 for pipeline debugging as it gives more informative error messages and stops at first error
    res = datasets_decoding_curve_GridSearchCV(clsfr,filenames, loader, loader_args=loader_args, cv=cv, 
                                            n_jobs=5, cv_clsfr_only=False, label='taums')

    print("Ave-DC")
    print(print_decoding_curve(*(average_results_per_config(res)['decoding_curve'][0])))  

    plt.figure()
    plot_decoding_curves(res['decoding_curve'],labels=res['filename'])
    plt.show(block=False)

    # Print the output to a file to be shown in GitHub Actions
    # print("Per file")
    # for si in np.argsort(res['audc']):
    #     dc,conf = (res['decoding_curve'][si],res['config'][si])
    #     print("\n\n{} {}\n".format(conf, res['filename'][si]))
    #     print(print_decoding_curve(*dc))


    # Save to a CSV file
    # Find where the data came from
    name_file = 'no_data_found.csv'
    baseline_audc = 'not found'
    if ('mindaffectBCI' in filenames[0]):
        name_file = 'kaggle.csv'
        baseline_audc = '48.5'
    elif ('LL' in filenames[0]):
        name_file = 'lowlands.csv'
        baseline_audc = '35.2'
    elif ('data.mat' in filenames[0]):
        name_file = 'plos_one.csv'
        baseline_audc = '51.9'

    # Clean up string
    string_clsfr = str(clsfr).replace('\n', '')
    string_clsfr = string_clsfr.replace(',', ';')
    string_clsfr = string_clsfr.replace('  ', '')

    # Get data into csv file.
    s, ave_dc = print_decoding_curve(*(average_results_per_config(res)['decoding_curve'][0]))
    data_int = np.transpose(np.array([filenames, [string_clsfr]*len(filenames), [str("%.3f" % x) for x in res['audc']], [str("%.3f" % ave_dc)]*len(filenames), [51.9]*len(filenames)]))
    with open(name_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'clsfr', 'AUDC', 'ave-AUDC', 'baseline-AUDC'])
        writer.writerows(data_int)
    

    return res



def concurrent_analyse_datasets_test(dataset:str, dataset_args:dict, loader_args:dict, pipeline, cv):
    # fallback when pipeline isn't available
    from mindaffectBCI.decoder.analyse_datasets import concurrent_analyse_datasets
    clsfr_args = pipeline[-1][1]
    clsfr_args.update(evtlabs=('re','fe'))
    res = concurrent_analyse_datasets(dataset,model='cca',
                                dataset_args=dataset_args,
                                loader_args=loader_args,
                                clsfr_args=clsfr_args,
                                cv=cv)
    return res

def analyse_datasets_test(dataset:str, dataset_args:dict, loader_args:dict, pipeline, cv):
    print("Warning serial loader fallback!!!")
    from mindaffectBCI.decoder.analyse_datasets import analyse_datasets
    clsfr_args = pipeline[-1][1]
    clsfr_args.update(evtlabs=('re','fe'))
    res=analyse_datasets(dataset,model='cca',
                    dataset_args=dataset_args,
                    loader_args=loader_args,
                    clsfr_args=clsfr_args,
                    cv=cv)
    return res


def regression_test(dataset:str, dataset_args:dict, loader_args:dict, pipeline, cv):
    ''' run cross datasets test, with fallback for older non-supported code paths. '''
    try:
        res = pipeline_test(dataset,dataset_args,loader_args,pipeline,cv)
    except:
        res = None
    return res


















if __name__=="__main__":
    print('------------------\n\n K A G G L E\n\n---------------------')
    dataset, dataset_args,loader_args,pipeline,cv = setup_kaggle()
    regression_test(dataset, dataset_args, loader_args=loader_args, pipeline=pipeline, cv=cv)
    print("BASELINE: 336f594\n AVE-Dn\n\
                    IntLen   134   270   371   507   642   743   878  1014\n\
              Perr  0.94  0.76  0.56  0.36  0.27  0.25  0.23  0.23   AUDC 48.5\n\
         Perr(est)  0.92  0.58  0.37  0.27  0.22  0.21  0.19  0.19   PSAE 24.7\n\
           StopErr  0.96  0.93  0.89  0.81  0.69  0.60  0.56  0.56   AUSC 77.0\n\
     StopThresh(P)  0.86  0.82  0.77  0.63  0.49  0.44  0.44  0.44   SSAE 23.9")



    print('------------------\n\n P L O S    O N E\n\n---------------------')
    dataset, dataset_args, loader_args, pipeline, cv = setup_plos_one()
    regression_test(dataset, dataset_args, loader_args=loader_args, pipeline=pipeline, cv=cv)
    print('BASELINE: 336f594\n Ave-DC\n\
            IntLen   100   201   277   378   478   554   655   756\n\
              Perr  0.72  0.41  0.39  0.27  0.26  0.20  0.18  0.15   AUDC 35.2\n\
         Perr(est)  0.57  0.39  0.31  0.26  0.21  0.19  0.17  0.16   PSAE 10.6\n\
           StopErr  0.94  0.94  0.78  0.53  0.44  0.42  0.41  0.41   AUSC 63.1\n\
     StopThresh(P)  0.79  0.79  0.62  0.44  0.38  0.38  0.39  0.39   SSAE 15.6')


    print('------------------\n\n L O W L A N D S\n\n---------------------')
    dataset, dataset_args,loader_args,pipeline,cv = setup_lowlands()
    regression_test(dataset, dataset_args, loader_args=loader_args, pipeline=pipeline, cv=cv)
    print("BASELINE: 336f594\n Ave-DC\n\
            IntLen    54   109   150   205   260   301   356   411\n\
              Perr  0.79  0.62  0.52  0.46  0.41  0.39  0.36  0.36   AUDC 51.9\n\
         Perr(est)  0.55  0.43  0.36  0.32  0.28  0.26  0.24  0.24   PSAE 45.4\n\
           StopErr  0.98  0.98  0.98  0.98  0.79  0.72  0.69  0.69   AUSC 86.4\n\
     StopThresh(P)  0.89  0.89  0.89  0.89  0.63  0.59  0.60  0.61   SSAE 14.7")

