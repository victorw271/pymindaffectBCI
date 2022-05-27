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

import os
import signal
from multiprocessing import Process
<<<<<<< HEAD
import subprocess
from time import sleep
import traceback
from mindaffectBCI.config_file import load_config, set_args_from_dict, askloadconfigfile
from mindaffectBCI.decoder.decoder import UNAME

# setup signal handler to forward to child processes
signal.signal(signal.SIGINT, lambda signum, frame: shutdown())
signal.signal(signal.SIGTERM, lambda signum, frame: shutdown())

# global process holders
hub_process = None
acquisition_process = None
decoder_process = None

class NoneProc:
    """tempory class simulating a working null sub-process
    """
    exitcode = 0
=======
import subprocess 
from time import sleep
import json
import argparse
import traceback

class NoneProc:
    """tempory class simulating a working null sub-process
    """
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    def is_alive(self): return True
    def terminate(self): pass
    def join(self): pass

<<<<<<< HEAD
def startHubProcess(hub=None, label='online_bci', logdir=None):
=======
def startHubProcess(label='online_bci', logdir=None):
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    """Start the process to manage the central utopia-hub

    Args:
        label (str): a textual name for this process

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        hub (Process): sub-process for managing the started acquisition driver
    """    
<<<<<<< HEAD
    if hub is None or hub == 'utopia':
        from mindaffectBCI.decoder import startUtopiaHub
        hub = startUtopiaHub.run(label=label, logdir=logdir)

=======
    from mindaffectBCI.decoder import startUtopiaHub
    hub = startUtopiaHub.run(label=label, logdir=logdir)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    #hub = Process(target=startUtopiaHub.run, kwargs=dict(label=label), daemon=True)
    #hub.start()
    sleep(1)
    return hub

<<<<<<< HEAD
=======

>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
def startacquisitionProcess(acquisition, acq_args, label='online_bci', logdir=None):
    """Start the process to manage the acquisition of data from the amplifier

    Args:
        label (str): a textual name for this process
        acquisition (str): the name for the acquisition device to start.  One-of:
                  'none' - do nothing,  
                  'brainflow' - use the mindaffectBCI.examples.acquisition.utopia_brainflow driver
                  'fakedata'- start a fake-data streamer
                  'eego' - start the ANT-neuro eego driver
                  'lsl' - start the lsl EEG sync driver
        acq_args (dict): dictionary of additional arguments to pass to the acquisition device

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        Process: sub-process for managing the started acquisition driver
    """    
    # start the ganglion acquisition process
    # Using brainflow for the acquisition driver.  
    #  the brainflowargs are kwargs passed to BrainFlowInputParams
    #  so change the board_id and other args to use other boards
    if acquisition.lower() == 'none':
        # don't run acq driver here, user will start it manually
        acquisition = NoneProc()
    elif acquisition.lower() == 'fakedata':
        print('Starting fakedata')
        from mindaffectBCI.examples.acquisition import utopia_fakedata
<<<<<<< HEAD
        if acq_args is None:
            acq_args=dict(host='localhost', fs=200)
        acquisition = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
        acquisition.start()

=======
        acq_args=dict(host='localhost', nch=4, fs=200)
        acquisition = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
        acquisition.start()
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    elif acquisition.lower() == 'brainflow':
        from mindaffectBCI.examples.acquisition import utopia_brainflow
        if acq_args is None:
            acq_args = dict(board_id=1, serial_port='com3', log=1) # connect to the ganglion
        acquisition = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
        acquisition.start()

        # give it some time to startup successfully
        sleep(5)
    elif acquisition.lower() == 'ganglion': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisition import utopia_ganglion
        acquisition = Process(target=utopia_ganglion.run, kwargs=acq_args, daemon=True)
        acquisition.start()

<<<<<<< HEAD
    elif acquisition.lower() == 'cyton': # pyOpenBCI cyton driver
=======
    elif acquisition.lower() == 'cyton': # pyOpenBCI ganglion driver
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
        from mindaffectBCI.examples.acquisition import utopia_cyton
        acquisition = Process(target=utopia_cyton.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'javacyton': # java cyton driver
        from mindaffectBCI.examples.acquisition import startJavaCyton
        acquisition = Process(target=startJavaCyton.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'eego': # ANT-neuro EEGO
        from mindaffectBCI.examples.acquisition import utopia_eego
        acquisition = Process(target=utopia_eego.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'lsl': # lsl eeg input stream
        from mindaffectBCI.examples.acquisition import utopia_lsl
        acquisition = Process(target=utopia_lsl.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'brainproducts' or acquisition.lower()=='liveamp': # brainproducts eeg input stream
        from mindaffectBCI.examples.acquisition import utopia_brainproducts
        acquisition = Process(target=utopia_brainproducts.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'tmsi' : # tmsi porti
        from mindaffectBCI.examples.acquisition import utopia_tmsi
        acquisition = Process(target=utopia_tmsi.run, kwargs=acq_args, daemon=True)
        acquisition.start()

<<<<<<< HEAD
    elif acquisition.lower() == 'ft' : # fieldtrip buffer port
        from mindaffectBCI.examples.acquisition import utopia_ft
        acquisition = Process(target=utopia_ft.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition.lower() == 'saga' : # tmsi saga 
        from mindaffectBCI.examples.acquisition import utopia_saga
        acquisition = Process(target=utopia_saga.run, kwargs=acq_args, daemon=True)
        acquisition.start()  

    elif acquisition.lower() == 'cmd' : # command line  
        from mindaffectBCI.examples.acquisition import utopia_cmd
        # subproc in a procces.. needed to be compatible with rest of acq code
        acquisition = Process(target=utopia_cmd.run, kwargs=acq_args, daemon=True)
        acquisition.start() 
=======
    elif acquisition.lower() == 'ft' : # tmsi porti
        from mindaffectBCI.examples.acquisition import utopia_ft
        acquisition = Process(target=utopia_ft.run, kwargs=acq_args, daemon=True)
        acquisition.start()
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    else:
        raise ValueError("Unrecognised acquisition driver! {}".format(acquisition))
    
    return acquisition

def startDecoderProcess(decoder,decoder_args, label='online_bci', logdir=None):
    """start the EEG decoder process

    Args:
        label (str): a textual name for this process
        decoder (str): the name for the acquisition device to start.  One-of:
                  'decoder' - use the mindaffectBCI.decoder.decoder
                  'none' - don't start a decoder
        decoder_args (dict): dictionary of additional arguments to pass to the decoder
        logdir (str, optional): directory to save log/save files.

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        Process: sub-process for managing the started decoder
    """    
<<<<<<< HEAD
    target=None
=======
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    if decoder.lower() == 'decoder' or decoder.lower() == 'mindaffectBCI.decoder.decoder'.lower():
        from mindaffectBCI.decoder import decoder
        if decoder_args is None:
            decoder_args = dict(calplots=True)
        if not 'logdir' in decoder_args or decoder_args['logdir']==None: 
            decoder_args['logdir']=logdir
        print('Starting: {}'.format('mindaffectBCI.decoder.decoder'))
<<<<<<< HEAD
        target = decoder.run
        # allow time for the decoder to startup
        sleep(4)
    elif isinstance(decoder,str) and not decoder == 'none':
        try:
            import importlib
            dec = importlib.import_module(decoder)
            target = dec.run
        except:
            print("Error: could not run the decoder method")
            traceback.print_exc()
    elif decoder.lower() == 'none':
        pass

    if not target is None:
        decoder = Process(target=target, kwargs=decoder_args, daemon=True)
        decoder.start()
        return decoder
    else:
        return NoneProc()


def startPresentationProcess(presentation,presentation_args:dict=dict()):
    """start the presentation process, i.e. the process which presents the user-interface and stimuli to the user

    Args:
        presentation (_type_): the presentation process to start -- normally a string with a fully-qualified python class name to run
        presentation_args (dict, optional): arguments to pass to the presentation object at creation. Defaults to dict().

    Returns:
        _type_: _description_
    """    
    print("Attempting to start presentation: {}".format(presentation))
    target=None
    if presentation.lower() == 'selectionMatrix'.lower() or presentation.lower() == 'mindaffectBCI.presentation.selectionMatrix'.lower():
        if presentation_args is None:
            presentation_args = dict(symbols= [['Hello', 'Good bye'], 
                                               ['Yes',   'No']])
        from mindaffectBCI.presentation import selectionMatrix
        target = selectionMatrix.run

    elif presentation.lower() == 'sigviewer':
        print('starting sigviewer')
        import mindaffectBCI.decoder.sigViewer 
        target= mindaffectBCI.decoder.sigViewer.run
        print("target: {}".format(target))
=======
        decoder = Process(target=decoder.run, kwargs=decoder_args, daemon=True)
        decoder.start()
        # allow time for the decoder to startup
        sleep(4)
    elif decoder.lower() == 'none':
        decoder = NoneProc()
    return decoder

def startPresentationProcess(presentation,presentation_args):
    target=None
    if presentation.lower() == 'selectionMatrix'.lower() or presentation.lower() == 'mindaffectBCI.examples.presentation.selectionMatrix'.lower():
        if presentation_args is None:
            presentation_args = dict(symbols= [['Hello', 'Good bye'], 
                                               ['Yes',   'No']])
        from mindaffectBCI.examples.presentation import selectionMatrix
        target = selectionMatrix.run

    elif presentation.lower() == 'sigviewer':
        from mindaffectBCI.decoder.sigViewer import sigViewer
        target=sigViewer.run
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    elif presentation =='fakepresentation':
        import mindaffectBCI.noisetag
        target=mindaffectBCI.noisetag.run

<<<<<<< HEAD
=======
    elif presentation.lower() == 'hue' or presentation.lower() == "colorwheel":
        from mindaffectBCI.examples.presentation import colorwheel
        target=colorwheel.run

    elif presentation.lower() == 'rpigpio':
        from mindaffectBCI.examples.presentation import rpigpio
        target = rpigpio.run

>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    elif isinstance(presentation,str) and not presentation == 'none':
        try:
            import importlib
            pres = importlib.import_module(presentation)
            target = pres.run
        except:
            print("Error: could not run the presentation method")
            traceback.print_exc()
    
    elif presentation is None or presentation is False:
        print('No presentation specified.  Running in background!  Be sure to terminate with `mindaffectBCI.online_bci.shutdown()` or <ctrl-c>')
        return None

    if not target is None:
        presentation = Process(target=target, kwargs=presentation_args, daemon=True)
        presentation.start()
        return presentation
    else:
<<<<<<< HEAD
        return NoneProc()

def logConfiguration(args):
    """log the configuration of the system to the hub/savefile

    Args:
        args (dict): the arguments used to start the BCI
    """
    import json
    from mindaffectBCI.utopiaController import utopiaController
    try:
        uc = utopicController()
        uc.autoconnect()
        uc.log(json.dumps(dict(component='online_bci', args=args)))
        # uc.log(json.dumps(dict(component='hub',hub_args=hub_args))
        # uc.log(json.dumps(dict(component='acquisition',acquisition=acquisition,acq_args=acq_args)))
        # uc.log(json.dumps(dict(component='decoder',decoder=decoder,decoder_args=decoder_args)))
        # uc.log(json.dumps(dict(component='presentation',presentation=presentation,presentation_args=presentation_args)))
    except:
        print('Error: logging the configuraiton')
        traceback.print_exc()
    return
    
def run(label='', logdir=None, hub=None, args:dict=dict(),
        acquisition:str=None, acq_args:dict=dict(), 
        decoder:str='decoder', decoder_args:dict=dict(), 
        presentation:str='selectionMatrix', presentation_args:dict=dict()):
    """ Run the full on-line analysis stack with hub, acquisition, decoder and presentation
=======
        return None

def run(label='', logdir=None, block=True, acquisition=None, acq_args=None, decoder='decoder', decoder_args=None, presentation='selectionMatrix', presentation_args=None):
    """[summary]
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    Args:
        label (str, optional): string label for the saved data file. Defaults to ''.
        logdir (str, optional): directory to save log files / data files.  Defaults to None = $installdir$/logs.
        acquisition (str, optional): the name of the acquisition driver to use. Defaults to None.
        acq_args (dict, optional): dictionary of optoins to pass to the acquisition driver. Defaults to None.
        decoder (str, optional): the name of the decoder function to use.  Defaults to 'decoder'.
        decoder_args (dict, optional): dictinoary of options to pass to the mindaffectBCI.decoder.run(). Defaults to None.
        presentation (str, optional): the name of the presentation function to use.  Defaults to: 'selectionMatrix'
<<<<<<< HEAD
        presentation_args (dict, optional): dictionary of options to pass to mindaffectBCI.presentation.selectionMatrix.run(). Defaults to None.
=======
        presentation_args (dict, optional): dictionary of options to pass to mindaffectBCI.examples.presentation.selectionMatrix.run(). Defaults to None.
        block (bool, optional): return immeadiately or wait for presentation to finish and then terminate all processes.  Default to True
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    Raises:
        ValueError: invalid options, e.g. unrecognised acq_driver
    """    
    global hub_process, acquisition_process, decoder_process
    if acquisition is None: 
        acquisition = 'brainflow'

<<<<<<< HEAD
    # make the logs directory if not already there
    if logdir is None:
        logdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../logs')
    if label is not None:
        logdir=os.path.join(logdir,label)
    # add the session info
    logdir = os.path.join(logdir,UNAME)

=======
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    hub_process = None
    acquisition_process = None
    decoder_process = None
    for retries in range(10):
        #--------------------------- HUB ------------------------------
        # start the utopia-hub process
        if hub_process is None or not hub_process.poll() is None:
            try:
<<<<<<< HEAD
                hub_process = startHubProcess(hub=hub, label=label, logdir=logdir)
=======
                hub_process = startHubProcess(label=label, logdir=logdir)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
            except:
                hub_process = None
                traceback.print_exc()

        #---------------------------acquisition ------------------------------
        if acquisition_process is None or not acquisition_process.is_alive():
            try:
                acquisition_process = startacquisitionProcess(acquisition, acq_args, label=label, logdir=logdir)
            except:
                acquisition_process = None
                traceback.print_exc()

        #---------------------------DECODER ------------------------------
        # start the decoder process - with default settings for a noise-tag
        if decoder_process is None or not decoder_process.is_alive():
            try:
                decoder_process = startDecoderProcess(decoder, decoder_args, label=label, logdir=logdir)
            except:
                decoder_process = None
                traceback.print_exc()

        # terminate if all started successfully
        # check all started up and running..
        component_failed=False
        if hub_process is None or hub_process.poll() is not None:
            print("Hub didn't start correctly!")
            component_failed=True
        if acquisition_process is None or not acquisition_process.is_alive():
            print("Acq didn't start correctly!")
            component_failed=True
        if decoder_process is None or not decoder_process.is_alive():
            print("Decoder didn't start correctly!")
            component_failed=True

        # stop re-starting if all are running fine
        if not component_failed:
            break
        else:
            sleep(1)

    if hub_process is None or not hub_process.poll() is None:
        print("Hub didn't start correctly!")
        shutdown(hub_process,acquisition_process,decoder_process)
        raise ValueError("Hub didn't start correctly!")
    if acquisition_process is None or not acquisition_process.is_alive():
        print("Acq didn't start correctly!")
        shutdown(hub_process,acquisition_process,decoder_process)
        raise ValueError("acquisition didn't start correctly!")
    if decoder_process is None or not decoder_process.is_alive():
        shutdown(hub_process,acquisition_process,decoder_process)
        raise ValueError("Decoder didn't start correctly!")

    # log our configuration to the hub
    try:
        if args is None or len(args)==0:
            args=dict(label=label, logdic=logdir,
                      acquisition=acquisition,acquisition_args=acquisition_args,
                      decoder=decoder, decoder_args=decoder_args,
                      presentation=presentation, presentation_args=presentation_args)
            logConfiguration(args)
    except:
        pass
    
    #--------------------------- PRESENTATION ------------------------------
    # run the stimulus, in a background processwith our matrix and default parameters for a noise tag
    presentation_process = startPresentationProcess(presentation, presentation_args)

<<<<<<< HEAD
    # check all the sub-processes are running correctly.... abort on crash
    def get_subprocess_liveness():
        #nonlocal hub_process, acquisition_process, decoder_process, presentation_process
        return hub_process.poll() is None, acquisition_process.is_alive(), decoder_process.is_alive(), presentation_process.is_alive()
    # run while all sub-processes are alive
    while all(get_subprocess_liveness()):
        sleep(1)
=======
    if block == True:
        if presentation_process is not None:
            # wait for presentation to terminate
            presentation_process.join()
        else:
            hub_process.wait()
    else:
        return False
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    # TODO []: pop-up a monitoring object / dashboard!

    # get the exit codes for the sub-processes
    # check the reason we stopped... if something crashed then raise an error
    exitcodes = {"hub": hub_process.poll(),
                 "acq":acquisition_process.exitcode, 
                 "decoder":decoder_process.exitcode,
                 "presentation":presentation_process.exitcode}

    #--------------------------- SHUTDOWN ------------------------------
<<<<<<< HEAD
    # shutdown the background processes cleanly
    shutdown(hub_process, acquisition_process, decoder_process)

    # raise error if sub-process crashed
    # get the exit codes for the sub-processes
    # check the reason we stopped... if something crashed then raise an error
    if exitcodes['acq'] is not None and exitcodes['acq'] > 0 :
        raise ValueError("acquisition process crashed!")
    if exitcodes['decoder'] is not None and exitcodes['decoder'] > 0 :
        raise ValueError("Decoder process crashed!")
    if exitcodes['presentation'] is not None and exitcodes['presentation'] > 0:
        raise ValueError("Presentation process crashed!")


=======
    # shutdown the background processes
    shutdown(hub_process, acquisition_process, decoder_process)

>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

def check_is_running(hub=None, acquisition=None, decoder=None):
    """check if the background processes are still running

    Args:
        hub_process ([type], optional): the hub subprocess. Defaults to hub_process.
        acquisition_process ([type], optional): the acquisation subprocess. Defaults to acquisition_process.
        decoder_process ([type], optional): the decoder subprocess. Defaults to decoder_process.

    Returns:
        bool: true if all are running else false
    """
    # use module globals if not given?
    if hub is None: 
        global hub_process
        hub = hub_process
    if acquisition is None:
        global acquisition_process
        acquisition = acquisition_process
    if decoder is None:
        global decoder_process
        decoder = decoder_process

    isrunning=True
<<<<<<< HEAD
    if hub is None or hub.poll() is not None:
=======
    if hub is None or not hub.poll() is None:
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
        isrunning=False
        print("Hub is dead!")
    if acquisition is None or not acquisition.is_alive():
        print("Acq is dead!")
        isrunning=False
    if decoder is None or not decoder.is_alive():
        print("Decoder is dead!")
        isrunning=False
    return isrunning

def shutdown(hub=None, acquisition=None, decoder=None):    
    """shutdown any background processes started for the BCI

    Args:
        hub (subprocess, optional): handle to the hub subprocess object. Defaults to hub_process.
        acquisition (subprocess, optional): the acquisatin subprocess object. Defaults to acquisition_process.
        decoder (subprocess, optional): the decoder subprocess object. Defaults to decoder_process.
<<<<<<< HEAD
    """
    print("Shutting down!!")

    # decoder shutdown
    if decoder is None:
        global decoder_process
        decoder = decoder_process
=======
    """    
    # use module globals if not given?
    if hub is None: 
        global hub_process
        hub = hub_process
    if acquisition is None:
        global acquisition_process
        acquisition = acquisition_process
    if decoder is None:
        global decoder_process
        decoder = decoder_process

    hub.terminate()

>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    try: 
        decoder.terminate()
        decoder.join()
    except:
        pass

    # acquisition shutdown
    if acquisition is None:
        global acquisition_process
        acquisition = acquisition_process
    try:
<<<<<<< HEAD
        # acquisition.send_signal(signal.SIGTERM) # shutdown test for saga driver in subproc 
=======
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
        acquisition.terminate()
        acquisition.join()
    except:
        pass
<<<<<<< HEAD


    # BODGE[]: This is a really big hack to kill the hub--- it really really should not be necessary!
    # hub shutdown
    if hub is None: 
        global hub_process
        hub = hub_process
    import subprocess
    if os.name == 'nt': # hard kill
        subprocess.Popen("TASKKILL /F /IM java.exe".format(pid=hub_process.pid))
    else: # hard kill
        subprocess.Popen("killall java")
    import signal
    hub.send_signal(signal.SIGTERM)
    hub.terminate()
    print("Waiting for the hub to die!")
    hub.wait()
    hub.communicate()
    print("Hub is dead?")
    print("If not kill with:  taskkill /F /IM java.exe")

=======
    

    hub.wait()
#    if os.name == 'nt': # hard kill
#        subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=hub_process.pid))
#    else: # hard kill
#        os.kill(hub_process.pid, signal.SIGTERM)
    #print('exit online_bci')
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a


def load_config(config_file):
    """load an online-bci configuration from a json file

    Args:
        config_file ([str, file-like]): the file to load the configuration from. 
    """    
    from mindaffectBCI.decoder.utils import search_directories_for_file
    if isinstance(config_file,str):
        # search for the file in py-dir if not in CWD
        if not os.path.splitext(config_file)[1] == '.json':
            config_file = config_file + '.json'
        config_file = search_directories_for_file(config_file,os.path.dirname(os.path.abspath(__file__)))
        print("Loading config from: {}".format(config_file))
        with open(config_file,'r',encoding='utf8') as f:
            config = json.load(f)
    else:
        print("Loading config from: {}".format(f))
        config = json.load(f)

    # set the label from the config file
    if 'label' not in config or config['label'] is None:
        # get filename without path or ext
        config['label'] = os.path.splitext(os.path.basename(config_file))[0]
        
    return config


def parse_args():
    """ load settings from the json config-file, parse command line arguments, and merge the two sets of settings.

    Returns:
        NameSpace: the combined arguments name-space
    """    
    import argparse
    import json
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--label', type=str, help='user label for the data savefile. configfile name if None.', default=None)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default=None)#'debug')#'online_bci.json')
    parser.add_argument('--hub', type=str, help='the type of hub to run one-of: "none","ft"', default=None)
    parser.add_argument('--acquisition', type=str, help='set the acquisition driver type: one-of: "none","brainflow","fakedata","ganglion","eego"', default=None)
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisition system', default=dict())
=======
    parser.add_argument('--label', type=str, help='user label for the data savefile', default=None)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default=None)#'debug')#'online_bci.json')
    parser.add_argument('--acquisition', type=str, help='set the acquisition driver type: one-of: "none","brainflow","fakedata","ganglion","eego"', default=None)
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisition system', default=None)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    parser.add_argument('--decoder', type=str, help='set eeg decoder function to use. one-of: "none", "decoder"', default=None)
    parser.add_argument('--decoder_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the decoder. Note: need to doublequote the keywords!', default=dict())
    parser.add_argument('--presentation', type=str, help='set stimulus presentation function to use: one-of: "none","selectionMatrix"', default=None)
<<<<<<< HEAD
    parser.add_argument('--presentation_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the presentation system', default=dict())
=======
    parser.add_argument('--presentation_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the presentation system', default=None)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
    parser.add_argument('--logdir', type=str, help='directory where the BCI output files will be saved. Uses $installdir$/logs if None.', default=None)

    args = parser.parse_args()
    if args.config_file is None:
<<<<<<< HEAD
        config_file = askloadconfigfile()
        setattr(args,'config_file',config_file)

    # load config-file
    if args.config_file is not None:
        config = load_config(args.config_file)
        if 'acquisition_args' in config:
            config['acq_args']=config['acquisition_args']
        # MERGE the config-file parameters with the command-line overrides
        args = set_args_from_dict(args, config)

    if args.label is None and args.config_file:
        args.label = os.path.splitext(os.path.basename(args.config_file))[0]
=======
        try:
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            root = Tk()
            root.withdraw()
            filename = askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
                                        title='Chose mindaffectBCI Config File',
                                        filetypes=(('JSON','*.json'),('All','*.*')))
            setattr(args,'config_file',filename)
        except:
            print("Can't make file-chooser dialog, and no config file specified!  Aborting")
            raise ValueError("No config file specified")



    # load config-file
    if args.config_file is not None:
        config = load_config(args.config_file)

        # MERGE the config-file parameters with the command-line overrides
        for name in config: # config file parameter
            val = config[name]
            if name in args: # command line override available
                newval = getattr(args, name)
                if newval is None: # ignore not set
                    pass
                elif isinstance(val,dict): # dict, merge with config-file version
                    val.update(newval)
                else: # otherwise just override
                    val = newval
            setattr(args,name,val)
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a

    return args

# N.B. we need this guard for multiprocessing on Windows!
if __name__ == '__main__':
    args = parse_args()
<<<<<<< HEAD
    run(label=args.label, logdir=args.logdir, hub=args.hub, acquisition=args.acquisition, acq_args=args.acq_args, 
=======
    run(label=args.label, logdir=args.logdir, acquisition=args.acquisition, acq_args=args.acq_args, 
>>>>>>> 53e3633bc55dd13512738c132868bdd9a2fa713a
                          decoder=args.decoder, decoder_args=args.decoder_args, 
                          presentation=args.presentation, presentation_args=args.presentation_args)
