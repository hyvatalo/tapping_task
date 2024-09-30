#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Mon Sep 30 14:52:45 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from conditions_shuffle
import csv
import random
# Run 'Before Experiment' code from intertrial_dur
import random
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'tapping_task'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/ylha/Documents/Study 1/Tapping task/tapping_task.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('continue1') is None:
        # initialise continue1
        continue1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue1',
        )
    if deviceManager.getDevice('continue2') is None:
        # initialise continue2
        continue2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "set_conditions" ---
    # Run 'Begin Experiment' code from conditions_shuffle
    conditions = []
    with open('conditions_active1_sham2.csv', mode = 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            for i in row:
                conditions.append(i)
    
    random.shuffle(conditions)
    
            
    
    # --- Initialize components for Routine "get_ready" ---
    getready = visual.TextStim(win=win, name='getready',
        text='Get ready for the 12 taps task',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue1 = keyboard.Keyboard(deviceName='continue1')
    
    # --- Initialize components for Routine "task_12_taps" ---
    red_cross_500 = visual.ShapeStim(
        win=win, name='red_cross_500', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=0.1,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    green_cross_9500 = visual.ShapeStim(
        win=win, name='green_cross_9500', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 0.0039, -1.0000], fillColor=[-1.0000, 0.0039, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "intertrial_interval" ---
    red_intertrial = visual.ShapeStim(
        win=win, name='red_intertrial', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from intertrial_dur
    intertrial_intervals=[0.25,0.5,0.75]
    
    # --- Initialize components for Routine "metronome_task" ---
    metro_prepare = visual.TextStim(win=win, name='metro_prepare',
        text='Tap with the metronome when the cross turns green',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    red_cross_4150_1 = visual.ShapeStim(
        win=win, name='red_cross_4150_1', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    metro_tap_1 = visual.TextStim(win=win, name='metro_tap_1',
        text='Tap with the metronome',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    green_cross_2000_1 = visual.ShapeStim(
        win=win, name='green_cross_2000_1', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=0.1,
        colorSpace='rgb', lineColor=[-1.0000, 0.0039, -1.0000], fillColor=[-1.0000, 0.0039, -1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    stop_tapping = visual.TextStim(win=win, name='stop_tapping',
        text='Stop tapping with the metronome',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    red_listen_2400 = visual.ShapeStim(
        win=win, name='red_listen_2400', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=-5.0, interpolate=True)
    metro_tap_2 = visual.TextStim(win=win, name='metro_tap_2',
        text='Tap with the metronome',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    green_cross_2000_2 = visual.ShapeStim(
        win=win, name='green_cross_2000_2', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=0.1,
        colorSpace='rgb', lineColor=[-1.0000, 0.0039, -1.0000], fillColor=[-1.0000, 0.0039, -1.0000],
        opacity=None, depth=-7.0, interpolate=True)
    red_cross_4150_2 = visual.ShapeStim(
        win=win, name='red_cross_4150_2', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=None, depth=-8.0, interpolate=True)
    
    # --- Initialize components for Routine "Break" ---
    intermediate_break = visual.TextStim(win=win, name='intermediate_break',
        text='Please, take a break.',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue2 = keyboard.Keyboard(deviceName='continue2')
    
    # --- Initialize components for Routine "pop_condition" ---
    
    # --- Initialize components for Routine "thanks" ---
    thankyou = visual.TextStim(win=win, name='thankyou',
        text='Thank you for participating! ',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "set_conditions" ---
    # create an object to store info about Routine set_conditions
    set_conditions = data.Routine(
        name='set_conditions',
        components=[],
    )
    set_conditions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for set_conditions
    set_conditions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    set_conditions.tStart = globalClock.getTime(format='float')
    set_conditions.status = STARTED
    thisExp.addData('set_conditions.started', set_conditions.tStart)
    set_conditions.maxDuration = None
    # keep track of which components have finished
    set_conditionsComponents = set_conditions.components
    for thisComponent in set_conditions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "set_conditions" ---
    set_conditions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            set_conditions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in set_conditions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "set_conditions" ---
    for thisComponent in set_conditions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for set_conditions
    set_conditions.tStop = globalClock.getTime(format='float')
    set_conditions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('set_conditions.stopped', set_conditions.tStop)
    thisExp.nextEntry()
    # the Routine "set_conditions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=2.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "get_ready" ---
        # create an object to store info about Routine get_ready
        get_ready = data.Routine(
            name='get_ready',
            components=[getready, continue1],
        )
        get_ready.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for continue1
        continue1.keys = []
        continue1.rt = []
        _continue1_allKeys = []
        # Run 'Begin Routine' code from code_2
        condition=conditions[0]
        if "active" in condition:
            stim="active"
        if "sham" in condition:
            stim="sham"
            
        # store start times for get_ready
        get_ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        get_ready.tStart = globalClock.getTime(format='float')
        get_ready.status = STARTED
        thisExp.addData('get_ready.started', get_ready.tStart)
        get_ready.maxDuration = None
        # keep track of which components have finished
        get_readyComponents = get_ready.components
        for thisComponent in get_ready.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "get_ready" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        get_ready.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *getready* updates
            
            # if getready is starting this frame...
            if getready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                getready.frameNStart = frameN  # exact frame index
                getready.tStart = t  # local t and not account for scr refresh
                getready.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(getready, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'getready.started')
                # update status
                getready.status = STARTED
                getready.setAutoDraw(True)
            
            # if getready is active this frame...
            if getready.status == STARTED:
                # update params
                pass
            
            # if getready is stopping this frame...
            if getready.status == STARTED:
                if bool(0.0):
                    # keep track of stop time/frame for later
                    getready.tStop = t  # not accounting for scr refresh
                    getready.tStopRefresh = tThisFlipGlobal  # on global time
                    getready.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'getready.stopped')
                    # update status
                    getready.status = FINISHED
                    getready.setAutoDraw(False)
            
            # *continue1* updates
            waitOnFlip = False
            
            # if continue1 is starting this frame...
            if continue1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                continue1.frameNStart = frameN  # exact frame index
                continue1.tStart = t  # local t and not account for scr refresh
                continue1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(continue1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'continue1.started')
                # update status
                continue1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(continue1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(continue1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if continue1.status == STARTED and not waitOnFlip:
                theseKeys = continue1.getKeys(keyList=['1','enter','space'], ignoreKeys=["escape"], waitRelease=False)
                _continue1_allKeys.extend(theseKeys)
                if len(_continue1_allKeys):
                    continue1.keys = _continue1_allKeys[-1].name  # just the last key pressed
                    continue1.rt = _continue1_allKeys[-1].rt
                    continue1.duration = _continue1_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                get_ready.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in get_ready.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "get_ready" ---
        for thisComponent in get_ready.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for get_ready
        get_ready.tStop = globalClock.getTime(format='float')
        get_ready.tStopRefresh = tThisFlipGlobal
        thisExp.addData('get_ready.stopped', get_ready.tStop)
        # check responses
        if continue1.keys in ['', [], None]:  # No response was made
            continue1.keys = None
        trials_2.addData('continue1.keys',continue1.keys)
        if continue1.keys != None:  # we had a response
            trials_2.addData('continue1.rt', continue1.rt)
            trials_2.addData('continue1.duration', continue1.duration)
        # the Routine "get_ready" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=2.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "task_12_taps" ---
            # create an object to store info about Routine task_12_taps
            task_12_taps = data.Routine(
                name='task_12_taps',
                components=[red_cross_500, green_cross_9500],
            )
            task_12_taps.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for task_12_taps
            task_12_taps.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            task_12_taps.tStart = globalClock.getTime(format='float')
            task_12_taps.status = STARTED
            thisExp.addData('task_12_taps.started', task_12_taps.tStart)
            task_12_taps.maxDuration = None
            # keep track of which components have finished
            task_12_tapsComponents = task_12_taps.components
            for thisComponent in task_12_taps.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "task_12_taps" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            task_12_taps.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *red_cross_500* updates
                
                # if red_cross_500 is starting this frame...
                if red_cross_500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    red_cross_500.frameNStart = frameN  # exact frame index
                    red_cross_500.tStart = t  # local t and not account for scr refresh
                    red_cross_500.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(red_cross_500, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_cross_500.started')
                    # update status
                    red_cross_500.status = STARTED
                    red_cross_500.setAutoDraw(True)
                
                # if red_cross_500 is active this frame...
                if red_cross_500.status == STARTED:
                    # update params
                    pass
                
                # if red_cross_500 is stopping this frame...
                if red_cross_500.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > red_cross_500.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        red_cross_500.tStop = t  # not accounting for scr refresh
                        red_cross_500.tStopRefresh = tThisFlipGlobal  # on global time
                        red_cross_500.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'red_cross_500.stopped')
                        # update status
                        red_cross_500.status = FINISHED
                        red_cross_500.setAutoDraw(False)
                
                # *green_cross_9500* updates
                
                # if green_cross_9500 is starting this frame...
                if green_cross_9500.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    green_cross_9500.frameNStart = frameN  # exact frame index
                    green_cross_9500.tStart = t  # local t and not account for scr refresh
                    green_cross_9500.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(green_cross_9500, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_cross_9500.started')
                    # update status
                    green_cross_9500.status = STARTED
                    green_cross_9500.setAutoDraw(True)
                
                # if green_cross_9500 is active this frame...
                if green_cross_9500.status == STARTED:
                    # update params
                    pass
                
                # if green_cross_9500 is stopping this frame...
                if green_cross_9500.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > green_cross_9500.tStartRefresh + 9.5-frameTolerance:
                        # keep track of stop time/frame for later
                        green_cross_9500.tStop = t  # not accounting for scr refresh
                        green_cross_9500.tStopRefresh = tThisFlipGlobal  # on global time
                        green_cross_9500.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'green_cross_9500.stopped')
                        # update status
                        green_cross_9500.status = FINISHED
                        green_cross_9500.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    task_12_taps.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in task_12_taps.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "task_12_taps" ---
            for thisComponent in task_12_taps.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for task_12_taps
            task_12_taps.tStop = globalClock.getTime(format='float')
            task_12_taps.tStopRefresh = tThisFlipGlobal
            thisExp.addData('task_12_taps.stopped', task_12_taps.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if task_12_taps.maxDurationReached:
                routineTimer.addTime(-task_12_taps.maxDuration)
            elif task_12_taps.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            
            # --- Prepare to start Routine "intertrial_interval" ---
            # create an object to store info about Routine intertrial_interval
            intertrial_interval = data.Routine(
                name='intertrial_interval',
                components=[red_intertrial],
            )
            intertrial_interval.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from intertrial_dur
            trial=random.shuffle(intertrial_intervals)
            trial_duration=intertrial_intervals[0]
            # store start times for intertrial_interval
            intertrial_interval.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            intertrial_interval.tStart = globalClock.getTime(format='float')
            intertrial_interval.status = STARTED
            thisExp.addData('intertrial_interval.started', intertrial_interval.tStart)
            intertrial_interval.maxDuration = None
            # keep track of which components have finished
            intertrial_intervalComponents = intertrial_interval.components
            for thisComponent in intertrial_interval.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "intertrial_interval" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            intertrial_interval.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *red_intertrial* updates
                
                # if red_intertrial is starting this frame...
                if red_intertrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    red_intertrial.frameNStart = frameN  # exact frame index
                    red_intertrial.tStart = t  # local t and not account for scr refresh
                    red_intertrial.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(red_intertrial, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_intertrial.started')
                    # update status
                    red_intertrial.status = STARTED
                    red_intertrial.setAutoDraw(True)
                
                # if red_intertrial is active this frame...
                if red_intertrial.status == STARTED:
                    # update params
                    pass
                
                # if red_intertrial is stopping this frame...
                if red_intertrial.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > red_intertrial.tStartRefresh + trial_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        red_intertrial.tStop = t  # not accounting for scr refresh
                        red_intertrial.tStopRefresh = tThisFlipGlobal  # on global time
                        red_intertrial.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'red_intertrial.stopped')
                        # update status
                        red_intertrial.status = FINISHED
                        red_intertrial.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    intertrial_interval.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in intertrial_interval.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "intertrial_interval" ---
            for thisComponent in intertrial_interval.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for intertrial_interval
            intertrial_interval.tStop = globalClock.getTime(format='float')
            intertrial_interval.tStopRefresh = tThisFlipGlobal
            thisExp.addData('intertrial_interval.stopped', intertrial_interval.tStop)
            # the Routine "intertrial_interval" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 2.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "metronome_task" ---
        # create an object to store info about Routine metronome_task
        metronome_task = data.Routine(
            name='metronome_task',
            components=[metro_prepare, red_cross_4150_1, metro_tap_1, green_cross_2000_1, stop_tapping, red_listen_2400, metro_tap_2, green_cross_2000_2, red_cross_4150_2],
        )
        metronome_task.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for metronome_task
        metronome_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        metronome_task.tStart = globalClock.getTime(format='float')
        metronome_task.status = STARTED
        thisExp.addData('metronome_task.started', metronome_task.tStart)
        metronome_task.maxDuration = None
        # skip Routine metronome_task if its 'Skip if' condition is True
        metronome_task.skipped = continueRoutine and not (stim=="sham")
        continueRoutine = metronome_task.skipped
        # keep track of which components have finished
        metronome_taskComponents = metronome_task.components
        for thisComponent in metronome_task.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "metronome_task" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        metronome_task.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 25.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *metro_prepare* updates
            
            # if metro_prepare is starting this frame...
            if metro_prepare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                metro_prepare.frameNStart = frameN  # exact frame index
                metro_prepare.tStart = t  # local t and not account for scr refresh
                metro_prepare.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(metro_prepare, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'metro_prepare.started')
                # update status
                metro_prepare.status = STARTED
                metro_prepare.setAutoDraw(True)
            
            # if metro_prepare is active this frame...
            if metro_prepare.status == STARTED:
                # update params
                pass
            
            # if metro_prepare is stopping this frame...
            if metro_prepare.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > metro_prepare.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    metro_prepare.tStop = t  # not accounting for scr refresh
                    metro_prepare.tStopRefresh = tThisFlipGlobal  # on global time
                    metro_prepare.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'metro_prepare.stopped')
                    # update status
                    metro_prepare.status = FINISHED
                    metro_prepare.setAutoDraw(False)
            
            # *red_cross_4150_1* updates
            
            # if red_cross_4150_1 is starting this frame...
            if red_cross_4150_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                red_cross_4150_1.frameNStart = frameN  # exact frame index
                red_cross_4150_1.tStart = t  # local t and not account for scr refresh
                red_cross_4150_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_cross_4150_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_cross_4150_1.started')
                # update status
                red_cross_4150_1.status = STARTED
                red_cross_4150_1.setAutoDraw(True)
            
            # if red_cross_4150_1 is active this frame...
            if red_cross_4150_1.status == STARTED:
                # update params
                pass
            
            # if red_cross_4150_1 is stopping this frame...
            if red_cross_4150_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red_cross_4150_1.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    red_cross_4150_1.tStop = t  # not accounting for scr refresh
                    red_cross_4150_1.tStopRefresh = tThisFlipGlobal  # on global time
                    red_cross_4150_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_cross_4150_1.stopped')
                    # update status
                    red_cross_4150_1.status = FINISHED
                    red_cross_4150_1.setAutoDraw(False)
            
            # *metro_tap_1* updates
            
            # if metro_tap_1 is starting this frame...
            if metro_tap_1.status == NOT_STARTED and tThisFlip >= 05.0-frameTolerance:
                # keep track of start time/frame for later
                metro_tap_1.frameNStart = frameN  # exact frame index
                metro_tap_1.tStart = t  # local t and not account for scr refresh
                metro_tap_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(metro_tap_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'metro_tap_1.started')
                # update status
                metro_tap_1.status = STARTED
                metro_tap_1.setAutoDraw(True)
            
            # if metro_tap_1 is active this frame...
            if metro_tap_1.status == STARTED:
                # update params
                pass
            
            # if metro_tap_1 is stopping this frame...
            if metro_tap_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > metro_tap_1.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    metro_tap_1.tStop = t  # not accounting for scr refresh
                    metro_tap_1.tStopRefresh = tThisFlipGlobal  # on global time
                    metro_tap_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'metro_tap_1.stopped')
                    # update status
                    metro_tap_1.status = FINISHED
                    metro_tap_1.setAutoDraw(False)
            
            # *green_cross_2000_1* updates
            
            # if green_cross_2000_1 is starting this frame...
            if green_cross_2000_1.status == NOT_STARTED and tThisFlip >= 05.0-frameTolerance:
                # keep track of start time/frame for later
                green_cross_2000_1.frameNStart = frameN  # exact frame index
                green_cross_2000_1.tStart = t  # local t and not account for scr refresh
                green_cross_2000_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_cross_2000_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_cross_2000_1.started')
                # update status
                green_cross_2000_1.status = STARTED
                green_cross_2000_1.setAutoDraw(True)
            
            # if green_cross_2000_1 is active this frame...
            if green_cross_2000_1.status == STARTED:
                # update params
                pass
            
            # if green_cross_2000_1 is stopping this frame...
            if green_cross_2000_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > green_cross_2000_1.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    green_cross_2000_1.tStop = t  # not accounting for scr refresh
                    green_cross_2000_1.tStopRefresh = tThisFlipGlobal  # on global time
                    green_cross_2000_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_cross_2000_1.stopped')
                    # update status
                    green_cross_2000_1.status = FINISHED
                    green_cross_2000_1.setAutoDraw(False)
            
            # *stop_tapping* updates
            
            # if stop_tapping is starting this frame...
            if stop_tapping.status == NOT_STARTED and tThisFlip >= 010.0-frameTolerance:
                # keep track of start time/frame for later
                stop_tapping.frameNStart = frameN  # exact frame index
                stop_tapping.tStart = t  # local t and not account for scr refresh
                stop_tapping.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stop_tapping, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stop_tapping.started')
                # update status
                stop_tapping.status = STARTED
                stop_tapping.setAutoDraw(True)
            
            # if stop_tapping is active this frame...
            if stop_tapping.status == STARTED:
                # update params
                pass
            
            # if stop_tapping is stopping this frame...
            if stop_tapping.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stop_tapping.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    stop_tapping.tStop = t  # not accounting for scr refresh
                    stop_tapping.tStopRefresh = tThisFlipGlobal  # on global time
                    stop_tapping.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stop_tapping.stopped')
                    # update status
                    stop_tapping.status = FINISHED
                    stop_tapping.setAutoDraw(False)
            
            # *red_listen_2400* updates
            
            # if red_listen_2400 is starting this frame...
            if red_listen_2400.status == NOT_STARTED and tThisFlip >= 010.0-frameTolerance:
                # keep track of start time/frame for later
                red_listen_2400.frameNStart = frameN  # exact frame index
                red_listen_2400.tStart = t  # local t and not account for scr refresh
                red_listen_2400.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_listen_2400, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_listen_2400.started')
                # update status
                red_listen_2400.status = STARTED
                red_listen_2400.setAutoDraw(True)
            
            # if red_listen_2400 is active this frame...
            if red_listen_2400.status == STARTED:
                # update params
                pass
            
            # if red_listen_2400 is stopping this frame...
            if red_listen_2400.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red_listen_2400.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    red_listen_2400.tStop = t  # not accounting for scr refresh
                    red_listen_2400.tStopRefresh = tThisFlipGlobal  # on global time
                    red_listen_2400.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_listen_2400.stopped')
                    # update status
                    red_listen_2400.status = FINISHED
                    red_listen_2400.setAutoDraw(False)
            
            # *metro_tap_2* updates
            
            # if metro_tap_2 is starting this frame...
            if metro_tap_2.status == NOT_STARTED and tThisFlip >= 15.0-frameTolerance:
                # keep track of start time/frame for later
                metro_tap_2.frameNStart = frameN  # exact frame index
                metro_tap_2.tStart = t  # local t and not account for scr refresh
                metro_tap_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(metro_tap_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'metro_tap_2.started')
                # update status
                metro_tap_2.status = STARTED
                metro_tap_2.setAutoDraw(True)
            
            # if metro_tap_2 is active this frame...
            if metro_tap_2.status == STARTED:
                # update params
                pass
            
            # if metro_tap_2 is stopping this frame...
            if metro_tap_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > metro_tap_2.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    metro_tap_2.tStop = t  # not accounting for scr refresh
                    metro_tap_2.tStopRefresh = tThisFlipGlobal  # on global time
                    metro_tap_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'metro_tap_2.stopped')
                    # update status
                    metro_tap_2.status = FINISHED
                    metro_tap_2.setAutoDraw(False)
            
            # *green_cross_2000_2* updates
            
            # if green_cross_2000_2 is starting this frame...
            if green_cross_2000_2.status == NOT_STARTED and tThisFlip >= 015.0-frameTolerance:
                # keep track of start time/frame for later
                green_cross_2000_2.frameNStart = frameN  # exact frame index
                green_cross_2000_2.tStart = t  # local t and not account for scr refresh
                green_cross_2000_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_cross_2000_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_cross_2000_2.started')
                # update status
                green_cross_2000_2.status = STARTED
                green_cross_2000_2.setAutoDraw(True)
            
            # if green_cross_2000_2 is active this frame...
            if green_cross_2000_2.status == STARTED:
                # update params
                pass
            
            # if green_cross_2000_2 is stopping this frame...
            if green_cross_2000_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > green_cross_2000_2.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    green_cross_2000_2.tStop = t  # not accounting for scr refresh
                    green_cross_2000_2.tStopRefresh = tThisFlipGlobal  # on global time
                    green_cross_2000_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_cross_2000_2.stopped')
                    # update status
                    green_cross_2000_2.status = FINISHED
                    green_cross_2000_2.setAutoDraw(False)
            
            # *red_cross_4150_2* updates
            
            # if red_cross_4150_2 is starting this frame...
            if red_cross_4150_2.status == NOT_STARTED and tThisFlip >= 020.0-frameTolerance:
                # keep track of start time/frame for later
                red_cross_4150_2.frameNStart = frameN  # exact frame index
                red_cross_4150_2.tStart = t  # local t and not account for scr refresh
                red_cross_4150_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_cross_4150_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_cross_4150_2.started')
                # update status
                red_cross_4150_2.status = STARTED
                red_cross_4150_2.setAutoDraw(True)
            
            # if red_cross_4150_2 is active this frame...
            if red_cross_4150_2.status == STARTED:
                # update params
                pass
            
            # if red_cross_4150_2 is stopping this frame...
            if red_cross_4150_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red_cross_4150_2.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    red_cross_4150_2.tStop = t  # not accounting for scr refresh
                    red_cross_4150_2.tStopRefresh = tThisFlipGlobal  # on global time
                    red_cross_4150_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_cross_4150_2.stopped')
                    # update status
                    red_cross_4150_2.status = FINISHED
                    red_cross_4150_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                metronome_task.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in metronome_task.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "metronome_task" ---
        for thisComponent in metronome_task.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for metronome_task
        metronome_task.tStop = globalClock.getTime(format='float')
        metronome_task.tStopRefresh = tThisFlipGlobal
        thisExp.addData('metronome_task.stopped', metronome_task.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if metronome_task.maxDurationReached:
            routineTimer.addTime(-metronome_task.maxDuration)
        elif metronome_task.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-25.000000)
        
        # --- Prepare to start Routine "Break" ---
        # create an object to store info about Routine Break
        Break = data.Routine(
            name='Break',
            components=[intermediate_break, continue2],
        )
        Break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for continue2
        continue2.keys = []
        continue2.rt = []
        _continue2_allKeys = []
        # store start times for Break
        Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Break.tStart = globalClock.getTime(format='float')
        Break.status = STARTED
        thisExp.addData('Break.started', Break.tStart)
        Break.maxDuration = None
        # skip Routine Break if its 'Skip if' condition is True
        Break.skipped = continueRoutine and not (stim=="active")
        continueRoutine = Break.skipped
        # keep track of which components have finished
        BreakComponents = Break.components
        for thisComponent in Break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Break" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        Break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *intermediate_break* updates
            
            # if intermediate_break is starting this frame...
            if intermediate_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                intermediate_break.frameNStart = frameN  # exact frame index
                intermediate_break.tStart = t  # local t and not account for scr refresh
                intermediate_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(intermediate_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'intermediate_break.started')
                # update status
                intermediate_break.status = STARTED
                intermediate_break.setAutoDraw(True)
            
            # if intermediate_break is active this frame...
            if intermediate_break.status == STARTED:
                # update params
                pass
            
            # if intermediate_break is stopping this frame...
            if intermediate_break.status == STARTED:
                if bool(0.0):
                    # keep track of stop time/frame for later
                    intermediate_break.tStop = t  # not accounting for scr refresh
                    intermediate_break.tStopRefresh = tThisFlipGlobal  # on global time
                    intermediate_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'intermediate_break.stopped')
                    # update status
                    intermediate_break.status = FINISHED
                    intermediate_break.setAutoDraw(False)
            
            # *continue2* updates
            waitOnFlip = False
            
            # if continue2 is starting this frame...
            if continue2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                continue2.frameNStart = frameN  # exact frame index
                continue2.tStart = t  # local t and not account for scr refresh
                continue2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(continue2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'continue2.started')
                # update status
                continue2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(continue2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(continue2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if continue2.status == STARTED and not waitOnFlip:
                theseKeys = continue2.getKeys(keyList=['1','enter','space'], ignoreKeys=["escape"], waitRelease=False)
                _continue2_allKeys.extend(theseKeys)
                if len(_continue2_allKeys):
                    continue2.keys = _continue2_allKeys[-1].name  # just the last key pressed
                    continue2.rt = _continue2_allKeys[-1].rt
                    continue2.duration = _continue2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Break" ---
        for thisComponent in Break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Break
        Break.tStop = globalClock.getTime(format='float')
        Break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Break.stopped', Break.tStop)
        # check responses
        if continue2.keys in ['', [], None]:  # No response was made
            continue2.keys = None
        trials_2.addData('continue2.keys',continue2.keys)
        if continue2.keys != None:  # we had a response
            trials_2.addData('continue2.rt', continue2.rt)
            trials_2.addData('continue2.duration', continue2.duration)
        # the Routine "Break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pop_condition" ---
        # create an object to store info about Routine pop_condition
        pop_condition = data.Routine(
            name='pop_condition',
            components=[],
        )
        pop_condition.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        conditions.pop(0)
        
        # store start times for pop_condition
        pop_condition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pop_condition.tStart = globalClock.getTime(format='float')
        pop_condition.status = STARTED
        thisExp.addData('pop_condition.started', pop_condition.tStart)
        pop_condition.maxDuration = None
        # keep track of which components have finished
        pop_conditionComponents = pop_condition.components
        for thisComponent in pop_condition.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pop_condition" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        pop_condition.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pop_condition.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pop_condition.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pop_condition" ---
        for thisComponent in pop_condition.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pop_condition
        pop_condition.tStop = globalClock.getTime(format='float')
        pop_condition.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pop_condition.stopped', pop_condition.tStop)
        # the Routine "pop_condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[thankyou],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thankyou* updates
        
        # if thankyou is starting this frame...
        if thankyou.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankyou.frameNStart = frameN  # exact frame index
            thankyou.tStart = t  # local t and not account for scr refresh
            thankyou.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thankyou, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thankyou.started')
            # update status
            thankyou.status = STARTED
            thankyou.setAutoDraw(True)
        
        # if thankyou is active this frame...
        if thankyou.status == STARTED:
            # update params
            pass
        
        # if thankyou is stopping this frame...
        if thankyou.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thankyou.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                thankyou.tStop = t  # not accounting for scr refresh
                thankyou.tStopRefresh = tThisFlipGlobal  # on global time
                thankyou.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thankyou.stopped')
                # update status
                thankyou.status = FINISHED
                thankyou.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks.maxDurationReached:
        routineTimer.addTime(-thanks.maxDuration)
    elif thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
