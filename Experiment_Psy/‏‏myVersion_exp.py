#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on January 09, 2024, at 20:20
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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

# Run 'Before Experiment' code from code
from constant import *
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'myVersion'  # from the Builder filename that created this script
expInfo = {
    'participant': '001',
    'group': '3',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
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
        originPath='C:\\Users\\אחינעם רוזנצויג\\Documents\\פרוייקט 2023\\Psy\\Experiment_Psy\\\u200f\u200fmyVersion_exp.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 720], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='dkl',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='cm'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'dkl'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'cm'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
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
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
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
    inputs : dict
        Dictionary of input devices by name.
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
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
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
    
    # --- Initialize components for Routine "file_selection" ---
    # Run 'Begin Experiment' code from code_3
    from time import sleep
    
    group = expInfo["group"]
    
    ###chage to the right link before use
    if group == '1':
        conditions_file="C:/Users/אחינעם רוזנצויג/Documents/פרוייקט 2023/Psy/Experiment_Psy/conditions_generalization.xlsx"
    elif group == '2':
        conditions_file="C:/Users/אחינעם רוזנצויג/Documents/פרוייקט 2023/Psy/Experiment_Psy/conditions_feedback.xlsx"
    elif group == '3':
        conditions_file="C:/Users/אחינעם רוזנצויג/Documents/פרוייקט 2023/Psy/Experiment_Psy/conditions_preparation.xlsx"
    elif group == '0':
        conditions_file="C:/Users/אחינעם רוזנצויג/Documents/פרוייקט 2023/Psy/Experiment_Psy/conditions_test.xlsx"
    
    
    # --- Initialize components for Routine "instruction_task" ---
    # Run 'Begin Experiment' code from code_4
    from psychopy import sound, core, visual, event
    instructions_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instructions_mouse.mouseClock = core.Clock()
    instructions = visual.ImageStim(
        win=win,
        name='instructions', 
        image='ins/Instructions_baseline.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(15, 15),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "start_trial_ring" ---
    # Run 'Begin Experiment' code from code
    def euclidean_dist(vec1, vec2):
        return sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2)
    
    
    ring_mouse = event.Mouse(win=win)
    x, y = [None, None]
    ring_mouse.mouseClock = core.Clock()
    Ring = visual.Polygon(
        win=win, name='Ring',
        edges=50, size=[1.0, 1.0],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1.0, depth=-2.0, interpolate=True)
    fixation_ring = visual.Polygon(
        win=win, name='fixation_ring',
        edges=36, size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1.0, depth=-3.0, interpolate=True)
    mouse_circle_ring = visual.Polygon(
        win=win, name='mouse_circle_ring',
        edges=36, size=(0.3, 0.3),
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0,1,0], fillColor=[0,1,0],
        opacity=0.0, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "origin_hold1" ---
    # Run 'Begin Experiment' code from origin_hold
    import random
    
    
    baseline_mouse = event.Mouse(win=win)
    x, y = [None, None]
    baseline_mouse.mouseClock = core.Clock()
    baseline_fixation = visual.Polygon(
        win=win, name='baseline_fixation',
        edges=36, size=(0.5, 0.5),
        ori=0, pos=(0, 0), anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-2.0, interpolate=True)
    baseline_target_circle = visual.Polygon(
        win=win, name='baseline_target_circle',
        edges=50, size=(0.4, 0.4),
        ori=45, pos=[0,0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[-1,-1,-1], fillColor=[-1,-1,-1],
        opacity=0, depth=-3.0, interpolate=True)
    mouse_circle = visual.Polygon(
        win=win, name='mouse_circle',
        edges=40, size=(0.3, 0.3),
        ori=0, pos=[0,0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[0,1,0], fillColor=[0,1,0],
        opacity=1, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "move" ---
    # Run 'Begin Experiment' code from rotated_moving
    import pandas as pd
    import math
    from psychopy.tools.coordinatetools import cart2pol
    from psychopy.tools.coordinatetools import pol2cart
    ###############
    '''
    from psychopy import core, event
    
    routineClock = core.Clock()
    # Set up variables
    beep_duration = 0.05  # 50 ms
    beep_interval = 0.25  # 250 ms
    target_duration = 0.3  # 300 ms before the last beep
    num_beeps = 4
    beep_times = [i * beep_interval for i in range(4)]  # Scheduled times for beeps
    current_beep = 0  # Index to track the current beep
    beep_start_time = -1
    
    
    # Function to make the target visible
    def show_target():
        rotated_target_circle_2.opacity = 1
    
    # Function to check if the mouse is at the origin
    def is_mouse_at_origin():
        return all(coord == 0 for coord in mouse.getPos())
    
    '''
    # Set up the clock
    clock = core.Clock()
    ################
    
    # Load the Excel file into a DataFrame
    df = pd.read_excel(conditions_file)
    
    # Initialize other variables
    current_row = 0
    
    import math
    
    def rotate_vector(dx, dy, radians):
        """Rotate a vector by a given angle in radians."""
        new_dx = dx * math.cos(radians) - dy * math.sin(radians)
        new_dy = dx * math.sin(radians) + dy * math.cos(radians)
        return new_dx, new_dy
    
    last_mouse_x, last_mouse_y = 0, 0
    point_x, point_y = 0, 0
    
    rotated_move_mouse = event.Mouse(win=win)
    x, y = [None, None]
    rotated_move_mouse.mouseClock = core.Clock()
    rotated_fixation_2 = visual.Polygon(
        win=win, name='rotated_fixation_2',
        edges=36, size=(0.5, 0.5),
        ori=0, pos=(0, 0), anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-2.0, interpolate=True)
    rotated_target_circle_2 = visual.Polygon(
        win=win, name='rotated_target_circle_2',
        edges=50, size=(0.5, 0.5),
        ori=45, pos=[0,0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[-1,-1,-1], fillColor=[-1,-1,-1],
        opacity=1, depth=-3.0, interpolate=True)
    mouse_circle_2 = visual.Polygon(
        win=win, name='mouse_circle_2',
        edges=40, size=(0.3, 0.3),
        ori=0, pos=[0,0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[0,1,0], fillColor=[0,1,0],
        opacity=1, depth=-4.0, interpolate=True)
    jumping_target = visual.Polygon(
        win=win, name='jumping_target',
        edges=50, size=(0.5, 0.5),
        ori=45, pos=[0,0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[0,1,0], fillColor=[0,1,0],
        opacity=1, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from rotated_feedback_code
    # Load the Excel file into a DataFrame
    df = pd.read_excel(conditions_file)
    #def euclidean_dist(vec1, vec2):
    #    return sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2)
    
    
    rotated_feedback_mouse = event.Mouse(win=win)
    x, y = [None, None]
    rotated_feedback_mouse.mouseClock = core.Clock()
    rotated_fixation2 = visual.Polygon(
        win=win, name='rotated_fixation2',
        edges=36, size=(0.5, 0.5),
        ori=0, pos=(0, 0), anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-2.0, interpolate=True)
    rotated_target_circle2 = visual.Polygon(
        win=win, name='rotated_target_circle2',
        edges=50, size=(0.5, 0.5),
        ori=45, pos=[0,0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[-1,-1,-1], fillColor=[-1,-1,-1],
        opacity=1, depth=-3.0, interpolate=True)
    rotated_mouse_circle = visual.Polygon(
        win=win, name='rotated_mouse_circle',
        edges=50, size=(0.3, 0.3),
        ori=0, pos=[0,0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[0,1,0], fillColor=[0,1,0],
        opacity=0, depth=-4.0, interpolate=True)
    jumping_target_2 = visual.Polygon(
        win=win, name='jumping_target_2',
        edges=50, size=(1,1),
        ori=45, pos=[0,0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[-1,0,0], fillColor=[-1,0,0],
        opacity=1, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "instruction" ---
    # Run 'Begin Experiment' code from code_2
    import pandas as pd
    
    # Load the Excel file
    df = pd.read_excel(conditions_file)
    
    # Create an index to keep track of your current row in the dataframe
    current_row = 0
    
    # Initialize a mouse object for collecting mouse clicks
    mouse = event.Mouse(win=win)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "file_selection" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('file_selection.started', globalClock.getTime())
    # keep track of which components have finished
    file_selectionComponents = []
    for thisComponent in file_selectionComponents:
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
    
    # --- Run Routine "file_selection" ---
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in file_selectionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "file_selection" ---
    for thisComponent in file_selectionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('file_selection.stopped', globalClock.getTime())
    # the Routine "file_selection" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_task" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_task.started', globalClock.getTime())
    # setup some python lists for storing info about the instructions_mouse
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    instruction_taskComponents = [instructions_mouse, instructions]
    for thisComponent in instruction_taskComponents:
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
    
    # --- Run Routine "instruction_task" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *instructions_mouse* updates
        
        # if instructions_mouse is starting this frame...
        if instructions_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_mouse.frameNStart = frameN  # exact frame index
            instructions_mouse.tStart = t  # local t and not account for scr refresh
            instructions_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instructions_mouse.started', t)
            # update status
            instructions_mouse.status = STARTED
            instructions_mouse.mouseClock.reset()
            prevButtonState = instructions_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instructions_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instructions_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    continueRoutine = False  # end routine on response        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions.started')
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_taskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_task" ---
    for thisComponent in instruction_taskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction_task.stopped', globalClock.getTime())
    # store data for thisExp (ExperimentHandler)
    x, y = instructions_mouse.getPos()
    buttons = instructions_mouse.getPressed()
    thisExp.addData('instructions_mouse.x', x)
    thisExp.addData('instructions_mouse.y', y)
    thisExp.addData('instructions_mouse.leftButton', buttons[0])
    thisExp.addData('instructions_mouse.midButton', buttons[1])
    thisExp.addData('instructions_mouse.rightButton', buttons[2])
    thisExp.nextEntry()
    # the Routine "instruction_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(conditions_file),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_1 = data.TrialHandler(nReps=5.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trials_1')
        thisExp.addLoop(trials_1)  # add the loop to the experiment
        thisTrial_1 = trials_1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_1.rgb)
        if thisTrial_1 != None:
            for paramName in thisTrial_1:
                globals()[paramName] = thisTrial_1[paramName]
        
        for thisTrial_1 in trials_1:
            currentLoop = trials_1
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_1.rgb)
            if thisTrial_1 != None:
                for paramName in thisTrial_1:
                    globals()[paramName] = thisTrial_1[paramName]
            
            # --- Prepare to start Routine "start_trial_ring" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('start_trial_ring.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            win.mouseVisible =True #change to false
            #win.mouseVisible = False
            
            x_mouse,y_mouse = ring_mouse.getPos()
            
            ring_mouse_radius = euclidean_dist([fixation_ring.pos[0],fixation_ring.pos[1]], [x_mouse,y_mouse])
            
            Ring_W = ring_mouse_radius
            Ring_H = ring_mouse_radius
            # setup some python lists for storing info about the ring_mouse
            ring_mouse.x = []
            ring_mouse.y = []
            ring_mouse.leftButton = []
            ring_mouse.midButton = []
            ring_mouse.rightButton = []
            ring_mouse.time = []
            gotValidClick = False  # until a click is received
            # keep track of which components have finished
            start_trial_ringComponents = [ring_mouse, Ring, fixation_ring, mouse_circle_ring]
            for thisComponent in start_trial_ringComponents:
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
            
            # --- Run Routine "start_trial_ring" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                x_mouse,y_mouse = ring_mouse.getPos()
                
                ring_mouse_radius = euclidean_dist([fixation_ring.pos[0],fixation_ring.pos[1]], [x_mouse,y_mouse])
                
                if ring_mouse_radius <= 1:
                    mouse_circle_ring.opacity = 1
                    Ring.opacity = 1
                    if fixation_ring.contains(ring_mouse):
                        continueRoutine = False
                else:
                    mouse_circle_ring.opacity = 0
                    Ring_W = ring_mouse_radius
                    Ring_H = ring_mouse_radius
                
                # *ring_mouse* updates
                
                # if ring_mouse is starting this frame...
                if ring_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ring_mouse.frameNStart = frameN  # exact frame index
                    ring_mouse.tStart = t  # local t and not account for scr refresh
                    ring_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ring_mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('ring_mouse.started', t)
                    # update status
                    ring_mouse.status = STARTED
                    ring_mouse.mouseClock.reset()
                    prevButtonState = ring_mouse.getPressed()  # if button is down already this ISN'T a new click
                if ring_mouse.status == STARTED:  # only update if started and not finished!
                    x, y = ring_mouse.getPos()
                    ring_mouse.x.append(x)
                    ring_mouse.y.append(y)
                    buttons = ring_mouse.getPressed()
                    ring_mouse.leftButton.append(buttons[0])
                    ring_mouse.midButton.append(buttons[1])
                    ring_mouse.rightButton.append(buttons[2])
                    ring_mouse.time.append(ring_mouse.mouseClock.getTime())
                
                # *Ring* updates
                
                # if Ring is starting this frame...
                if Ring.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Ring.frameNStart = frameN  # exact frame index
                    Ring.tStart = t  # local t and not account for scr refresh
                    Ring.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Ring, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Ring.started')
                    # update status
                    Ring.status = STARTED
                    Ring.setAutoDraw(True)
                
                # if Ring is active this frame...
                if Ring.status == STARTED:
                    # update params
                    Ring.setSize([Ring_W, Ring_H], log=False)
                
                # *fixation_ring* updates
                
                # if fixation_ring is starting this frame...
                if fixation_ring.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_ring.frameNStart = frameN  # exact frame index
                    fixation_ring.tStart = t  # local t and not account for scr refresh
                    fixation_ring.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_ring, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_ring.started')
                    # update status
                    fixation_ring.status = STARTED
                    fixation_ring.setAutoDraw(True)
                
                # if fixation_ring is active this frame...
                if fixation_ring.status == STARTED:
                    # update params
                    pass
                
                # *mouse_circle_ring* updates
                
                # if mouse_circle_ring is starting this frame...
                if mouse_circle_ring.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_circle_ring.frameNStart = frameN  # exact frame index
                    mouse_circle_ring.tStart = t  # local t and not account for scr refresh
                    mouse_circle_ring.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_circle_ring, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'mouse_circle_ring.started')
                    # update status
                    mouse_circle_ring.status = STARTED
                    mouse_circle_ring.setAutoDraw(True)
                
                # if mouse_circle_ring is active this frame...
                if mouse_circle_ring.status == STARTED:
                    # update params
                    mouse_circle_ring.setPos([baseline_mouse.getPos()], log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in start_trial_ringComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "start_trial_ring" ---
            for thisComponent in start_trial_ringComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('start_trial_ring.stopped', globalClock.getTime())
            # store data for trials_1 (TrialHandler)
            trials_1.addData('ring_mouse.x', ring_mouse.x)
            trials_1.addData('ring_mouse.y', ring_mouse.y)
            trials_1.addData('ring_mouse.leftButton', ring_mouse.leftButton)
            trials_1.addData('ring_mouse.midButton', ring_mouse.midButton)
            trials_1.addData('ring_mouse.rightButton', ring_mouse.rightButton)
            trials_1.addData('ring_mouse.time', ring_mouse.time)
            # the Routine "start_trial_ring" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "origin_hold1" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('origin_hold1.started', globalClock.getTime())
            # Run 'Begin Routine' code from origin_hold
            continue_loop = True
            
            
            win.mouseVisible = False
            
            fixed_delay_started = False
            random_delay_started = False
            fixed_delay_timer = core.Clock()
            random_delay_timer = core.Clock()
            fixed_delay_timer.reset()
            random_delay_timer.reset()
            random_delay_duration = random.choice(RANDOM_DELAY_DURATIONS)
            
            
            
            
            # setup some python lists for storing info about the baseline_mouse
            baseline_mouse.x = []
            baseline_mouse.y = []
            baseline_mouse.leftButton = []
            baseline_mouse.midButton = []
            baseline_mouse.rightButton = []
            baseline_mouse.time = []
            gotValidClick = False  # until a click is received
            baseline_target_circle.setPos((target_x, target_y))
            # keep track of which components have finished
            origin_hold1Components = [baseline_mouse, baseline_fixation, baseline_target_circle, mouse_circle]
            for thisComponent in origin_hold1Components:
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
            
            # --- Run Routine "origin_hold1" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from origin_hold
                x_mouse, y_mouse = mouse.getPos()  # Replace 'mouse' with your mouse component's name
                ring_mouse_radius = euclidean_dist(fixation_ring.pos, [x_mouse, y_mouse])
                
                if ring_mouse_radius > 1:
                    continueRoutine = False  # This ends the current routine
                
                
                if not baseline_fixation.contains(baseline_mouse):
                    fixed_delay_started = False
                    random_delay_started = False
                    fixed_delay_timer.reset()
                    random_delay_timer.reset()
                else:
                    if not fixed_delay_started:
                        fixed_delay_started = True
                        fixed_delay_timer.reset()
                    elif not random_delay_started:
                        if fixed_delay_timer.getTime() >= FIXED_DELAY_DURATION:
                            random_delay_started = True
                            random_delay_timer.reset()
                            baseline_target_circle.autodraw = True
                            baseline_target_circle.status = STARTED
                    else:
                        if random_delay_timer.getTime() >= random_delay_duration:
                            continueRoutine = False
                
                
                
                 
                
                # *baseline_mouse* updates
                
                # if baseline_mouse is starting this frame...
                if baseline_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    baseline_mouse.frameNStart = frameN  # exact frame index
                    baseline_mouse.tStart = t  # local t and not account for scr refresh
                    baseline_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(baseline_mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('baseline_mouse.started', t)
                    # update status
                    baseline_mouse.status = STARTED
                    baseline_mouse.mouseClock.reset()
                    prevButtonState = baseline_mouse.getPressed()  # if button is down already this ISN'T a new click
                if baseline_mouse.status == STARTED:  # only update if started and not finished!
                    x, y = baseline_mouse.getPos()
                    baseline_mouse.x.append(x)
                    baseline_mouse.y.append(y)
                    buttons = baseline_mouse.getPressed()
                    baseline_mouse.leftButton.append(buttons[0])
                    baseline_mouse.midButton.append(buttons[1])
                    baseline_mouse.rightButton.append(buttons[2])
                    baseline_mouse.time.append(baseline_mouse.mouseClock.getTime())
                
                # *baseline_fixation* updates
                
                # if baseline_fixation is starting this frame...
                if baseline_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    baseline_fixation.frameNStart = frameN  # exact frame index
                    baseline_fixation.tStart = t  # local t and not account for scr refresh
                    baseline_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(baseline_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'baseline_fixation.started')
                    # update status
                    baseline_fixation.status = STARTED
                    baseline_fixation.setAutoDraw(True)
                
                # if baseline_fixation is active this frame...
                if baseline_fixation.status == STARTED:
                    # update params
                    pass
                
                # *baseline_target_circle* updates
                
                # if baseline_target_circle is active this frame...
                if baseline_target_circle.status == STARTED:
                    # update params
                    pass
                
                # *mouse_circle* updates
                
                # if mouse_circle is starting this frame...
                if mouse_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_circle.frameNStart = frameN  # exact frame index
                    mouse_circle.tStart = t  # local t and not account for scr refresh
                    mouse_circle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_circle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'mouse_circle.started')
                    # update status
                    mouse_circle.status = STARTED
                    mouse_circle.setAutoDraw(True)
                
                # if mouse_circle is active this frame...
                if mouse_circle.status == STARTED:
                    # update params
                    mouse_circle.setPos([baseline_mouse.getPos()], log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in origin_hold1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "origin_hold1" ---
            for thisComponent in origin_hold1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('origin_hold1.stopped', globalClock.getTime())
            # Run 'End Routine' code from origin_hold
            if ring_mouse_radius <= 1:
                trials_1.finished = True  # Replace 'yourLoopName' with the name of your loop
            
            # store data for trials_1 (TrialHandler)
            trials_1.addData('baseline_mouse.x', baseline_mouse.x)
            trials_1.addData('baseline_mouse.y', baseline_mouse.y)
            trials_1.addData('baseline_mouse.leftButton', baseline_mouse.leftButton)
            trials_1.addData('baseline_mouse.midButton', baseline_mouse.midButton)
            trials_1.addData('baseline_mouse.rightButton', baseline_mouse.rightButton)
            trials_1.addData('baseline_mouse.time', baseline_mouse.time)
            # the Routine "origin_hold1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 5.0 repeats of 'trials_1'
        
        
        # --- Prepare to start Routine "move" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('move.started', globalClock.getTime())
        # Run 'Begin Routine' code from rotated_moving
        from psychopy import sound, core, visual, event
        #routineClock.reset()
        #sound_1.play()
        
        bl = df.iloc[current_row]['block']
        ty= df.iloc[current_row]['type']
        
        # Begin Routine
        rotated_target_circle_2.setPos((0, 0))  # Set target position to the center
        rotated_target_circle_2.opacity = 0  # Make the target invisible
        
        #feedback change!
        rotated_target_circle_2.opacity = 1
        jumping_target.opacity = 0
        '''
        #preparation change
        if ty == 'preparation':
            prep_time = df.iloc[current_row]['p_time']
            time_value = 0.750 - prep_time
            interval = 0.25
        
            for i in range(4):
                beep = sound.Sound('A', secs=0.1)
                beep.play()
        
                # Check if it's time to show the rotated_target_circle_2
                while i == 1:  
                    if p_time==0.4:
                        core.wait(0.1)
                        rotated_target_circle_2.opacity = 1
                        break
        
                core.wait(interval)
        '''
        #
        point_x, point_y = 0, 0
        last_mouse_x, last_mouse_y = 0, 0
        
        if bl == 'washout' or ty == 'generalization':
            mouse_circle_2.opacity = 0
        else:
            mouse_circle_2.opacity = 1
        
        # rotation angle
        angle = df.iloc[current_row]['rotation_angle']
        win.mouseVisible = False
        
        movement_started = False
        movement_ended = False
        
        movement_clock = core.Clock()
        
        lastPos = rotated_move_mouse.getPos()
        
        def test_movement_started():
            return euclidean_dist(rotated_move_mouse.getPos(), rotated_fixation_2.pos) > 2
        
        def test_movement_ended(mouseDist, movement_duration):
            if  mouseDist < 0.1 or movement_duration > 2:
                movement_ended = True
            else:
                movement_ended = False
            return movement_ended
        
        feedback_timer = core.Clock()
        feedback_timer.reset()
        
        
        
        # setup some python lists for storing info about the rotated_move_mouse
        rotated_move_mouse.x = []
        rotated_move_mouse.y = []
        rotated_move_mouse.leftButton = []
        rotated_move_mouse.midButton = []
        rotated_move_mouse.rightButton = []
        rotated_move_mouse.time = []
        gotValidClick = False  # until a click is received
        rotated_move_mouse.mouseClock.reset()
        rotated_target_circle_2.setPos((target_x, target_y))
        jumping_target.setPos((f_x,f_y))
        # keep track of which components have finished
        moveComponents = [rotated_move_mouse, rotated_fixation_2, rotated_target_circle_2, mouse_circle_2, jumping_target]
        for thisComponent in moveComponents:
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
        
        # --- Run Routine "move" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from rotated_moving
            interval = 0.25
            
            # Play beeps
            for i in range(4):
                beep = sound.Sound('A', secs=0.1)  # Initialize sound each time
                beep.play()
                core.wait(interval)
            '''
            t = routineClock.getTime()
            
            if current_beep < 4 and t >= beep_times[current_beep]:
                beep_start_time = t  # Set start time for the beep
                current_beep += 1
            else:
                beep_start_time = -1 
                
            # Show target 300 ms before the last beep
            if t > beep_duration * 3 + beep_interval * 2 and t < beep_duration * 4 + beep_interval * 3 - target_duration:
                show_target()
            
            # Check if the mouse is at the origin
            if not is_mouse_at_origin():
                    # Perform action if the mouse is not at the origin (e.g., end the routine)
                    # You can add your own code here
                pass
            '''
            
            # Get current mouse position
            current_mouse_x, current_mouse_y = mouse.getPos()
            
            # Calculate mouse movement vector
            delta_x = current_mouse_x - last_mouse_x
            delta_y = current_mouse_y - last_mouse_y
            
            # Rotate the movement vector by 30 degrees
            theta = math.radians(-angle)
            delta_x_rotated, delta_y_rotated = rotate_vector(delta_x, delta_y, theta)
            
            mouseDist = euclidean_dist((0, 0), (current_mouse_x, current_mouse_y))
            if mouseDist <= MAX_DISTANCE:
                point_x += delta_x_rotated
                point_y += delta_y_rotated
                
            # Update last known mouse position
            last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y
            
            # Calculate distance between rotated_move_mouse and center
            thisPos = rotated_move_mouse.getPos()
            lastPos = thisPos
            movement_duration = movement_clock.getTime()
            
            if not movement_started:
                movement_clock.reset()
                if test_movement_started():
                    movement_started = True
                    move_onset_time1 = core.monotonicClock.getTime()
                    move_onset_time2 = ring_mouse.mouseClock.getTime()
                    
            else:
                if test_movement_ended(mouseDist, movement_duration):
                    movement_duration = movement_clock.getTime()
                    move_end_time1 = core.monotonicClock.getTime()
                    move_end_time2 = ring_mouse.mouseClock.getTime()
                    continueRoutine = False
            
            current_mouse_x, current_mouse_y = mouse.getPos()
            delta_x = current_mouse_x - last_mouse_x
            delta_y = current_mouse_y - last_mouse_y
            
            # Calculate the Euclidean distance moved by the mouse in this frame
            mouseDist = euclidean_dist((0, 0), (current_mouse_x, current_mouse_y))
            
            if mouseDist <= MAX_DISTANCE:
                delta_x_rotated, delta_y_rotated = rotate_vector(delta_x, delta_y, theta)
                # Update the position of rotated_move_mouse
                point_x += delta_x_rotated
                point_y += delta_y_rotated
            # *rotated_move_mouse* updates
            
            # if rotated_move_mouse is starting this frame...
            if rotated_move_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                rotated_move_mouse.frameNStart = frameN  # exact frame index
                rotated_move_mouse.tStart = t  # local t and not account for scr refresh
                rotated_move_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_move_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('rotated_move_mouse.started', t)
                # update status
                rotated_move_mouse.status = STARTED
                prevButtonState = rotated_move_mouse.getPressed()  # if button is down already this ISN'T a new click
            if rotated_move_mouse.status == STARTED:  # only update if started and not finished!
                x, y = rotated_move_mouse.getPos()
                rotated_move_mouse.x.append(x)
                rotated_move_mouse.y.append(y)
                buttons = rotated_move_mouse.getPressed()
                rotated_move_mouse.leftButton.append(buttons[0])
                rotated_move_mouse.midButton.append(buttons[1])
                rotated_move_mouse.rightButton.append(buttons[2])
                rotated_move_mouse.time.append(rotated_move_mouse.mouseClock.getTime())
            
            # *rotated_fixation_2* updates
            
            # if rotated_fixation_2 is starting this frame...
            if rotated_fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_fixation_2.frameNStart = frameN  # exact frame index
                rotated_fixation_2.tStart = t  # local t and not account for scr refresh
                rotated_fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rotated_fixation_2.started')
                # update status
                rotated_fixation_2.status = STARTED
                rotated_fixation_2.setAutoDraw(True)
            
            # if rotated_fixation_2 is active this frame...
            if rotated_fixation_2.status == STARTED:
                # update params
                pass
            
            # *rotated_target_circle_2* updates
            
            # if rotated_target_circle_2 is starting this frame...
            if rotated_target_circle_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                rotated_target_circle_2.frameNStart = frameN  # exact frame index
                rotated_target_circle_2.tStart = t  # local t and not account for scr refresh
                rotated_target_circle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_target_circle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rotated_target_circle_2.started')
                # update status
                rotated_target_circle_2.status = STARTED
                rotated_target_circle_2.setAutoDraw(True)
            
            # if rotated_target_circle_2 is active this frame...
            if rotated_target_circle_2.status == STARTED:
                # update params
                pass
            
            # *mouse_circle_2* updates
            
            # if mouse_circle_2 is starting this frame...
            if mouse_circle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_circle_2.frameNStart = frameN  # exact frame index
                mouse_circle_2.tStart = t  # local t and not account for scr refresh
                mouse_circle_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_circle_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mouse_circle_2.started')
                # update status
                mouse_circle_2.status = STARTED
                mouse_circle_2.setAutoDraw(True)
            
            # if mouse_circle_2 is active this frame...
            if mouse_circle_2.status == STARTED:
                # update params
                mouse_circle_2.setPos([point_x, point_y], log=False)
            
            # *jumping_target* updates
            
            # if jumping_target is starting this frame...
            if jumping_target.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                jumping_target.frameNStart = frameN  # exact frame index
                jumping_target.tStart = t  # local t and not account for scr refresh
                jumping_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(jumping_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'jumping_target.started')
                # update status
                jumping_target.status = STARTED
                jumping_target.setAutoDraw(True)
            
            # if jumping_target is active this frame...
            if jumping_target.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in moveComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "move" ---
        for thisComponent in moveComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('move.stopped', globalClock.getTime())
        # Run 'End Routine' code from rotated_moving
        win.mouseVisible = False
        continueRoutine = False
        
        thisExp.addData('MoveOnsetTime1', move_onset_time1)
        thisExp.addData('MoveEndTime1', move_end_time1)
        
        thisExp.addData('MoveOnsetTime2', move_onset_time2)
        thisExp.addData('MoveEndTime2', move_end_time2)
        
        
        '''AR
        new_mouse_pos_x = rotated_move_mouse.getPos()[0]
        new_mouse_pos_y = rotated_move_mouse.getPos()[1]
        '''
        #new_mouse_pos_x = point_x
        #new_mouse_pos_y =point_y
        
        # store data for trials (TrialHandler)
        trials.addData('rotated_move_mouse.x', rotated_move_mouse.x)
        trials.addData('rotated_move_mouse.y', rotated_move_mouse.y)
        trials.addData('rotated_move_mouse.leftButton', rotated_move_mouse.leftButton)
        trials.addData('rotated_move_mouse.midButton', rotated_move_mouse.midButton)
        trials.addData('rotated_move_mouse.rightButton', rotated_move_mouse.rightButton)
        trials.addData('rotated_move_mouse.time', rotated_move_mouse.time)
        # the Routine "move" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from rotated_feedback_code
        # Assuming 'current_row' is defined somewhere in your script
        bl = df.iloc[current_row]['block']
        ty = df.iloc[current_row]['type']
        
        # Assuming 'jumping_target', 'rotated_target_circle_2', and 'rotated_mouse_circle' are defined stimuli
        jumping_target_2.opacity = 0
        rotated_target_circle_2.opacity = 0
        rotated_mouse_circle.opacity = 0
        
        if ty == 'feedback':
            jumping_target_2.opacity = 1
        else:
            rotated_target_circle_2.opacity = 1
        
        if bl == 'washout' or ty == 'generalization':
            rotated_mouse_circle.opacity = 0
        else:
            rotated_mouse_circle.opacity = 1
        
        # Assuming 'feedback_timer' is defined as a core.Clock somewhere in your script
        feedback_timer.reset()
        
        feedback_turned_on = False
        feedback_turned_off = False
        
        # Assuming 'rotated_target_circle2' is defined
        rotated_target_circle2.opacity = 0
        
        # setup some python lists for storing info about the rotated_feedback_mouse
        rotated_feedback_mouse.x = []
        rotated_feedback_mouse.y = []
        rotated_feedback_mouse.leftButton = []
        rotated_feedback_mouse.midButton = []
        rotated_feedback_mouse.rightButton = []
        rotated_feedback_mouse.time = []
        gotValidClick = False  # until a click is received
        rotated_feedback_mouse.mouseClock.reset()
        rotated_target_circle2.setPos((target_x, target_y))
        rotated_mouse_circle.setPos((point_x,point_y))
        jumping_target_2.setPos((f_x,f_y))
        # keep track of which components have finished
        feedbackComponents = [rotated_feedback_mouse, rotated_fixation2, rotated_target_circle2, rotated_mouse_circle, jumping_target_2]
        for thisComponent in feedbackComponents:
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
        
        # --- Run Routine "feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from rotated_feedback_code
            
            if not feedback_turned_on:
                if feedback_timer.getTime() >= FEEDBACK_DELAY:
                    #rotated_mouse_circle.opacity = 1
                    #rotated_target_circle2.opacity = 1
                    feedback_turned_on = True
            elif not feedback_turned_off:
                if feedback_timer.getTime() >= (FEEDBACK_DELAY + FEEDBACK_DURATION):
                    rotated_mouse_circle.opacity = 0
                    feedback_turned_off = True
                    rotated_fixation2.opacity = 0
                    rotated_target_circle2.opacity = 0
            else:
                if feedback_timer.getTime() >= (FEEDBACK_DELAY + FEEDBACK_DURATION + BLANK_SCREEN):
                    continueRoutine = False
            
            ty= df.iloc[current_row]['type']
            
            
            if bl == 'washout' or ty == 'generalization':
                mouse_circle_2.opacity = 0
            else:
                mouse_circle_2.opacity = 1
            
            
            # *rotated_feedback_mouse* updates
            
            # if rotated_feedback_mouse is starting this frame...
            if rotated_feedback_mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                rotated_feedback_mouse.frameNStart = frameN  # exact frame index
                rotated_feedback_mouse.tStart = t  # local t and not account for scr refresh
                rotated_feedback_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_feedback_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('rotated_feedback_mouse.started', t)
                # update status
                rotated_feedback_mouse.status = STARTED
                prevButtonState = rotated_feedback_mouse.getPressed()  # if button is down already this ISN'T a new click
            if rotated_feedback_mouse.status == STARTED:  # only update if started and not finished!
                x, y = rotated_feedback_mouse.getPos()
                rotated_feedback_mouse.x.append(x)
                rotated_feedback_mouse.y.append(y)
                buttons = rotated_feedback_mouse.getPressed()
                rotated_feedback_mouse.leftButton.append(buttons[0])
                rotated_feedback_mouse.midButton.append(buttons[1])
                rotated_feedback_mouse.rightButton.append(buttons[2])
                rotated_feedback_mouse.time.append(rotated_feedback_mouse.mouseClock.getTime())
            
            # *rotated_fixation2* updates
            
            # if rotated_fixation2 is starting this frame...
            if rotated_fixation2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_fixation2.frameNStart = frameN  # exact frame index
                rotated_fixation2.tStart = t  # local t and not account for scr refresh
                rotated_fixation2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_fixation2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rotated_fixation2.started')
                # update status
                rotated_fixation2.status = STARTED
                rotated_fixation2.setAutoDraw(True)
            
            # if rotated_fixation2 is active this frame...
            if rotated_fixation2.status == STARTED:
                # update params
                pass
            
            # *rotated_target_circle2* updates
            
            # if rotated_target_circle2 is starting this frame...
            if rotated_target_circle2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                rotated_target_circle2.frameNStart = frameN  # exact frame index
                rotated_target_circle2.tStart = t  # local t and not account for scr refresh
                rotated_target_circle2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_target_circle2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rotated_target_circle2.started')
                # update status
                rotated_target_circle2.status = STARTED
                rotated_target_circle2.setAutoDraw(True)
            
            # if rotated_target_circle2 is active this frame...
            if rotated_target_circle2.status == STARTED:
                # update params
                pass
            
            # *rotated_mouse_circle* updates
            
            # if rotated_mouse_circle is starting this frame...
            if rotated_mouse_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_mouse_circle.frameNStart = frameN  # exact frame index
                rotated_mouse_circle.tStart = t  # local t and not account for scr refresh
                rotated_mouse_circle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_mouse_circle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rotated_mouse_circle.started')
                # update status
                rotated_mouse_circle.status = STARTED
                rotated_mouse_circle.setAutoDraw(True)
            
            # if rotated_mouse_circle is active this frame...
            if rotated_mouse_circle.status == STARTED:
                # update params
                pass
            
            # *jumping_target_2* updates
            
            # if jumping_target_2 is starting this frame...
            if jumping_target_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                jumping_target_2.frameNStart = frameN  # exact frame index
                jumping_target_2.tStart = t  # local t and not account for scr refresh
                jumping_target_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(jumping_target_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'jumping_target_2.started')
                # update status
                jumping_target_2.status = STARTED
                jumping_target_2.setAutoDraw(True)
            
            # if jumping_target_2 is active this frame...
            if jumping_target_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback.stopped', globalClock.getTime())
        # Run 'End Routine' code from rotated_feedback_code
        
        
        
        # store data for trials (TrialHandler)
        trials.addData('rotated_feedback_mouse.x', rotated_feedback_mouse.x)
        trials.addData('rotated_feedback_mouse.y', rotated_feedback_mouse.y)
        trials.addData('rotated_feedback_mouse.leftButton', rotated_feedback_mouse.leftButton)
        trials.addData('rotated_feedback_mouse.midButton', rotated_feedback_mouse.midButton)
        trials.addData('rotated_feedback_mouse.rightButton', rotated_feedback_mouse.rightButton)
        trials.addData('rotated_feedback_mouse.time', rotated_feedback_mouse.time)
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "instruction" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('instruction.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_2
        image_value = df.iloc[current_row]['ins_images']
        
        display_image = False
        wait_for_click = False
        
        if pd.notna(image_value) and os.path.exists(image_value):  # Check if the image value exists
            # Setup the image stimulus for display
            image_stim = visual.ImageStim(win, image=image_value, pos=(0,0), size=(15, 15))
            display_image = True
            wait_for_click = True  # If there's an image, we'll wait for a click
        
        mouse.clickReset()  # Reset the mouse click state at the beginning of each routine
        
        
        
        
        # keep track of which components have finished
        instructionComponents = []
        for thisComponent in instructionComponents:
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
        
        # --- Run Routine "instruction" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_2
            if display_image:
                image_stim.draw()
                
                if wait_for_click:
                    waiting = True
                    while waiting:
                        if mouse.getPressed()[0]:  # If the left mouse button is pressed
                            waiting = False
                        image_stim.draw()  # You have to redraw the image while waiting
                        win.flip()  # Ensure that the image is displayed while waiting
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruction" ---
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('instruction.stopped', globalClock.getTime())
        # Run 'End Routine' code from code_2
        if current_row < len(df) - 1:  # Check if it's not the last row
            current_row += 1
        # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


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


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
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
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
