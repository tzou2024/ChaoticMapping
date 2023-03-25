import mido
import os
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
from pprint import pprint
import sys


def generate_pitches(track):
    '''
    generae pitch order, start and stop time for note_on note_off commands
    '''
    pitches = []
    timer = 0

    #reformat to understand midi format better
    #get pitch ordering
    for message in track:
        timer += message.time
        if message.type == 'note_on':
            pitches.append({
                        "note":    message.note,
                        "starttime": timer
                        })
        else:
            for i in pitches[::-1]:
                if i["note"] == message.note:
                    i["endtime"] = timer
                    continue
    return pitches

# pitches = generate_pitches(track)
    
# runge-kutta fourth-order numerical integration
def rk4(func, tk, _yk, _dt=0.01, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time k+1
    """

    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

# # Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations



def lorenz(_t, _y, sigma=10, beta=(8 / 3), rho=28):
    '''
    lorenz chaotic differential equation: dy/dt = f(t, y)
    _t: time tk to evaluate system
    _y: 3D state vector [x, y, z]
    sigma: constant related to Prandtl number
    beta: geomatric physical property of fluid layer
    rho: constant related to the Rayleigh number
    return: [x_dot, y_dot, z_dot]
    '''
    
    return np.array([
        sigma * (_y[1] - _y[0]),
        (rho * _y[0]) - _y[1] -( _y[0] * _y[2]),
        (_y[0] * _y[1]) - (beta * _y[2]),
    ])






# propagate state
def states(initial_conditions, steps=176):
    #==============================================================
    # simulation harness
    #initial_conditions: 3D vector representing initial xyz conditions
    #og_state_history: state history to match up with 176 steps of bach176
    # ==============================================================
    # discrete step size
    dt = 0.01
    # lorenz initial conditions (x, y, z) at t = 0
    y0 = initial_conditions
        # simulation results
    og_state_history = []

    # initialize yk
    yk = y0
    
    # intialize time
    t = 0

    # iterate over time
    for q in range(0,steps):
        # save current state
        og_state_history.append(yk)

        # update state variables yk to yk+1
        yk = rk4(lorenz, t, yk, dt)
        # print(t)
        t += dt

    # convert list to numpy array
    og_state_history = np.array(og_state_history)
    
    return og_state_history

def generate_mapping(pitches, x1):
    #generate original IC reference tragectory to pitch mapping [(ref_trag0, pitch0)...]
    mapping1 = []
    for i in range(len(pitches)):
        mapping1.append((x1[i], pitches[i]["note"]))
    # pprint(mapping1)
    return mapping1

def new_mapping(pitchsequence, mapping,refx1, refx2):
    new_pitches = []
    for i in range(len(pitchsequence)):
        # print()
        # print("======================")
        # print("dealing with pitch: ", i)
        ex1, noter = mapping[i]
        # print("current (x1val, pitch):", ex1, noter)
        greater_than = [x for x in refx1 if x >= refx2[i]]
        # print("searching for smallest x1 greater than x2's value:", len(greater_than), "found")
        if greater_than:
            # print("found greater or equal")
            greater_than.sort()
            newx1val = greater_than[0]
            #now that we've found the smallest gte, we can look back into original mapping for a trajectory value j1
            #and replace the original pitch with the new correspondant pitch j2
            for j1, j2 in mapping:
                #because the reference trajectory values are all unique in this case, I can just straight up check for matching
                if newx1val == j1:
                    # print("found new pitch, reassigning new pitch based on new found value")
                    newa = {**pitchsequence[i]}
                    newa["note"] = j2
                    # print("old pitch", pitchsequence[i])
                    # print("new pitched", newa)
                    # if pitchsequence[i]["note"] != newa["note"]:
                        # print("*****ATTENTION************")
                        # print("NOT THE SAME PITCH", pitchsequence[i]["note"], newa["note"])
                        # print("**************************")
                    new_pitches.append(newa)
                    
        else:
            # print("no x1 val found greaterthan or equal to than x2 current val, using origional pitch")
            new_pitches.append(pitchsequence[i])
    # print("======================")
    # print()
    return new_pitches


if __name__ == "__main__":
    '''
    ex.  python3 chaoticmapping.py bach176.mid 1 .999 1 1 altered3.mid
    arg1: song file
    arg2: channel to work on
    arg 3-5: chaotic reference trajectory x,y,z
    arg 6: file name to save altered midi
    
    returns:
    saves .mid file of altered song
    '''
  
    
    midiFile = sys.argv[1]
    channel = int(sys.argv[2])
    reftrag = (np.array([float(sys.argv[3]), float(sys.argv[4]),float(sys.argv[5])]))
    
    #import midi file
    mid = mido.MidiFile(midiFile, clip=True)
    
    #determine track based on input channel
    track = mid.tracks[channel][1:len(mid.tracks[channel]) - 1]
    
    #generate pitch ordering and reformat to help me understand midi
    #format better
    pitches = generate_pitches(track)
    #propogate state for 2 sets of initial conditions
    state_history1= states(np.array([1, 1, 1]))
    state_history2 = states(np.array([1.01, 1, 1]))

    #assemble 2 reference trajectories
    x1 = [z[0] for z in state_history1]
    x2 = [z[0] for z in state_history2]
    
    #generate pitch mapping
    mapping1 = generate_mapping(pitches, x1)
    
    #new pitches
    new_pitches = new_mapping(pitches, mapping1, x1, x2)
    
    
    commandlist = []
    for i in new_pitches:
        commandlist.append({
            "note": i["note"],
            "time": i["starttime"]
        })
        commandlist.append({
            "note": i["note"],
            "time": i["endtime"]
        })

    #because midi messages are send in order with delta times since last command
    #i can just sort my reformatted start and stop messages based on the time theyre
    #supposed to happen
    commandlist = sorted(commandlist, key=lambda i: i["time"])
    
    # pprint("++++++++++++++++++++++++++++=") 
    # pprint(mid.tracks[1])
    # pprint("++++++++++++++++++++++++++++=")
    # print(len(mapping1), len(x1), len(x2), len(new_pitches))
    # print(len(commandlist),len(mid.tracks[channel][1:len(pitches) * 2 + 1]))
    
    
    
    for i in range(len(mid.tracks[channel][1:len(pitches) * 2 + 1])):
        #overwrite messages in bach176 track with new notes
        mid.tracks[channel][1+i].note = commandlist[i]["note"]
    mid.save(sys.argv[6])