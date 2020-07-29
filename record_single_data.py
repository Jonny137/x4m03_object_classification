import sys
from optparse import OptionParser
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pymoduleconnector
from pymoduleconnector import DataType
from pathlib import Path
import serial
import serial.tools.list_ports

import warnings
warnings.filterwarnings("ignore")

# Current directory for data recording storage
currentPath = str(Path().absolute())
storePath = currentPath + '\\' + 'recordedData\\'

# Reset module method
def resetModule(devicePort):
    mc = pymoduleconnector.ModuleConnector(devicePort)
    x4m03 = mc.get_xep()
    x4m03.module_reset()
    mc.close()
    sleep(3)

def clearBuffer(mc):
    """Clears the frame buffer"""
    x4m03 = mc.get_xep()
    while x4m03.peek_message_data_float():
        x4m03.read_message_data_float()

def simple_xep_plot(devicePort, record=False, baseband=True, startDist=0, endDist=5):

    FPS = 20
    directory = '.'
    resetModule(devicePort)
    mc = pymoduleconnector.ModuleConnector(devicePort)

    # Driver parameter initialization
    x4m03 = mc.get_xep()
    # Set DAC range
    x4m03.x4driver_set_dac_min(900)
    x4m03.x4driver_set_dac_max(1150)

    # Set integration
    x4m03.x4driver_set_iterations(16)
    x4m03.x4driver_set_pulses_per_step(26)

    x4m03.x4driver_set_downconversion(int(baseband))

    # Edit radar range
    # Frame offset for X4M03 is fixed on 0.18 meters
    x4m03.x4driver_set_frame_area_offset(0.18)
    x4m03.x4driver_set_frame_area(startDist, endDist)

    # Start streaming of data
    x4m03.x4driver_set_fps(FPS)

    def readFrame():
        """Gets frame data from module"""
        d = x4m03.read_message_data_float()
        frame = np.array(d.data)
         # Convert the resulting frame to a complex array if downconversion is 
         # enabled
        if baseband:
            n = len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
        return frame

    quit = False

    plt.ion()
    plt.figure()
    plt.title("X4M03 Baseband Data")

    clearBuffer(mc)
    frame = readFrame()

    t = np.arange(len(frame))

    data_amp = np.array(t)
    data_phi = np.array(t)

    while True:

        clearBuffer(mc)
        frame = readFrame()
        amps = np.abs(frame)
        phis = np.angle(frame)

        plt.subplot(211)
        plt.title("X4M03 Baseband Data")
        plt.subplots_adjust(hspace=0.4)

        plt.ylim((-0.001, 0.03))
        plt.plot(amps)
        plt.xlabel("Bin #")
        plt.ylabel("Normalized Amplitude")
        plt.pause(0.001)
        plt.grid(True)

        plt.subplot(212)
        plt.ylim((-3.5, 3.5))
        plt.plot(phis)
        plt.xlabel("Bin #")
        plt.ylabel("Phase [rad]")
        plt.pause(0.001)
        plt.grid(True)

        data_amp = np.vstack((data_amp, np.abs(frame)))
        data_phi = np.vstack((data_phi, np.angle(frame)))

        while True:
            print("Comands: q - quit, s - new record, Enter - new data, \
                   w name - save data to file")
            command = input("Enter command >> ")
            print("Command: ",command)

            if (command == "Q" or command == "q"):
                quit = True
                break
            elif(command == "s" or command == ""):
                plt.clf()
                data_amp = np.array(t)
                data_phi = np.array(t)
                break
            elif(command == "c"):
                break
            elif(command[0] == "w" and command[1]==" "):
                filename_amp = storePath + command[2:] + "_amp.csv"
                filename_phi = storePath + command[2:] + "_phi.csv"
                np.savetxt(filename_amp, np.transpose(data_amp), delimiter=", ")
                print("File recorded with name: ", filename_amp)
                np.savetxt(filename_phi, np.transpose(data_phi), delimiter=", ")
                print("File recorded with name: ", filename_phi)

        if (quit):
            break

    x4m03.x4driver_set_fps(0)
    print("Exiting...")

def main():
    # Code block for automatic search of port where X4M03 is connected
    devicePort = ''
    print('Searching...')
    ports = serial.tools.list_ports.comports(include_links=False)
    for port in ports :
        print('Found port '+ port.device)
        devicePort = port.device

    startDist = 0
    endDist = 5
    simple_xep_plot(devicePort, record=False, 
                                baseband=True, 
                                startDist=0, 
                                endDist=5)

if __name__ == "__main__":
   main()
