import pymoduleconnector as pmc
from time import sleep
import numpy as np
from pathlib import Path
from datetime import datetime
import serial
import serial.tools.list_ports

# Current directory for data recording storage
currentPath = str(Path().absolute())
storePath = currentPath + '\\recordedData\\'

# Method to check whether connection is established between PC and module
def checkConnection(devicePort):
    mc = pmc.ModuleConnector(devicePort)
    x4m03 = mc.get_xep()
    pong = x4m03.ping()
    print("Received pong, value is:",  hex(pong))

# Reset module method
def resetModule(devicePort):
    mc = pmc.ModuleConnector(devicePort)
    x4m03 = mc.get_xep()
    x4m03.module_reset()
    mc.close()
    sleep(3)

# Buffer clear method
def clearBuffer(mc):
    """Clears the frame buffer"""
    x4m03 = mc.get_xep()
    while x4m03.peek_message_data_float():
        x4m03.read_message_data_float()

def recordData(devicePort, staticRemoval=False, startDist=0, endDist=5,
                                                                baseband=True):
    # Initialize module
    resetModule(devicePort)
    mc = pmc.ModuleConnector(devicePort)
    x4m03 = mc.get_xep()

    # Initialize FPS
    FPS = 20
    # DAC parameters
    x4m03.x4driver_set_dac_min(900)
    x4m03.x4driver_set_dac_max(1150)
    # Integration parameters
    x4m03.x4driver_set_iterations(16)
    x4m03.x4driver_set_pulses_per_step(26)
    # Baseband parameter method
    x4m03.x4driver_set_downconversion(int(baseband))

    # Edit radar range
    # Frame offset for X4M03 is fixed on 0.18 meters
    x4m03.x4driver_set_frame_area_offset(0.18)
    x4m03.x4driver_set_frame_area(startDist, endDist)

    # Start streaming of data
    x4m03.x4driver_set_fps(FPS)

    # Read frame method
    def readFrame():
        """Gets frame data from module"""
        d = x4m03.read_message_data_float()
        frame = np.array(d.data)
         # Convert the resulting frame to a complex array
         # if downconversion is enabled
        if baseband:
            n = len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
        return frame

    # Prepare data for storing
    clearBuffer(mc)
    frame = readFrame()
    temp = np.arange(len(frame))
    dataAmp = np.array(temp);
    dataPhi = np.array(temp);

    startTime = datetime.now()
    # Prepare current time in string format for name of recording file
    nameTime = str(datetime.today().strftime('%Y-%m-%d_%H-%M'))

    # Loop for 15 seconds and record data
    while True:

        option = input('Insert filename to record or q to quit: ')
        if option.lower() == 'q':
            break
        else:
            frame = readFrame()
            frameAmp = np.abs(frame)
            for i in range(9):
                frameAmp[i] = 0.001
            framePhi = np.angle(frame)

            if staticRemoval:
                sum = 0
                for i in range(len(frameAmp)):
                    avg = sum / 180
                for i in range(len(frameAmp)):
                    frameAmp[i] -= avg

            dataAmp = np.vstack((dataAmp, frameAmp))
            dataPhi = np.vstack((dataPhi, framePhi))

            # Save data to the designated spot in csv format
            filenameAmp = storePath + 'Amp_' + option + '.csv'
            filenamePhi = storePath + 'Phi_' + option + '.csv'
            np.savetxt(filenameAmp, np.transpose(dataAmp), delimiter=", ")
            sleep(0.001)
            np.savetxt(filenamePhi, np.transpose(dataPhi), delimiter=", ")
            sleep(0.001)
            print('Recorded data')

    # Stop streaming of data
    x4m03.x4driver_set_fps(0)

def main():
    # Code block for automatic search of port where X4M03 is connected
    devicePort = ''
    print('Searching...')
    ports = serial.tools.list_ports.comports(include_links=False)
    for port in ports :
        print('Found port '+ port.device)
        devicePort = port.device

    # Change value on True for removal of static objects
    staticRemoval = False

    # Default values for full range from 0m..10m
    startDist = 0
    endDist = 5

    recordData(devicePort, staticRemoval, startDist, endDist, baseband=True)

if __name__ == "__main__":
   main()
