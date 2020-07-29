import pymoduleconnector as pmc
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports

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

def plotModuleData(devicePort, staticRemoval=False, startDist=0, endDist=5,
                   baseband=False):

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

    def animate(i):
        frame = readFrame()
        frameAmp = np.abs(frame)
        # Loop for first 16 samples to virtually negate low isolation
        for i in range(16):
            frameAmp[i] = 1e-03
        framePhi = np.angle(frame)

        if staticRemoval:
            sum = 0
            for i in range(len(frameAmp)):
                sum += frameAmp[i]
            avg = sum / 180
            for i in range(len(frameAmp)):
                frameAmp[i] -= avg

        line1.set_ydata(frameAmp)
        line2.set_ydata(framePhi)

        return line1, line2

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2,1)
    # Configure subplots title, grid, ylims, subplot gap
    fig.suptitle('X4M03 Baseband Time Domain')
    ax1.set_ylim(-0.001, 0.03)
    ax1.title.set_text('Amplitude')
    ax1.set_xlabel('Bin #')
    ax1.set_ylabel('Norm. Amplitude [0->1]')
    ax2.set_ylim(-3.5, 3.5)
    ax2.title.set_text('Phase')
    ax2.set_xlabel('Bin #')
    ax2.set_ylabel('Phi[rad]')

    # Enable grid, set tick labels and adjust subplot spacing
    for ax in [ax1, ax2]:
        ax.grid()
    fig.subplots_adjust(hspace=.6)

    frame = readFrame()
    frameAmp = np.abs(frame)
    # Loop for first 16 samples to virtually negate low isolation
    for i in range(16):
        frameAmp[i] = 1e-03
    framePhi = np.angle(frame)

    if staticRemoval:
        sum = 0
        for i in range(len(frameAmp)):
            avg = sum / 180
        for i in range(len(frameAmp)):
            frameAmp[i] -= avg

    line1, = ax1.plot(frameAmp)
    line2, = ax2.plot(framePhi)

    clearBuffer(mc)

    ani = FuncAnimation(fig, animate, interval=FPS)
    try:
        plt.show()
    finally:
        print('Exiting...')
        # Stop streaming of data
        x4m03.x4driver_set_fps(0)

def main():
    # Code block for automatic search of port where X4M03 is connected
    # Win 10 tested
    devicePort = ''
    print('Searching...')
    ports = serial.tools.list_ports.comports(include_links=False)
    for port in ports :
        print('Found port '+ port.device)
        devicePort = port.device

    # Change value on True for removal of static objects
    staticRemoval = False

    # Default values for full range from 0m..10m
    # Adjust these two values for range setup
    startDist = 0
    endDist = 5
    print(f'Range set to: Start = {startDist}m, End = {endDist}m')

    plotModuleData(devicePort, staticRemoval, startDist, endDist,baseband=True)

if __name__ == "__main__":
   main()
