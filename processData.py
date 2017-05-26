def initialFiltering():
    import sqlite3
    import numpy
    import matplotlib
    conn = sqlite3.connect('test.db')
    cursor = conn.execute("SELECT AMP FROM DATATABLE")
    for row in cursor:
        #Parse AMP string values back into an array
        ampString = row[0][0]
        ampFloat = parseAmp(ampString)
        
        #Convert values to log scale
        ampLog = numpy.log10(ampFloat)
        #Apply median filter
        #Normalize with respect to a steady-state


def parseAmp(ampString):
    import numpy
    splitString = ampString.split('-')
    ampData = []
    for i in range(0,len(splitString)-1):
        ampData.append(float(splitString[i]))
    return ampData

def medianAmp(ampLog):
    import numpy as np
    from scipy import signal
    from matplotlib import pyplot as p
    timeRange = list(range(-40,120))
    ampArray = np.asarray(ampLog)
    ampMedian = signal.medfilt(ampArray, 5)
    p.subplot(211)
    p.plot(timeRange, ampLog)
    p.axis([-40, 120, 1.25, 1.5])
    p.subplot(212)
    p.plot(timeRange,ampMedian)
    p.axis([-40, 120, 1.25, 1.5])
    p.show()
    return ampMedian

    
