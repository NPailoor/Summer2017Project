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
        ampMedian = medianAmp(ampLog)
        #Normalize with respect to a steady-state
        ampNormalized = medianAmp(ampMedian)


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
    #timeRange = list(range(-40,120))
    ampArray = np.asarray(ampLog)
    ampMedian = signal.medfilt(ampArray, 5)
    #p.subplot(211)
    #p.plot(timeRange, ampLog)
    #p.axis([-40, 120, 1.25, 1.5])
    #p.subplot(212)
    #p.plot(timeRange,ampMedian)
    #p.axis([-40, 120, 1.25, 1.5])
    #p.show()
    return ampMedian

def normalizedAmp(ampMedian):
    import numpy as np
    from scipy import signal
    from matplotlib import pyplot as p
    #timeRange = list(range(-40,120))
    steadyState = np.median(ampMedian[20:40])
    ampNormalized = ampMedian / steadyState
    #p.subplot(211)
    #p.plot(timeRange, ampMedian)
    #p.subplot(212)
    #p.plot(timeRange, ampNormalized)
    #p.show()
    return ampNormalized
    
def selectSample(tag):
    import sqlite3
    conn = sqlite3.connect('test.db')
    cursor = conn.execute("SELECT AMP FROM DATATABLE WHERE TAG = '" + tag + "'")
    row = cursor.fetchall()
    dataString = row[0][0]
    return dataString

def dataTest(tag):
    import numpy as np
    from matplotlib import pyplot as p
    dataRaw = parseAmp(selectSample(tag))
    dataLog = np.log10(dataRaw)
    dataMedian = medianAmp(dataLog)
    dataNormal = normalizedAmp(dataMedian)
    timeRange = list(range(-40,120))
    p.subplot(411)
    p.plot(timeRange, dataRaw)
    p.subplot(412)
    p.plot(timeRange, dataLog)
    p.subplot(413)
    p.plot(timeRange, dataMedian)
    p.subplot(414)
    p.plot(timeRange, dataNormal)
    p.show()
    
    
