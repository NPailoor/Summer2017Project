import sqlite3
import numpy
from  matplotlib import pyplot as p
from scipy import signal
from caffe2.python import workspace

def initialFiltering():
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    data = cursor.execute('''SELECT * FROM DATATABLE''')
    data = data.fetchall()
    with open("init_net.pb", 'rb') as f:
        init_net = f.read()
    with open("predict_net.pb", 'rb') as f:
        predict_net = f.read()
    p = workspace.Predictor(init_net, predict_net)
        
    for row in data:
        tag = row[8]
        #Parse AMP string values back into an array
        ampBytes = row[0]
        ampFloat = parseAmp(ampBytes)
        
        #Convert values to log scale
        ampLog = numpy.log10(ampFloat)
        #Apply median filter
        ampMedian = medianAmp(ampLog)
        #Normalize with respect to a steady-state
        ampNormalized = normalizedAmp(ampMedian)
        #Select narrower range of -10:30
        ampNarrowSample = ampNormalized[30:70]
        #normalize between 0 and 1
        min = numpy.amin(ampNarrowSample)
        max = numpy.amax(ampNarrowSample)
        ampFeatures = (ampNarrowSample - min)/(max - min)
        ampFeatures = ampFeatures.astype(numpy.float32)
        ampFeatures = ampFeatures.reshape(1,1,1,40)
        result = p.run([ampFeatures])
        event = numpy.argmax(result)
        cursor.execute('''UPDATE DATATABLE SET EVENT=? WHERE TAG=?''', (event, tag))
        print("Updated Table successfully")
    cursor.close()
    conn.commit()
        
def reduceFalsePositives():
    conn = sqlite3.connect('test.db')
    print "Opened database successfully";
    cursor = conn.cursor()
    data = cursor.execute('''SELECT * FROM DATATABLE WHERE EVENT=1''')
    data = data.fetchall()
    with open("init_net.pb", 'rb') as f:
        init_net = f.read()
    with open("predict_net.pb", 'rb') as f:
        predict_net = f.read()
    p = workspace.Predictor(init_net, predict_net)
    for row in data:
        tag = row[8]
        #Parse AMP string values back into an array
        ampString = row[0]
        ampFloat = parseAmp(ampString)
        
        #Convert values to log scale
        ampLog = numpy.log10(ampFloat)
        #Apply median filter
        ampMedian = medianAmp(ampLog)
        #Normalize with respect to a steady-state
        ampNormalized = normalizedAmp(ampMedian)
        #Select narrower range of -10:30
        ampNarrowSample = ampNormalized[30:70]
        #normalize between 0 and 1
        min = numpy.amin(ampNarrowSample)
        max = numpy.amax(ampNarrowSample)
        ampFeatures = (ampNarrowSample - min)/(max - min)
        ampFeatures = ampFeatures.astype(numpy.float32)
        ampFeatures = ampFeatures.reshape(1,1,1,40)
        result = p.run([ampFeatures])
        event = numpy.argmax(result)
        cursor.execute('''UPDATE DATATABLE SET EVENT=? WHERE TAG=?''', (event, tag))
        print("Updated Table successfully")
    cursor.close()
    conn.commit()

def eventParameters():
    conn = sqlite3.connect('test.db')
    print "Opened database successfully";
    cursor = conn.cursor()
    data = cursor.execute('''SELECT * FROM DATATABLE WHERE EVENT=1''')
    data = data.fetchall()
    for row in data:
        tag = row[8]
        data = normalizedAmp(medianAmp(numpy.log10(parseAmp(row[0]))))
        data = data[30:70]
    	polarity = numpy.median(data[10:20]) - numpy.median(data[5:10])
	eventvalues = data[12:25]
        polarity = numpy.sign(polarity)
        print "Polarity: %f" %polarity
	if (polarity > 0):
	    peak = numpy.argmax(eventvalues)+2
        else:
	    peak = numpy.argmin(eventvalues)+2
	print "Peak at %d seconds" %peak
	if (peak > 10):
            print "Long Recovery"
            bestLine = numpy.polyfit(range(peak-5,peak+10), data[peak+5:peak+20], 1)
        else:
            bestLine = numpy.polyfit(range(peak,peak+15),data[10+peak:peak+25],1)
        print "Slope: %f" %bestLine[0]
        cursor.execute('''UPDATE DATATABLE SET Polarity=?, EventPeak=?, EventSlope=? WHERE TAG=?''', (polarity, peak, bestLine[0], tag))
        print("Updated Table successfully")
    cursor.close()
    conn.commit()

def createEventTable():
    conn = sqlite3.connect('test.db')
    print "Opened database successfully";
    cursor = conn.execute("SELECT AMP from DATATABLE WHERE Event=1")
    eventTable = numpy.zeros(40)
    for row in cursor:
        ampData = normalizedAmp(medianAmp(numpy.log10(parseAmp(row[0]))))
        ampData = ampData[30:70]
        #min = numpy.amin(ampData)
        #max = numpy.amax(ampData)
        #ampData = (ampData - min)/(max - min)
        #ampData = ampData.astype(numpy.float32)        
        
        eventTable = numpy.vstack((eventTable, ampData))

    print "Operation done successfully";
    conn.close()
    numpy.savetxt("EventTable2.csv", eventTable, delimiter =",")
    print(eventTable.shape)

def parseAmp(ampBytes):
    ampData = numpy.frombuffer(ampBytes[0][0], dtype = numpy.float32)
    return ampData

def medianAmp(ampLog):
    #timeRange = list(range(-40,120))
    ampArray = numpy.asarray(ampLog)
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
    #timeRange = list(range(-40,120))
    steadyState = numpy.median(ampMedian[20:40])
    ampNormalized = ampMedian / steadyState
    #p.subplot(211)
    #p.plot(timeRange, ampMedian)
    #p.subplot(212)
    #p.plot(timeRange, ampNormalized)
    #p.show()
    return ampNormalized
    
def selectSample(tag):
    conn = sqlite3.connect('test.db')
    cursor = conn.execute("SELECT AMP FROM DATATABLE WHERE TAG = '" + tag + "'")
    row = cursor.fetchall()
    dataString = row[0][0]
    return dataString

def dataTest(tag):
    dataRaw = parseAmp(selectSample(tag))
    dataLog = numpy.log10(dataRaw)
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
      
