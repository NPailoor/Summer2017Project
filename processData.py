import sqlite3
import numpy
from  matplotlib import pyplot as p
from scipy import signal
from caffe2.python import workspace
import math

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
        #Normalize between 0 and 1
        ampNormalized = normalizedAmp(ampMedian)
        #Select narrower range of -10:30
        ampNarrowSample = ampNormalized[30:70]
        #normalize between 0 and 1
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

def distanceParameters():
    conn = sqlite3.connect('test.db')
    print "Opened database successfully";
    cursor = conn.cursor()
    data = cursor.execute('''SELECT * FROM DATATABLE WHERE EVENT=1''')
    data = data.fetchall()
    for row in data:
        tag = row[8]
        rec = row[1]
        lat1 = row[5]
        lon1 = row[6]
        (lat2, lon2) = stationCoordinates(rec)
        d2rec = distance(lat1, lon1, lat2, lon2)
        cursor.execute('''UPDATE DATATABLE SET d2rec=? WHERE TAG=?''', (d2rec, tag))
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
    ampData = numpy.frombuffer(ampBytes, dtype = numpy.float32)
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

def fixCoordinates():
    import os
    import scipy.io
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    data = cursor.execute('''SELECT * FROM DATATABLE''')
    data = data.fetchall()
    for row in data:
        date = row[3]
        time = row[4]
        tag = row[8]
        path = '/home/nikhil/Desktop/geniza/Morris/EventsEF/'
        dateString = str(date/10000) + '_' + str((date%10000)/100).zfill(2) + '_' + str(date%100).zfill(2)
        fileString = str(date/10000) + '-' + str((date%10000)/100).zfill(2) + '-' + str(date%100).zfill(2) \
                     + '-' + str(time/3600).zfill(2) + '-' +str((time%3600)/60).zfill(2) + '-' + str(time%60) \
                     + '-' + str(abs(int(round(row[6]))));
        fileStringalt = str(date/10000) + '-' + str((date%10000)/100).zfill(2) + '-' + str(date%100).zfill(2) \
                     + '-' + str(time/3600).zfill(2) + '-' +str((time%3600)/60).zfill(2) + '-' + \
                     str(time%60 - 1).zfill(2) + '-' + str(abs(int(round(row[6]))));
        print dateString
        print fileString
        for i in  os.listdir(path + dateString):
            if os.path.isfile(os.path.join(path + dateString,i)) and \
               ((fileString in i) or (fileStringalt in i)):
                print("successfully accessed file")
                incident = scipy.io.loadmat(path + '/' + dateString + '/' + i)
                info = incident['LightningInfo']
                info = info[0]
                lat = info[6]
                lon = info[7]
                peak = info[8]
                cursor.execute('''UPDATE DATATABLE SET LAT=?, LONG=?, PEAK=? WHERE TAG=?''',\
                               (lat, lon, peak, tag))
                print("Table updated successfully")
    cursor.close()
    conn.commit()
    
    
             
            


        
    

def normalizedAmp(ampMedian):
    #timeRange = list(range(-40,120))
    min = numpy.amin(ampMedian)
    max = numpy.amax(ampMedian)
    ampNormalized = (ampMedian - min)/(max - min)
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
    dataRaw = dataRaw[30:70]
    dataLog = numpy.log10(dataRaw)
    dataMedian = medianAmp(dataLog)
    dataNormal = normalizedAmp(dataMedian)
    timeRange = list(range(-10,30))
    p.subplot(411)
    p.plot(timeRange, dataRaw)
    p.title('Raw data')
    p.subplot(412)	
    p.plot(timeRange, dataLog)
    p.title('Log scale raw data')
    p.subplot(413)
    p.plot(timeRange, dataMedian)
    p.title('Median filtered log scale data')
    p.subplot(414)
    p.plot(timeRange, dataNormal)
    p.title('Fully normalized data')
    p.show()

def distance(lat1, long1, lat2, long2):
    if (lat2 > 90):
        return 0
    lat1 = math.radians(lat1)
    long1 = math.radians(long1)
    lat2 = math.radians(lat2)
    long2 = math.radians(long2)

    angle = math.acos(math.sin(lat1)*math.sin(lat2) \
                      + math.cos(lat1)*math.cos(lat2)*math.cos(long1-long2))
    angle = math.degrees(angle)
    #Distance in nautical miles
    distance = 60.0 * angle
    return distance

def stationCoordinates(station):
    if station == "NAA":
        return(44 + (38/60) + (47.02/3600), -1*(67 + (16/60) + (51.85/3600)))
    elif station == "NAU":
        return (18.398775, -67.177486)
    elif station == "NML":
        return (46.365987, -98.335667)
    elif station == "NPM":
        return (-21.816325, 114.16546)
    elif station == "NRK":
        return (63.850365, -22.466773)
    elif station == "DHO":
        return (53.078900, 7.615000)
    elif (station == "HWU" or station == "HW2" or station == "HW3"):
        return (46.714119, 1.244309)
    elif station == "NLK":
        return (48.203633, 121.916828)
    elif station == "GBZ":
        return (52.72246, -3.06295)
    elif station == "GQD":
        return (54.911, -3.280)
    elif station == "JJI":
        return (32 + (4/60) + (36/3600), -1*(130 + (49/60) + (44/3600)))
    elif station == "ICV":
        return (40.923127, 9.731011)
    elif station == "HWV":
        return (48.544, 2.576)
    elif station == "JXN":
        return (59.910, 10.520)
    elif station == "Burden":
        return (37 + (18/60) + (51/3600), -1*(96 + (45/60) + (19/3600)))
    elif station == "Baxley":
        return (31 + (46/60) + (31/3600), -1*(82 + (20/60) + (51/3600)))
    elif station == "Briarwood":
        return (33 + (34/60) + (50/3600), -1*(84 + (26/60) + (37/3600)))
    elif station == "Atlanta":
        return (33 + (46/60) + (33/2600), -1*(84 + (23/60) + (41/3600)))
    elif station == "Deleware":
        return (39 + (9/60) + (29/3600), -1*(75 + (31/60) + (28/3600)))
    elif station == "PARI":
        return (35.1996, -82.8724)
    elif station == "Toolik":
        return (68 + (38/60), -1*(149 + (36/60)))
    elif station == "Juneau":
        return (58.3014485, -134.4216125)
    else:
        return (91,181)

        
    
    

      
