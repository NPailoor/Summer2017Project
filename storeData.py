def test():
	print 'hello'

def findData(event):
	import scipy.io
	import sqlite3
	import math
	import numpy
	import os.path
	conn = sqlite3.connect('test.db')
	info = event['LightningInfo']
	info = info[0]
	time = info[3]*3600 + info[4]*60 + round(info[5])
	time = int(time)
	start = time - int(event['WindowBefore'])
	end = time + int(event['WindowAfter'])
	sites = event['MatchingSiteNamesEF']
	tags = event['MatchingFileNameTimeStringsEF']
	transmitters = event['MatchingCallSignsEF']
	year = int(info[0])
	month = int(info[1])
	day = int(info[2])
	date = day + 100*month + 10000*year;
	dateTime= date*100000 + time;
	location = abs(info[8]*10000*1000000) + abs(info[7]*10000);
	VLFData = [];
	for i in range(0, sites.size):
                #FilePath can end in 0 or 1
                for k in range(0,2):
                        filePath = ('/home/nikhil/Desktop/geniza/NarrowbandFull/' + sites[i][0][0] + '/' + str(int(year)) + '_' + str(int(month)).zfill(2) + '_' + str(int(day)).zfill(2) + '/' + tags[i][0][0] +  transmitters[i][0][0] + '_00' + str(k) + 'A.mat')
                        if os.path.isfile(filePath):
                                if (k == 0):
                                        dir = 'NS'
                                else:
                                        dir = 'EW'
                                primeKey = str(int(location)) + str(dateTime) + str(sites[i][0][0]) + str(transmitters[i][0][0]) + dir
                                fullData = scipy.io.loadmat(filePath, variable_names='data')
                                fullData = fullData['data']
                                #Unique identifying tags
                                #Matlab is causing this list to be structured as a list of single-element lists
                                #Needs to be reformatted
                                dataSample = fullData[start:end]
                                dataRow = []
                                #Some datasets contain null elements, if this is the case ignore them
                                dataIsReal = True
                                dataSample = numpy.squeeze(dataSample.astype(numpy.float32))
                                dataSample = numpy.asarray(dataSample)
                                dataBytes = dataSample.tobytes()
                                print primeKey
                                #dataString = ""
                                #delim = '-'
                                #Check if nan
                                for j in range(0, dataSample.size):
                                        if numpy.isnan(dataSample[j]):
                                                dataIsReal = False
                                if dataIsReal:
                                        conn.execute("INSERT INTO DATATABLE (AMP,REC,TRAN,DATE,TIME,LAT,LONG,PEAK,TAG,DIR) \
                                        VALUES (?,?,?,?,?,?,?,?,?,?)",(sqlite3.Binary(dataBytes),str(sites[i][0][0]),str(transmitters[i][0][0]),str(date), str(time),str(info[7]),str(info[8]),str(info[9]),primeKey,dir));
                                        conn.commit()
                                        print "Records created successfully"
                                else:
                                        print "Bad data"
                        else:
                                print filePath
                                print "File Does Not Exist"
        conn.close()
                

def lightningScan():
        import os
        import scipy.io
        filePath = '/home/nikhil/Desktop/geniza/Morris/EventsEF'
        date = 20160911
        for dirname in os.listdir(filePath):
                dateString = dirname[0:4] + dirname[5:7] + dirname[8:10]
                if (int(dateString) < date):
                        print "starting search"
                        for filename in os.listdir(filePath + '/' + dirname):
                                event = scipy.io.loadmat(filePath + '/' + dirname + '/' + filename)
                                findData(event)
                        
def createTables():
	import sqlite3
	conn = sqlite3.connect('test.db')
	print "Opened database successfully";

	conn.execute('''CREATE TABLE DATATABLE
		(AMP BLOB NOT NULL,
		REC TEXT NOT NULL,
		TRAN TEXT NOT NULL,
		DATETIME INT PRIMARY KEY NOT NULL,
		TAG TEXT NOT NULL
		LAT REAL NOT NULL
		LONG REAL NOT NULL);''')
	print "Table created successfully";

	conn.execute('''CREATE TABLE LIGHTNINGTABLE
		(Tag INT PRIMARY KEY NOT NULL,
		DATE INT NOT NULL,
		TIME INT NOT NULL,
		LAT REAL NOT NULL,
		LONG REAL NOT NULL,
		PEAK REAL NOT NULL);''')
	print "Table created successfully";
	conn.close()


if __name__=='__main__':
	test()
	findData("exampleData")
