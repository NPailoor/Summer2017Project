def test():
	print 'hello'

def findData(event):
	import scipy.io
	import sqlite3
	import math
	import array
	import numpy
	import pickle
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
		filePath = ('geniza/NarrowbandFull/' + sites[i][0][0] + '/' + str(int(year)) + '_' + str(int(month)).zfill(2)
		+ '_' + str(int(day)) + '/' + tags[i][0][0] +  transmitters[i][0][0] + '_000A.mat')
		print filePath
		fullData = scipy.io.loadmat(filePath, variable_names='data')
		fullData = fullData['data']
		#Unique identifying tags
		primeKey = str(int(location)) + str(dateTime) + str(sites[i][0][0]) + str(transmitters[i][0][0])
		#Matlab is causing this list to be structured as a list of single-element lists
		#Needs to be reformatted
		dataSample = fullData[start:end]
		dataRow = []
		#Some datasets contain null elements, if this is the case ignore them
		dataIsReal = True
		dataString = ""
		delim = '-'
		for j in range(0, dataSample.size):
			if numpy.isnan(dataSample[j]):
				dataIsReal = False
			dataString = dataString + str(dataSample[j][0]) + delim
		print "INSERT INTO DATATABLE (AMP,REC,TRAN,DATE,TIME,LAT,LONG,PEAK,TAG) \
			VALUES ('" + dataString + "','" + str(sites[i][0][0]) + "','" + str(transmitters[i][0][0]) \
			+ "','" + str(date) + "'," + str(time) + "," + \
			str(info[7]) + "," + str(info[8]) + "," + str(info[9]) + ",'" + primeKey + "')"
		if dataIsReal:
			conn.execute("INSERT INTO DATATABLE (AMP,REC,TRAN,DATE,TIME,LAT,LONG,PEAK, TAG) \
				VALUES ('" + dataString + "','" + str(sites[i][0][0]) + "','" + str(transmitters[i][0][0]) \
				+ "','" + str(date) + "'," + str(time) + "," + \
				str(info[7]) + "," + str(info[8]) + "," + str(info[9]) + ",'" + primeKey + "')");
			conn.commit()
			print "Records created successfully";

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
