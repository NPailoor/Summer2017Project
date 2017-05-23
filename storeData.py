def test():
	print 'hello'

def findData(event):
	import scipy.io
	import sqlite3
	import math
	import array
	conn = sqlite3.connect('test.db')
	info = event['LightningInfo']
	info = info[0]
	time = info[3]*3600 + info[4]*60 + round(info[5])
	start = int(time) - int(event['WindowBefore'])
	end = int(time) + int(event['WindowAfter'])
	sites = event['MatchingSiteNamesEF']
	tags = event['MatchingFileNameTimeStringsEF']
	transmitters = event['MatchingCallSignsEF']
	year = info[0]
	month = info[1]
	day = info[2]
	date = day + 100*month + 10000*year;
	dateTime= date*100000 + time;
	location = abs(info[8]*10000*1000000) + abs(info[7]*10000);
	tag = str(dateTime) + str(int(location))
	print tag
	VLFData = [];
	conn.execute("INSERT INTO LIGHTNINGTABLE (TAG,DATE,TIME,LAT,LONG,PEAK) \
		VALUES (" + str(tag) + "," + str(date) + "," + str(dateTime)+ "," + str(info[7]) + "," \
		+ str(info[8]) + "," + str(info[9]) + ")");
	conn.commit()
	print "Records created successfully";
	for i in range(0, sites.size):
		filePath = ('geniza/NarrowbandFull/' + sites[i][0][0] + '/' + str(int(year)) + '_' + str(int(month)).zfill(2)
		+ '_' + str(int(day)) + '/' + tags[i][0][0] +  transmitters[i][0][0] + '_000A.mat')
		print filePath
		fullData = scipy.io.loadmat(filePath, variable_names='data')
		fullData = fullData['data']
		dataRow = fullData[start:end]
		if not math.isnan(dataRow[0]):
			a = array.array('B',fullData)
			amp = a.tostring()
			conn.execute("INSERT INTO DATATABLE (AMP,REC,TRAN,DATE,TAG) \
				VALUES (" + amp + "," + sites[i][0][0] + "," + transmitters[i][0][0] \
				+ "," + date + "," + tag +")");
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
		DATE INT PRIMARY KEY NOT NULL,
		TAG TEXT NOT NULL);''')
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
