#!/usr/bin/env python
# encoding: utf-8
"""
pyadb.py

public access and class structure for python audioDb api bindings.

Created by Benjamin Fields on 2009-09-22.
Copyright (c) 2009 Goldsmith University of London.
"""

import sys
import os, os.path
import unittest
import _pyadb

ADB_HEADER_FLAG_L2NORM = 0x1#annoyingly I can't find a means
ADB_HEADER_FLAG_POWER = 0x4#around defining these flag definitions 
ADB_HEADER_FLAG_TIMES = 0x20#as they aren't even exported to the 
ADB_HEADER_FLAG_REFERENCES = 0x40#api, so this is the only way to get them.

class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg

class Pyadb:
	
	def __init__(self, path, mode='w'):
		self.path = path
		if not (mode=='w' or mode =='r'):
			raise(ValueError, "if specified, mode must be either\'r\' or \'w\'.")
		if os.path.exists(path):
			self._db = _pyadb._pyadb_open(path, mode)
		else:
			self._db = _pyadb._pyadb_create(path,0,0,0)
		self._updateDBAttributes()
		return
	
	def insert(self, featFile=None, powerFile=None, timesFile=None, featData=None, powerData=None, timesData=None, key=None):
		"""Insert features into database.  Can be done with data provided directly or by giving a path to a binary fftExtract style feature file.  If power and/or timing is engaged in the database header, it must be provided (via the same means as the feature) or a Usage exception will be raised. Power files should be of the same binary type as features.  Times files should be the ascii number length of time in seconds from the begining of the file to segment start, one per line.\n---Note that direct data insertion is not yet implemented.---"""
		#While python normally advocates leaping before looking, these check are nessecary as 
		#it is very difficult to assertain why the insertion failed once it has been called.
		if (self.hasPower and (((featFile) and powerFile==None) or ((featData) and powerData==None))):
			raise(Usage, "The db you are attempting an insert on (%s) expects power and you either\
 haven't provided any or have done so in the wrong format."%self.path)
		if (self.hasTimes and (((timesFile) and timesFile==None) or ((timesData) and timesData==None))):
			raise(Usage, "The db you are attempting an insert on (%s) expects times and you either\
 haven't provided any or have done so in the wrong format."%self.path)
		args = {"db":self._db}
		if featFile:
			args["features"] = featFile
		elif featData:
			args["features"] = featData
		else:
			raise(Usage, "Must provide some feature data!")
		if self.hasPower:
			if featFile:
				args["power"]=powerFile
			elif featData:
				pass
		if self.hasTimes:
			if featFile:
				args["times"]=timesFile
			elif timesData:
				pass
		if key:
			args["key"]=str(key)
		if featFile:
			if not _pyadb._pyadb_insertFromFile(**args):
				raise(RuntimeError, "Insertion failed for an unknown reason.")
			else:
				self._updateDBAttributes()
				return
		elif featData:
			raise(NotImplementedError, "direct data insertion not yet implemented")
			
	
	###internal methods###
	def _updateDBAttributes(self):
		'''run _pyadb_status to fill/update the database level flags and info'''
		rawFlags = long(0)
		(self.numFiles, 
		self.dims, 
		self.dudCount, 
		self.nullCount, 
		rawFlags, 
		self.length, 
		self.data_region_size) = _pyadb._pyadb_status(self._db)
		self.l2Normed = bool(rawFlags & ADB_HEADER_FLAG_L2NORM)
		self.hasPower = bool(rawFlags & ADB_HEADER_FLAG_POWER)
		self.hasTimes = bool(rawFlags & ADB_HEADER_FLAG_TIMES)
		self.usesRefs = bool(rawFlags & ADB_HEADER_FLAG_REFERENCES)
		return
		
	
		
		
	class Result:
		def __init__(self):
			pass

class untitledTests(unittest.TestCase):
	def setUp(self):
		pass


if __name__ == '__main__':
	unittest.main()