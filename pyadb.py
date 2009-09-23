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
	"""error to indicate that a method has been called with incorrect args"""
	def __init__(self, msg):
		self.msg = msg
class ConfigWarning(Warning):
	def __init__(self, msg):
		self.msg = msg

class Pyadb(object):
	"""Pyadb class.  Allows for creation, access, insertion and query of an audioDB vector matching database."""
	validConfigTerms = {"seqLength":int, "seqStart":int, "exhaustive":bool, 
		"falsePositives":bool, "accumulation":str, "distance":str, "npoints":int,
		"ntracks":int, "includeKeys":list, "excludeKeys":list, "radius":float, "absThres":float,
		"relThres":float, "durRatio":float, "hopSize":int, "resFmt":str}
	def __init__(self, path, mode='w'):
		self.path = path
		self.configQuery = {}
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
		#While python style normally advocates leaping before looking, these check are nessecary as 
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
	
	def configCheck(self, scrub=False):
		"""examine self.configQuery dict.  For each key encouters confirm it is in the validConfigTerms list and if appropriate, type check.  If scrub is False, leave unexpected keys and values alone and return False, if scrub  try to correct errors (attempt type casts and remove unexpected entries) and continue.  If self.configQuery only contains expected keys with correctly typed values, return True.  See Pyadb.validConfigTerms for allowed keys and types.  Note also that include/exclude key lists memebers or string switched are not verified here, but rather when they are converted to const char * in the C api call and if malformed, an error will be rasied from there.  Valid keys and values in  queryconfig:
		{seqLength    : Int Sequence Length, \n\
		seqStart      : Int offset from start for key, \n\
		exhaustive    : boolean - True for exhaustive (false by default),\n\
		falsePositives: boolean - True to keep fps (false by defaults),\n\
		accumulation  : [\"db\"|\"track\"|\"one2one\"] (\"db\" by default),\n\
		distance      : [\"dot\"|\"eucNorm\"|\"euclidean\"] (\"dot\" by default),\n\
		npoints       : int number of points per track,\n\
		ntracks       : max number of results returned in db accu mode,\n\
		includeKeys   : list of strings to include (use all by default),\n\
		excludeKeys   : list of strings to exclude (none by default),\n\
		radius        : double of nnRadius (1.0 default, overrides npoints if specified),\n\
		absThres      : double absolute power threshold (db must have power),\n\
		relThres      : double relative power threshold (db must have power),\n\
		durRatio      : double time expansion/compresion ratio,\n\
		hopSize       : int hopsize (1 by default)])->resultDict\n\
		resFmt        : [\"list\"|\"dict\"](\"dict\" by default)}"""
		for key in self.configQuery.keys():
			if key not in Pyadb.validConfigTerms.keys():
				if not scrub: return False
				del self.configQuery[key]
			if not isinstance(self.configQuery[key], Pyadb.validConfigTerms[key]):
				if not scrub: return False
				self.configQuery[key] = Pyadb.validConfigTerms[key](self.configQuery[key])#hrm, syntax?
		return True	
				
				# 
	
	def query(self, key=None, featData=None, strictConfig=False):
		"""query the database.  Query parameters as defined in self.configQuery. For details on this consult the doc string in the configCheck method."""
		if not self.configCheck():
			if strictConfig:
				raise ValueError("configQuery dict contains unsupported terms and strict configure mode is on.\n\
Only keys found in Pyadb.validConfigTerms may be defined")
			else:
				raise ConfigWarning("configQuery dict contains unsupported terms and strict configure mode is off.\n\
Only keys found in Pyadb.validConfigTerms should be defined.  Removing invalid terms and proceeding...")
				self.configCheck(scrub=True)
		if ((not key and not featData) or (key and featData)):
			raise Usage("query require either key or featData to be defined, you have defined both or neither.")
		if key:
			result = _pyadb._pyadb_queryFromKey(self._db, key, **self.configQuery)
		elif featData:
			raise NotImplementedError("direct data query not yet implemented.  Sorry.")
		return Pyadb.Result(result, self.configQuery)
	
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
		
	class Result(object):
		def __init__(self, rawData, currentConfig):
			self.rawData = rawData
			if "resFmt" in currentConfig:
				self.type = currentConfig["resFmt"]
			else:
				self.type = "dict"
		def __str__(self):
			str(self.rawData)
		def __repr__(self):
			repr(self.rawData)

class untitledTests(unittest.TestCase):
	def setUp(self):
		pass


if __name__ == '__main__':
	unittest.main()
