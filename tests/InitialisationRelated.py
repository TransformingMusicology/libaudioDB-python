#!/usr/bin/env python
# encoding: utf-8
"""
InitialisationRelated.py

designed to mirror the numbering for the C/C++ api's unit tests
this performs tests 0001, 0002, 0003


Created by Ben Fields on 2010-01-11.
"""

import sys
import os,os.path
import pyadb
import struct
import unittest


class CreateADBTests(unittest.TestCase):
	def setUp(self):
		self.adb = pyadb.Pyadb("test.adb")
	def test_DBcreation(self):
		self.assert_(os.path.exists(self.adb.path))
		self.assertRaises(TypeError, pyadb.Pyadb)
	def test_DBstatus(self):
		try:
			self.adb.status()
		except:
			self.assert_(False)
	def test_1DinsertionSelfQuery(self):
		tH = open("testfeature", 'w')
		tH.write(struct.pack("=id",1,1))
		tH.close()
		self.adb.insert("testfeature")
		self.adb.configQuery["seqLength"] = 1
		result = self.adb.query("testfeature")
		self.assert_(len(result.rawData) == 1)
		self.assert_(result.rawData.has_key("testfeature"))
		self.assert_(len(result.rawData["testfeature"]) == 1)
		self.assert_(result.rawData["testfeature"][0] == (float("-inf"), 0,0))
		
		


if __name__ == '__main__':
	unittest.main()