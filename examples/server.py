#!/usr/bin/python

import _pyadb
import web
import json
import sys
import getopt

# DB Path goes here for now!
dbPath = "9.adb"

urls = (
	'/', 'index',
	'/status', 'status',
	'/query', 'query'
)

app = web.application(urls, globals())
class index:
	def GET(self):
		return """
<html>
<body>
<ul>
<li><a href="/status">Status</a></li>
<li><a href="/query">Query</a></li>
</ul>
</body>
</html>"""


class status:
	def GET(self):
		db = _pyadb._pyadb_open(dbPath, "r")
		status = _pyadb._pyadb_status(db)
		results = dict(zip(["numFiles", "dim", "dudCount", "nullCount", "flags", "length", "data_region_size"], status))
		return json.dumps(dict(status = "ok", data = results))

class query:
	def GET(self):
		params = web.input(key="", ntracks=100, seqStart=0, seqLength=16, npoints=1, radius=1.0, hopSize=1)
		results = dict()
		db = _pyadb._pyadb_open(dbPath, "r")

		params.ntracks = int(params.ntracks)
		params.npoints = int(params.npoints)
		params.seqStart = int(params.seqStart)
		params.seqLength = int(params.seqLength)
		params.radius = float(params.radius)
		params.hopSize = int(params.hopSize)
		
		try:
			results = _pyadb._pyadb_queryFromKey(db, **params) 
		except Exception as inst:
			return json.dumps(dict(status = "error", message=str(inst)))

		return json.dumps(dict(status = "ok", data = results))

if __name__ == "__main__": 
	app.run()
