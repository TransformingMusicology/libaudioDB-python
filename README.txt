README.txt

to install, try:

python setup.py build
python setup.py install

Notes: 
 - a compatible build of audioDB (>=r914) needs to be linkable at runtime
 - currently only the direct C-api exposure layer is visible, so the calling semantics are a bit gross


The actual query call is a bit of a mess, but will be more intuitive from the native python layer (to be written)...
so the python bindings now have a complete path:
	>>import _pyadb
	>>aDB = _pyadb._pyadb_create("test.adb", 0,0,0)
	>>_pyadb._pyadb_status(aDB)
	>>_pyadb._pyadb_insertFromFile(aDB, "someFeats.mfcc12")
		...(add some more data)
	>>result = _pyadb._pyadb_queryFromKey(aDB, "a Key in aDB", [options])
	
	and then result has a nice dict of your results.  



21 September 2009, Ben Fields, b.fields@gold.ac.uk