all:
	python setup.py build

test: ../../libaudioDB.so.0.0 all
	(cd tests && \
	 env PYTHONPATH=$$(python -c 'import distutils; import distutils.util; import sys; print "../build/lib.%s-%s" % (distutils.util.get_platform(), sys.version[0:3])') \
		LD_LIBRARY_PATH=../../.. \
		python InitialisationRelated.py)

clean:
	rm -rf tests/test* pyadb.pyc build
