all:
	python setup.py build

test: ../../libaudioDB.so.0.0 all
	(cd tests && \
	 env PYTHONPATH=../build/lib.linux-`uname -m`-2.5 \
		LD_LIBRARY_PATH=../../.. \
		python InitialisationRelated.py)

clean:
	rm -rf tests/test* pyadb.pyc build
