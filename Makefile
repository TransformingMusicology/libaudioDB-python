all:
	python setup.py build

test:
	env PYTHONPATH=./build/lib.linux-i686-2.5 \
		LD_LIBRARY_PATH=../.. \
		python tests/InitialisationRelated.py