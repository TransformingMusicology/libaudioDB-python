// pyadbmodule.c
// 
// the internal portion of the wrapper for audio
// see pyadb.py for the public classes
// 
// Created by Benjamin Fields on 2009-09-04.
// Copyright (c) 2009 Goldsmith University of London. All rights reserved.
// 	
#include <fcntl.h>
#include "Python.h"
#include "audioDB_API.h"
#include "numpy/arrayobject.h"

static void _pyadb_close(void *ptr);


/* create a new database */
/* returns a struct or NULL on failure */
/* api call: */
/* adb_ptr audiodb_create(const char *path, unsigned datasize, unsigned ntracks, unsigned datadim);*/
PyObject * _pyadb_create(PyObject *self, PyObject *args)
{
	unsigned datasize, ntracks, datadim;
	const char *path;
	int ok;
	adb_ptr new_database;
	ok = PyArg_ParseTuple(args, "sIII", &path, &datasize, &ntracks, &datadim);
	if (!ok) return 0;
	new_database = audiodb_create(path, datasize, ntracks, datadim);
	if (!new_database) return 0;
	
	return PyCObject_FromVoidPtr( new_database, _pyadb_close);	
}

/* open an existing database */
/* returns a struct or NULL on failure */
/* flags expects fcntl flags concerning the opening mode */
/* api call: */
// adb_ptr audiodb_open(const char *path, int flags);
PyObject * _pyadb_open(PyObject *self, PyObject *args)
{
	const char *path;
	char mode;
	int ok;//in python layer need to translate boolean flags to byte mask
	adb_ptr fresh_database;
	ok = PyArg_ParseTuple(args, "sc", &path, &mode);
	if (!ok) return 0;
	if (mode == 'r'){
		fresh_database = audiodb_open(path, O_RDONLY);
	}else if (mode == 'w'){
		fresh_database = audiodb_open(path, O_RDWR);
	}else{
		PyErr_SetString(PyExc_ValueError,
		"mode must be either \'r\' or \'w\'.  It appears to be something else.");
		return 0;
	}
	if (!fresh_database) return 0;
	
	return PyCObject_FromVoidPtr( fresh_database, _pyadb_close);	
}

/* database status */  
/* api call: */
// int audiodb_status(adb_ptr mydb, adb_status_ptr status);
PyObject * _pyadb_status(PyObject *self, PyObject *args)
{
	adb_ptr check_db;
	adb_status_ptr status;
	int flags, ok;
	PyObject * incoming = 0;
	status = (adb_status_ptr)malloc(sizeof(struct adbstatus));
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	check_db = (adb_ptr)PyCObject_AsVoidPtr(incoming);
	
	
	flags = audiodb_status(check_db, status);
	return Py_BuildValue("IIIIILL", status->numFiles, 
									status->dim, 
									status->dudCount, 
									status->nullCount, 
									status->flags, 
									status->length, 
									status->data_region_size);

}

/*engage l2norm in the referenced db*/
/*api call:*/
//int audiodb_l2norm(adb_ptr mydb);
PyObject * _pyadb_l2norm(PyObject *self, PyObject *args)
{
	adb_ptr current_db;
	int ok;
	PyObject * incoming = 0;
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	current_db = (adb_ptr)PyCObject_AsVoidPtr(incoming);
	
	ok = audiodb_l2norm(current_db);
	return Py_BuildValue("i", ok);
	
}

/*engage power thresholding in the referenced db*/
/*api call:*/
// int audiodb_power(adb_ptr mydb);
PyObject * _pyadb_power(PyObject *self, PyObject *args)
{
	adb_ptr current_db;
	int ok;
	PyObject * incoming = 0;
	
	ok = PyArg_ParseTuple(args, "O", &incoming);
	if (!ok) return 0;
	current_db = (adb_ptr)PyCObject_AsVoidPtr(incoming);
	
	ok = audiodb_power(current_db);
	return Py_BuildValue("i", ok);
	
}

/* close a database */
/* api call: */
// void audiodb_close(adb_ptr db);
static void _pyadb_close(void *ptr)
{
	adb_ptr stale_database;
	stale_database = (adb_ptr)ptr; 
	
	audiodb_close(stale_database);
}

static PyMethodDef _pyadbMethods[] = 
{
	{ "_pyadb_create", _pyadb_create, METH_VARARGS, 
	  "_pyadb_create(string path, unsigned datasize, unsigned ntracks, unsigned datadim)->adb object"},
	{ "_pyadb_open", _pyadb_open, METH_VARARGS, 
	  "_pyadb_open(string path, [\'r\'|\'w\'])->adb object\nNote that specifing \'w\' opens the file in read and write mode.  \
	There is currently no way to open in write only."},
	{ "_pyadb_status", _pyadb_status, METH_VARARGS,
	  "_status(adb_ptr)->(numFiles, dims, dudCount, nullCount, flags, length, data_region_size)"},
	{ "_pyadb_l2norm", _pyadb_l2norm, METH_VARARGS,
	  "_pyadb_l2norm(adb_ptr)->int return code (0 for sucess)"},
	{ "_pyadb_power", _pyadb_power, METH_VARARGS,
	  "_pyadb_power(adb_ptr)->int return code (0 for sucess)"},
	{NULL,NULL, 0, NULL}
};

void init_pyadb()
{
	Py_InitModule3("_pyadb", _pyadbMethods, "internal c bindings for audioDB.  Use pyadb for pythonic access to adb (when it exists).");
	// import_array();
	return;
}

	
