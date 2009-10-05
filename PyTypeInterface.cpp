/*

 * Vampy : This plugin is a wrapper around the Vamp plugin API.
 * It allows for writing Vamp plugins in Python.

 * Centre for Digital Music, Queen Mary University of London.
 * Copyright (C) 2008-2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
 * for licence information.)

*/

#include <Python.h>

#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL VAMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#endif

#include "PyTypeInterface.h"
#include "PyRealTime.h"
#include "PyExtensionModule.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::map;

static std::map<std::string, o::eOutDescriptors> outKeys;
static std::map<std::string, p::eParmDescriptors> parmKeys;
static std::map<std::string, eSampleTypes> sampleKeys;
static std::map<std::string, eFeatureFields> ffKeys;
static bool isMapInitialised = false;

/*  Note: NO FUNCTION IN THIS CLASS SHOULD ALTER REFERENCE COUNTS
	(EXCEPT FOR TEMPORARY PYTHON OBJECTS)!  						 */

PyTypeInterface::PyTypeInterface() : 
	m_strict(false),
	m_error(false),
	error(m_error) // const public reference for easy access
{
}

PyTypeInterface::~PyTypeInterface()
{
}

/// floating point numbers (TODO: check numpy.float128)
float 
PyTypeInterface::PyValue_To_Float(PyObject* pyValue) const
{
	// convert float
	if (pyValue && PyFloat_Check(pyValue)) 
		//TODO: check for limits here (same on most systems)
		return (float) PyFloat_AS_DOUBLE(pyValue);
	
	if (pyValue == NULL)
	{
		setValueError("Error while converting float object.",m_strict);
		return 0.0;		
	}
		
	// in strict mode we will not try harder
	if (m_strict) {
		setValueError("Strict conversion error: object is not float.",m_strict);
		return 0.0;
	}

	// convert other objects supporting the number protocol
	if (PyNumber_Check(pyValue))
	{
		PyObject* pyFloat = PyNumber_Float(pyValue); // new ref
		if (!pyFloat)
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			setValueError("Error while converting " + PyValue_Get_TypeName(pyValue) + " object to float.",m_strict);
			return 0.0;
		}
		float rValue = (float) PyFloat_AS_DOUBLE(pyFloat);
		Py_DECREF(pyFloat);
		return rValue;
	}
/*	
	// convert other objects supporting the number protocol
	if (PyNumber_Check(pyValue)) 
	{	
		// PEP353: Py_ssize_t is size_t but signed !
		// This will work up to numpy.float64
		Py_ssize_t rValue = PyNumber_AsSsize_t(pyValue,NULL);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0.0;
		}
		if (rValue > (Py_ssize_t)FLT_MAX || rValue < (Py_ssize_t)FLT_MIN)
		{
			setValueError("Overflow error. Object can not be converted to float.",m_strict);
			return 0.0;
		}
		return (float) rValue;
	}
*/	
    // convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyFloat = PyFloat_FromString(pyValue,NULL);
		if (!pyFloat) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String value can not be converted to float.",m_strict);
			return 0.0;
		}
		float rValue = (float) PyFloat_AS_DOUBLE(pyFloat);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear(); 
			Py_CLEAR(pyFloat);
			setValueError("Error while converting float object.",m_strict);
			return 0.0;
		}
		Py_DECREF(pyFloat);
		return rValue;
	}
	
	// convert the first element of any iterable sequence (for convenience and backwards compatibility)
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			float rValue = this->PyValue_To_Float(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				std::string msg = "Could not convert sequence element to float. ";
				setValueError(msg,m_strict);
				return 0.0;
			}
		}
	}

    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + PyValue_Get_TypeName(pyValue) + " to float is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_Float failed. " << msg << endl;
#endif	
	return 0.0;
}

/// size_t (unsigned integer types)
size_t 
PyTypeInterface::PyValue_To_Size_t(PyObject* pyValue) const
{
	// convert objects supporting the number protocol 
	if (PyNumber_Check(pyValue)) 
	{	
		if (m_strict && !PyInt_Check(pyValue) && !PyLong_Check(pyValue)) 
			setValueError("Strict conversion error: object is not integer type.",m_strict);
		// Note: this function handles Bool,Int,Long,Float
		// speed is not critical in the use of this type by Vamp
		// PEP353: Py_ssize_t is size_t but signed ! 
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0;
		}
		if ((unsigned long)rValue > SIZE_T_MAX || (unsigned long)rValue < 0)
		{
			setValueError("Overflow error. Object can not be converted to size_t.",m_strict);
			return 0;
		}
		return (size_t) rValue;
	}
	
	// in strict mode we will not try harder and throw an exception
	// then the caller should decide what to do with it
	if (m_strict) {
		setValueError("Strict conversion error: object is not integer.",m_strict);
		return 0;
	}
	
	// convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyLong = PyNumber_Long(pyValue);
		if (!pyLong) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String object can not be converted to size_t.",m_strict);
			return 0;
		}
		size_t rValue = this->PyValue_To_Size_t(pyLong);
		if (!m_error) {
			Py_DECREF(pyLong);
			return rValue;
		} else {
			Py_CLEAR(pyLong);
			setValueError ("Error converting string to size_t.",m_strict);
			return 0;
		}
	}
	
	// convert the first element of iterable sequences
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			size_t rValue = this->PyValue_To_Size_t(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to size_t. ",m_strict);
				return 0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to size_t is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_Size_t failed. " << msg << endl;
#endif	
	return 0;
}

/// long and int
long 
PyTypeInterface::PyValue_To_Long(PyObject* pyValue) const
{
	// most common case: convert int (faster)
	if (pyValue && PyInt_Check(pyValue)) {
		// if the object is not NULL and verified, this macro just extracts the value.
		return PyInt_AS_LONG(pyValue);
	} 
	
	// long
	if (PyLong_Check(pyValue)) {
		long rValue = PyLong_AsLong(pyValue);
		if (PyErr_Occurred()) { 
			PyErr_Print(); PyErr_Clear(); 
			setValueError("Error while converting long object.",m_strict);
			return 0;
		}
		return rValue;
	}
	
	if (m_strict) {
		setValueError("Strict conversion error: object is not integer or long integer.",m_strict);
		return 0;
	}
	
	// convert all objects supporting the number protocol
	if (PyNumber_Check(pyValue)) 
	{	
		// Note: this function handles Bool,Int,Long,Float
		// PEP353: Py_ssize_t is size_t but signed ! 
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting integer object.",m_strict);
			return 0;
		}
		if (rValue > LONG_MAX || rValue < LONG_MIN)
		{
			setValueError("Overflow error. Object can not be converted to size_t.",m_strict);
			return 0;
		}
		return (long) rValue;
	}
	
	// convert string
	if (PyString_Check(pyValue))
	{
		PyObject* pyLong = PyNumber_Long(pyValue);
		if (!pyLong) 
		{
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
			setValueError("String object can not be converted to long.",m_strict);
			return 0;
		}
		long rValue = this->PyValue_To_Long(pyLong);
		if (!m_error) {
			Py_DECREF(pyLong);
			return rValue;
		} else {
			Py_CLEAR(pyLong);
			setValueError ("Error converting string to long.",m_strict);
			return 0;
		}
	}
	
	// convert the first element of iterable sequences
	if (PySequence_Check(pyValue) && PySequence_Size(pyValue) > 0) 
	{
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			size_t rValue = this->PyValue_To_Long(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to long. ",m_strict);
				return 0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to long is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_Long failed. " << msg << endl;
#endif	
	return 0;
}


bool 
PyTypeInterface::PyValue_To_Bool(PyObject* pyValue) const
{
	// convert objects supporting the number protocol
	// Note: PyBool is a subclass of PyInt
	if (PyNumber_Check(pyValue)) 
	{	
		if (m_strict && !PyBool_Check(pyValue)) 
			setValueError
			("Strict conversion error: object is not boolean type.",m_strict);

		// Note: this function handles Bool,Int,Long,Float
		Py_ssize_t rValue = PyInt_AsSsize_t(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError ("Error while converting boolean object.",m_strict);
		}
		if (rValue != 1 && rValue != 0)
		{
			setValueError ("Overflow error. Object can not be converted to boolean.",m_strict);
		}
		return (bool) rValue;
	}
	
	if (m_strict) {
		setValueError ("Strict conversion error: object is not numerical type.",m_strict);
		return false;
	}
	
	// convert iterables: the rule is the same as in the interpreter:
	// empty sequence evaluates to False, anything else is True
	if (PySequence_Check(pyValue)) 
	{
		return PySequence_Size(pyValue)?true:false;
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to boolean is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_Bool failed. " << msg << endl;
#endif	
	return false;
}

/// string and objects that support .__str__() 
/// TODO: check unicode objects
std::string 
PyTypeInterface::PyValue_To_String(PyObject* pyValue) const
{
	// convert string
	if (PyString_Check(pyValue)) 
	{	
		char *cstr = PyString_AS_STRING(pyValue);
		if (!cstr) 
		{
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			setValueError("Error while converting string object.",m_strict);
			return std::string();
		}
		return std::string(cstr);
	}
	// TODO: deal with unicode here (argh!)
	
	// in strict mode we will not try harder
	if (m_strict) {
		setValueError("Strict conversion error: object is not string.",m_strict);
		return std::string();
	}
	
	// accept None as empty string
	if (pyValue == Py_None) return std::string();
			
	// convert list or tuple: empties are turned into empty strings conventionally
	if (PyList_Check(pyValue) || PyTuple_Check(pyValue)) 
	{
		if (!PySequence_Size(pyValue)) return std::string();
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			std::string rValue = this->PyValue_To_String(item);
			if (!m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				setValueError("Could not convert sequence element to string.",m_strict);
				return std::string();
			}
		}
	}

	// convert any other object that has .__str__() or .__repr__()
	PyObject* pyString = PyObject_Str(pyValue);
	if (pyString && !PyErr_Occurred())
	{
		std::string rValue = this->PyValue_To_String(pyString);
		if (!m_error) {
			Py_DECREF(pyString);
			return rValue;
		} else {
			Py_CLEAR(pyString);
			std::string msg = "Object " + this->PyValue_Get_TypeName(pyValue) +" can not be represented as string. ";
			setValueError (msg,m_strict);
			return std::string();
		}
	}

	// give up
	PyErr_Print(); PyErr_Clear();
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to string is not possible.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_String failed. " << msg << endl;
#endif	
	return std::string();
}

/*			 			C Values to Py Values				  		*/


PyObject*
PyTypeInterface::PyValue_From_CValue(const char* cValue) const
{
	// returns new reference
#ifdef _DEBUG
	if (!cValue) {
		std::string msg = "PyTypeInterface::PyValue_From_CValue: Null pointer encountered while converting from const char* .";
		cerr << msg << endl;
		setValueError(msg,m_strict);
		return NULL;
	}
#endif
	PyObject *pyValue = PyString_FromString(cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from char* or string.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_From_CValue: Interpreter failed to convert from const char*" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeInterface::PyValue_From_CValue(size_t cValue) const
{
	// returns new reference
	PyObject *pyValue = PyInt_FromSsize_t((Py_ssize_t)cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from size_t.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_From_CValue: Interpreter failed to convert from size_t" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeInterface::PyValue_From_CValue(double cValue) const
{
	// returns new reference
	PyObject *pyValue = PyFloat_FromDouble(cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from float or double.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_From_CValue: Interpreter failed to convert from float or double" << endl;
#endif
		return NULL;
	}
	return pyValue;
}

PyObject*
PyTypeInterface::PyValue_From_CValue(bool cValue) const
{
	// returns new reference
	PyObject *pyValue = PyBool_FromLong((long)cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from bool.",m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_From_CValue: Interpreter failed to convert from bool" << endl;
#endif
		return NULL;
	}
	return pyValue;
}


/*			 			Sequence Types to C++ Types	    		  	*/

//convert Python list to C++ vector of strings
std::vector<std::string> 
PyTypeInterface::PyValue_To_StringVector (PyObject *pyList) const 
{
	
	std::vector<std::string> Output;
	std::string ListElement;
	PyObject *pyString = NULL;
	
	if (PyList_Check(pyList)) {

		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
			//Get next list item (Borrowed Reference)
			pyString = PyList_GET_ITEM(pyList,i);
			ListElement = (string) PyString_AsString(PyObject_Str(pyString));
			Output.push_back(ListElement);
		}
		return Output;
	}
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_StringVector: Warning: Value is not list of strings." << endl;
#endif

	/// Assume a single value that can be casted as string 
	/// this allows to write e.g. Feature.label = 5.2 instead of ['5.2']
	Output.push_back(PyValue_To_String(pyList));
	if (m_error) {
		std::string msg = "Value is not list of strings nor can be casted as string. ";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_To_StringVector failed. " << msg << endl;
#endif
	}
	return Output;
}

//convert PyFeature.value (typically a list or numpy array) to C++ vector of floats
std::vector<float> 
PyTypeInterface::PyValue_To_FloatVector (PyObject *pyValue) const 
{

#ifdef HAVE_NUMPY
	// there are four types of values we may receive from a numpy process:
	// * a python scalar, 
	// * an array scalar, (e.g. numpy.float32)
	// * an array with nd = 0  (0D array)
	// * an array with nd > 0

	/// check for scalars
	if (PyArray_CheckScalar(pyValue) || PyFloat_Check(pyValue)) {

		std::vector<float> Output;

		// we rely on the behaviour the scalars are either floats
		// or support the number protocol
		// TODO: a potential optimisation is to handle them directly
		Output.push_back(PyValue_To_Float(pyValue));
		return Output;
	}

	/// numpy array
	if (PyArray_CheckExact(pyValue)) 
		return PyArray_To_FloatVector(pyValue);

#endif

	/// python list of floats (backward compatible)
	if (PyList_Check(pyValue)) {
		return PyList_To_FloatVector(pyValue);
	}

	std::vector<float> Output;
	
	/// finally assume a single value supporting the number protocol 
	/// this allows to write e.g. Feature.values = 5 instead of [5.00]
	Output.push_back(PyValue_To_Float(pyValue));
	if (m_error) {
		std::string msg = "Value is not list or array of floats nor can be casted as float. ";
		setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_FloatVector failed." << msg << endl;
#endif
	}
	return Output;
}

//convert a list of python floats
std::vector<float> 
PyTypeInterface::PyList_To_FloatVector (PyObject *inputList) const 
{
	std::vector<float> Output;
	
#ifdef _DEBUG
	// This is a low level function normally called from 
	// PyValue_To_FloatVector(). Checking for list is not required.
	if (!PyList_Check(inputList)) {
		std::string msg = "Value is not list.";
		setValueError(msg,true);
		cerr << "PyTypeInterface::PyList_To_FloatVector failed. " << msg << endl;
		return Output; 
	} 
#endif

	float ListElement;
	PyObject *pyFloat = NULL;
	PyObject **pyObjectArray = PySequence_Fast_ITEMS(inputList);

	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(inputList); ++i) {

		// pyFloat = PyList_GET_ITEM(inputList,i);
		pyFloat = pyObjectArray[i];

#ifdef _DEBUG
		if (!pyFloat) {
			if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
			cerr << "PyTypeInterface::PyList_To_FloatVector: Could not obtain list element: " 
			<< i << " PyList_GetItem returned NULL! Skipping value." << endl;
			continue;
		}
#endif		

		// ListElement = (float) PyFloat_AS_DOUBLE(pyFloat);
		ListElement = PyValue_To_Float(pyFloat);
		

#ifdef _DEBUG_VALUES
		cerr << "value: " << ListElement << endl;
#endif
		Output.push_back(ListElement);
	}
	return Output;
}

#ifdef HAVE_NUMPY
std::vector<float> 
PyTypeInterface::PyArray_To_FloatVector (PyObject *pyValue) const 
{
	std::vector<float> Output;
	
#ifdef _DEBUG
	// This is a low level function, normally called from 
	// PyValue_To_FloatVector(). Checking the array here is not required.
	if (!PyArray_Check(pyValue)) {
		std::string msg = "Object has no array interface.";
		setValueError(msg,true);
		cerr << "PyTypeInterface::PyArray_To_FloatVector failed. " << msg << endl;
		return Output; 
	} 
#endif

	PyArrayObject* pyArray = (PyArrayObject*) pyValue;
	PyArray_Descr* descr = pyArray->descr;
	
	/// check raw data and descriptor pointers
	if (pyArray->data == 0 || descr == 0) {
		std::string msg = "NumPy array with NULL data or descriptor pointer encountered.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyArray_To_FloatVector failed. Error: " << msg << endl;
#endif		
		return Output;
	}

	/// check dimensions
	if (pyArray->nd != 1) {
		std::string msg = "NumPy array must be a one dimensional vector.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyArray_To_FloatVector failed. Error: " << msg << " Dims: " << (int) pyArray->nd << endl;
#endif	
		return Output;
	}

#ifdef _DEBUG_VALUES
	cerr << "PyTypeInterface::PyArray_To_FloatVector: Numpy array verified." << endl;
#endif
	
	/// check strides (useful if array is not continuous)
	size_t strides =  *((size_t*) pyArray->strides);
    
	/// convert the array
	switch (descr->type_num)
	{
		case NPY_FLOAT : // dtype='float32'
			return PyArray_Convert<float,float>(pyArray->data,pyArray->dimensions[0],strides);
		case NPY_DOUBLE : // dtype='float64'
			return PyArray_Convert<float,double>(pyArray->data,pyArray->dimensions[0],strides);
		case NPY_INT : // dtype='int'
			return PyArray_Convert<float,int>(pyArray->data,pyArray->dimensions[0],strides);
		case NPY_LONG : // dtype='long'
			return PyArray_Convert<float,long>(pyArray->data,pyArray->dimensions[0],strides);
		default :
			std::string msg = "Unsupported value type in NumPy array object.";
			setValueError(msg,m_strict);
#ifdef _DEBUG
			cerr << "PyTypeInterface::PyArray_To_FloatVector failed. Error: " << msg << endl;
#endif			
			return Output;
	}
}
#endif


/// FeatureSet (an integer map of OutputLists)
Vamp::Plugin::FeatureSet
PyTypeInterface::PyValue_To_FeatureSet(PyObject* pyValue) const
{
	Vamp::Plugin::FeatureSet rFeatureSet;

	/// Convert PyFeatureSet 
	if (PyFeatureSet_CheckExact(pyValue)) {

		Py_ssize_t pyPos = 0;
		PyObject *pyKey, *pyDictValue; // Borrowed References
		int key;
		// bool it_error = false;

		m_error = false;
		while (PyDict_Next(pyValue, &pyPos, &pyKey, &pyDictValue))
		{
			key = (int) PyInt_AS_LONG(pyKey);
#ifdef _DEBUG_VALUES			
			cerr << "key: '" << key << "' value: '" << PyValue_To_String(pyDictValue) << "' " << endl;
#endif			
			// DictValue -> Vamp::FeatureList
			PyValue_To_rValue(pyDictValue,rFeatureSet[key]);
			if (m_error) {
				// it_error = true;
				lastError() << " in output number: " << key;
			}
		}
		// if (it_error) m_error = true;
		if (!m_errorQueue.empty()) {
			setValueError("Error while converting FeatureSet.",m_strict);
		} 
		return rFeatureSet;
	}

	/// Convert Python list (backward compatibility)
	if (PyList_Check(pyValue)) {
		
		PyObject *pyFeatureList; // This will be borrowed reference

		//Parse Output List for each element (FeatureSet)
		m_error = false;
		for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyValue); ++i) {
			//Get i-th FeatureList (Borrowed Reference)
			pyFeatureList = PyList_GET_ITEM(pyValue,i);
			PyValue_To_rValue(pyFeatureList,rFeatureSet[i]);
			if (m_error) {
				lastError() << " in output number: " << i;
			}
		}
		if (!m_errorQueue.empty()) m_error = true; 
		return rFeatureSet;
	}

	/// accept no return values
	if (pyValue == Py_None) return rFeatureSet;

	/// give up
	std::string msg = "Unsupported return type. Expected list or vampy.FeatureSet(). ";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_FeatureSet failed. Error: " << msg << endl;
#endif			
	return rFeatureSet;
}

Vamp::RealTime
PyTypeInterface::PyValue_To_RealTime(PyObject* pyValue) const
{
// We accept integer sample counts (for backward compatibility)
// or PyRealTime objects and convert them to Vamp::RealTime
	
	if (PyRealTime_CheckExact(pyValue))
	{
		/// just create a copy of the wrapped object
		return Vamp::RealTime(*PyRealTime_AS_REALTIME(pyValue));
	}

	// assume integer sample count
	long sampleCount = PyValue_To_Long(pyValue);
	if (m_error) {
		std::string msg = "Unexpected value passed as RealTime.\nMust be vampy.RealTime type or integer sample count.";
		setValueError(msg,m_strict);
#ifdef _DEBUG
		cerr << "PyTypeInterface::PyValue_To_RealTime failed. " << msg << endl;
#endif		
		return Vamp::RealTime();
	}

#ifdef _DEBUG_VALUES
	Vamp::RealTime rt = 
		Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );
	cerr << "RealTime: " << (long)sampleCount << ", ->" << rt.toString() << endl;
	return rt;
#else
	return Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );	
#endif

}

Vamp::Plugin::OutputDescriptor::SampleType 
PyTypeInterface::PyValue_To_SampleType(PyObject* pyValue) const
{
	/// convert simulated enum values 
	/// { OneSamplePerStep,FixedSampleRate,VariableSampleRate }
	if (PyInt_CheckExact(pyValue)) {
		long lst = PyInt_AS_LONG(pyValue);
		if (lst<0 || lst>2) {
			setValueError("Overflow error. SampleType has to be one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.",m_strict);
			return Vamp::Plugin::OutputDescriptor::SampleType();
		}
		return (Vamp::Plugin::OutputDescriptor::SampleType) lst; 
	}
	
	/// convert string (backward compatible)
	if (PyString_CheckExact(pyValue)) {
		Vamp::Plugin::OutputDescriptor::SampleType st;
		st = (Vamp::Plugin::OutputDescriptor::SampleType) sampleKeys[PyValue_To_String(pyValue)]; 
		if (m_error) {
			std::string msg = "Unexpected value passed as SampleType. Must be one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.";
			setValueError(msg,m_strict);
			return Vamp::Plugin::OutputDescriptor::SampleType();
		}
		return st;
	}

	/// give up
	std::string msg = "Unsupported return type. Expected one of { OneSamplePerStep,FixedSampleRate,VariableSampleRate }\n(an integer in the range of 0..2) or a string value naming the type.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_SampleType failed. Error: " << msg << endl;
#endif			
	return Vamp::Plugin::OutputDescriptor::SampleType();
}

Vamp::Plugin::InputDomain 
PyTypeInterface::PyValue_To_InputDomain(PyObject* pyValue) const
{
	/// convert simulated enum values { TimeDomain,FrequencyDomain }
	if (PyInt_CheckExact(pyValue)) {
		long lst = PyInt_AS_LONG(pyValue);
		if (lst!=0 && lst!=1) {
			setValueError("Overflow error. InputDomain has to be one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.",m_strict);
			return Vamp::Plugin::InputDomain();
		}
		return (Vamp::Plugin::InputDomain) lst; 
	}
	
	/// convert string (backward compatible)
	if (PyString_CheckExact(pyValue)) {
		Vamp::Plugin::InputDomain id;
		id = (PyValue_To_String(pyValue) == "FrequencyDomain")?Vamp::Plugin::FrequencyDomain:Vamp::Plugin::TimeDomain;
		if (m_error) 
		{
			std::string msg = "Unexpected value passed as SampleType. Must be one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.";
			setValueError(msg,m_strict);
			return Vamp::Plugin::InputDomain();
		}
		return id;
	}

	/// give up
	std::string msg = "Unsupported return type. Expected one of { TimeDomain,FrequencyDomain }\n(an integer in the range of 0..1) or a string value naming the type.";
	setValueError(msg,m_strict);
#ifdef _DEBUG
	cerr << "PyTypeInterface::PyValue_To_InputDomain failed. Error: " << msg << endl;
#endif			
	return Vamp::Plugin::InputDomain();
}


/// OutputDescriptor
void
PyTypeInterface::SetValue(Vamp::Plugin::OutputDescriptor& od, std::string& key, PyObject* pyValue) const
{
	switch (outKeys[key])
	{
		case o::not_found:
			setValueError("Unknown key in Vamp OutputDescriptor",m_strict);
			cerr << "Unknown key in Vamp OutputDescriptor: " << key << endl;
			break;
			case o::identifier: 
			_convert(pyValue,od.identifier);
			break;				
		case o::name: 			
			_convert(pyValue,od.name);
			break;
		case o::description:
			_convert(pyValue,od.description);
			break;
		case o::unit:
			_convert(pyValue,od.unit);
			break;
		case o::hasFixedBinCount:
			_convert(pyValue,od.hasFixedBinCount);
			break;
		case o::binCount:
			_convert(pyValue,od.binCount);
			break;
		case o::binNames:
			_convert(pyValue,od.binNames);
			break;
		case o::hasKnownExtents:
			_convert(pyValue,od.hasKnownExtents);
			break;
		case o::minValue:
			_convert(pyValue,od.minValue);
			break;
		case o::maxValue:
			_convert(pyValue,od.maxValue);
			break;
		case o::isQuantized:
			_convert(pyValue,od.isQuantized);
			break;					
		case o::quantizeStep:
			_convert(pyValue,od.quantizeStep);
			break;
		case o::sampleType:
			_convert(pyValue,od.sampleType);
			break;
		case o::sampleRate:
			_convert(pyValue,od.sampleRate);
			break;
		case o::hasDuration:
			_convert(pyValue,od.hasDuration);
			break;
		default:
			setValueError("Unknown key in Vamp OutputDescriptor",m_strict);
			cerr << "Invalid key in Vamp OutputDescriptor: " << key << endl;
	}
}

/// ParameterDescriptor
void
PyTypeInterface::SetValue(Vamp::Plugin::ParameterDescriptor& pd, std::string& key, PyObject* pyValue) const
{
	switch (parmKeys[key]) 
	{
		case p::not_found :
			setValueError("Unknown key in Vamp ParameterDescriptor",m_strict);
			cerr << "Unknown key in Vamp ParameterDescriptor: " << key << endl; 
			break;
		case p::identifier:
			_convert(pyValue,pd.identifier);
			break;				
		case p::name:
			_convert(pyValue,pd.name);
			break;
		case p::description: 	
			_convert(pyValue,pd.description);
			break; 								
		case p::unit:
			_convert(pyValue,pd.unit);
			break; 																		
		case p::minValue:	
			_convert(pyValue,pd.minValue);
			break;
		case p::maxValue:
			_convert(pyValue,pd.maxValue);
			break;
		case p::defaultValue:
			_convert(pyValue,pd.defaultValue);
			break;
		case p::isQuantized:
			_convert(pyValue,pd.isQuantized);
			break;									
		case p::quantizeStep:
			_convert(pyValue,pd.quantizeStep);
			break;
		default :
		 	setValueError("Unknown key in Vamp ParameterDescriptor",m_strict);
			cerr << "Invalid key in Vamp ParameterDescriptor: " << key << endl; 
	}
}

/// Feature (it's like a Descriptor)
bool
PyTypeInterface::SetValue(Vamp::Plugin::Feature& feature, std::string& key, PyObject* pyValue) const
{
	bool found = true;
	switch (ffKeys[key])
	{
		case unknown :
			setValueError("Unknown key in Vamp Feature",m_strict);
			cerr << "Unknown key in Vamp Feature: " << key << endl; 
			found = false;
			break;
		case hasTimestamp:
			_convert(pyValue,feature.hasTimestamp);
			break;				
		case timestamp:
			_convert(pyValue,feature.timestamp);
			break;
		case hasDuration: 	
			_convert(pyValue,feature.hasDuration);
			break;
		case duration:
			_convert(pyValue,feature.duration);
			break;
		case values:
			_convert(pyValue,feature.values);
			break; 								
		case label:
			_convert(pyValue,feature.label);
			break;
		default:
			setValueError("Unknown key in Vamp Feature",m_strict);
			found = false;
	}
	return found;
}


/*			   			  	Error handling		   			  		*/

void
PyTypeInterface::setValueError (std::string message, bool strict) const
{
	m_error = true;
	m_errorQueue.push(ValueError(message,strict));
}

/// return a reference to the last error or creates a new one.
PyTypeInterface::ValueError&
PyTypeInterface::lastError() const 
{
	m_error = false;
	if (!m_errorQueue.empty()) return m_errorQueue.back();
	else {
		m_errorQueue.push(ValueError("Type conversion error.",m_strict));
		return m_errorQueue.back();
	}
}

/// helper function to iterate over the error message queue:
/// pops the oldest item
PyTypeInterface::ValueError 
PyTypeInterface::getError() const
{
	if (!m_errorQueue.empty()) {
		PyTypeInterface::ValueError e = m_errorQueue.front();
		m_errorQueue.pop();
		if (m_errorQueue.empty()) m_error = false;
		return e;
	}
	else {
		m_error = false;
		return PyTypeInterface::ValueError();
	}
}

/*			   			  	Utilities						  		*/

/// get the type name of an object
std::string
PyTypeInterface::PyValue_Get_TypeName(PyObject* pyValue) const
{
	PyObject *pyType = PyObject_Type(pyValue);
	if (!pyType) 
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		return std::string ("< unknown type >");
	}
	PyObject *pyString = PyObject_Str(pyType);
	if (!pyString)
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_CLEAR(pyType);
		return std::string ("< unknown type >");
	}
	char *cstr = PyString_AS_STRING(pyString);
	if (!cstr)
	{
		cerr << "Warning: Object type name could not be found." << endl;
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_DECREF(pyType);
		Py_CLEAR(pyString);
		return std::string("< unknown type >");
	}
	Py_DECREF(pyType);
	Py_DECREF(pyString);
	return std::string(cstr);
	
}

bool
PyTypeInterface::initMaps() const
{

	if (isMapInitialised) return true;

	outKeys["identifier"] = o::identifier;
	outKeys["name"] = o::name;
	outKeys["description"] = o::description;
	outKeys["unit"] = o::unit;
	outKeys["hasFixedBinCount"] = o::hasFixedBinCount; 
	outKeys["binCount"] = o::binCount;
	outKeys["binNames"] = o::binNames;
	outKeys["hasKnownExtents"] = o::hasKnownExtents;
	outKeys["minValue"] = o::minValue;
	outKeys["maxValue"] = o::maxValue;
	outKeys["isQuantized"] = o::isQuantized;
	outKeys["quantizeStep"] = o::quantizeStep;
	outKeys["sampleType"] = o::sampleType;
	outKeys["sampleRate"] = o::sampleRate;
	outKeys["hasDuration"] = o::hasDuration;

	sampleKeys["OneSamplePerStep"] = OneSamplePerStep;
	sampleKeys["FixedSampleRate"] = FixedSampleRate;
	sampleKeys["VariableSampleRate"] = VariableSampleRate;

	ffKeys["hasTimestamp"] = hasTimestamp;
	ffKeys["timestamp"] = timestamp; // this is the correct one
	ffKeys["timeStamp"] = timestamp; // backward compatible
	ffKeys["hasDuration"] = hasDuration;
	ffKeys["duration"] = duration;
	ffKeys["values"] = values;
	ffKeys["label"] = label;

	parmKeys["identifier"] = p::identifier;
	parmKeys["name"] = p::name;
	parmKeys["description"] = p::description;
	parmKeys["unit"] = p::unit;
	parmKeys["minValue"] = p::minValue;
	parmKeys["maxValue"] = p::maxValue;
	parmKeys["defaultValue"] = p::defaultValue;
	parmKeys["isQuantized"] = p::isQuantized;
	parmKeys["quantizeStep"] = p::quantizeStep;

	isMapInitialised = true;
	return true;
}
