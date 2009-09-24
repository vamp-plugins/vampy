/*
*/

#include <Python.h>
#ifdef HAVE_NUMPY
#include "arrayobject.h"
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
	m_lastError(m_noError),
	error(m_error)
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
	if (PyFloat_Check(pyValue)) 
	{	
		float rValue = (float) PyFloat_AS_DOUBLE(pyValue);
		if (PyErr_Occurred()) 
		{
			PyErr_Print(); PyErr_Clear();
			setValueError("Error while converting float object.",m_strict);
			return 0.0;
		}
		return rValue;
	}

	// in strict mode we will not try harder
	if (m_strict) {
		setValueError("Strict conversion error: object is not float.",m_strict);
		return 0.0;
	}
	
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
		if (rValue > FLT_MAX || rValue < FLT_MIN)
		{
			setValueError("Overflow error. Object can not be converted to float.",m_strict);
			return 0.0;
		}
		return (float) rValue;
	}
	
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
				std::string msg = "Could not convert sequence element. " + lastError().message;
				setValueError(msg,m_strict);
				return 0.0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + PyValue_Get_TypeName(pyValue) + " to float is not possible.";
	setValueError(msg,m_strict);
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
			setValueError (lastError().message,m_strict);
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
				std::string msg = "Could not convert sequence element. " + lastError().message;
				setValueError(msg,m_strict);
				return 0;
			}
		}
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to size_t is not possible.";
	setValueError(msg,m_strict);
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
	
	// convert iterables: the rule is the same as of the interpreter:
	// empty sequence evaluates to False, anything else is True
	if (PySequence_Check(pyValue)) 
	{
		return PySequence_Size(pyValue)?true:false;
	}
	
    // give up
	if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to boolean is not possible.";
	setValueError(msg,m_strict);
	return false;
}

/// string and objects that support .__str__() (TODO: check unicode objects)
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
		
	// convert list or tuple: empty lists are turned into empty strings
	if (PyList_Check(pyValue) || PyTuple_Check(pyValue)) 
	{
		if (!PySequence_Size(pyValue)) return std::string();
		PyObject* item = PySequence_GetItem(pyValue,0);
		if (item)
		{
			std::string rValue = this->PyValue_To_String(item);
			if (m_error) {
				Py_DECREF(item);
				return rValue;
			} else {
				Py_CLEAR(item);
				std::string msg = "Could not convert sequence element. " + lastError().message;
				setValueError(msg,m_strict);
				return std::string();
			}
		}
	}

	// convert any other object that has .__str__() or .__repr__()
	PyObject* pyString = PyObject_Str(pyValue);
	if (pyString && !PyErr_Occurred())
	{
		std::string rValue = this->PyValue_To_String(pyString);
		if (m_error) {
			Py_DECREF(pyString);
			return rValue;
		} else {
			Py_CLEAR(pyString);
			std::string msg = "Object " + this->PyValue_Get_TypeName(pyValue) +" can not be represented as string. " + lastError().message;
			setValueError (msg,m_strict);
			return std::string();
		}
	}

	// give up
	PyErr_Print(); PyErr_Clear();
	std::string msg = "Conversion from " + this->PyValue_Get_TypeName(pyValue) + " to string is not possible.";
	setValueError(msg,m_strict);
	return std::string();
}

/*			 			C Values to Py Values				  		*/


PyObject*
PyTypeInterface::PyValue_From_CValue(const char* cValue) const
{
	// returns new reference
	if (!cValue) 
		setValueError("Invalid pointer encountered while converting from char* .",m_strict);
	PyObject *pyValue = PyString_FromString(cValue);
	if (!pyValue)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		setValueError("Error while converting from char* or string.",m_strict);
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
		return NULL;
	}
	return pyValue;
}


/*			 			Sequence Types to C++ Types	    		  	*/

//convert Python list to C++ vector of strings
std::vector<std::string> 
PyTypeInterface::PyValue_To_StringVector (PyObject *inputList) const 
{
	
	std::vector<std::string> Output;
	std::string ListElement;
	PyObject *pyString = NULL;
	
	if (!PyList_Check(inputList)) return Output;

	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(inputList); ++i) {
		//Get next list item (Borrowed Reference)
		pyString = PyList_GET_ITEM(inputList,i);
		ListElement = (string) PyString_AsString(PyObject_Str(pyString));
		Output.push_back(ListElement);
	}
	return Output;
}

//convert PyFeature.value (typically a list or numpy array) to C++ vector of floats
std::vector<float> 
PyTypeInterface::PyValue_To_FloatVector (PyObject *pyValue) const 
{
	std::vector<float> Output;

#ifdef HAVE_NUMPY
	/// Check for NumPy Array: this requires linking with numpy
	/// but, we don't really need this macro
	// if (PyArray_CheckExact(inputList)) cerr << "PyPyArray_CheckExact OK" << endl;
	
	/// numpy array
	if (PyObject_HasAttrString(pyValue,"__array_struct__")) {
		return PyArray_To_FloatVector(pyValue);
	}
#endif

	/// python list
	if (PyList_Check(pyValue)) {
		return PyList_To_FloatVector(pyValue);
	}
	
	/// assume a single number 
	/// this allows to write e.g. Feature.values = 5 instead of [5.00]
	Output.push_back(PyValue_To_Float(pyValue));
	return Output;
	
	/// TODO : set error
	
}

//convert a list of python floats
std::vector<float> 
PyTypeInterface::PyList_To_FloatVector (PyObject *inputList) const 
{
	std::vector<float> Output;
	
	float ListElement;
	PyObject *pyFloat = NULL;
	
	if (!PyList_Check(inputList)) return Output; 

	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(inputList); ++i) {
		//Get next list item (Borrowed Reference)
		pyFloat = PyList_GET_ITEM(inputList,i);
		ListElement = (float) PyFloat_AS_DOUBLE(pyFloat);
#ifdef _DEBUG
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
	 
	/// we don't verify the array here as it'd be duplicated mostly
	// if (!PyObject_HasAttrString(pyValue,"__array_struct__")) {
	// 	return Output;
	// }

	PyArrayObject* pyArray = (PyArrayObject*) pyValue;
	PyArray_Descr* descr = pyArray->descr;
	
	/// check raw data pointer
	if (pyArray->data == 0) return Output;

	/// check dimensions
	if (pyArray->nd != 1) {
		cerr << "Error: array must be 1D" << endl;
		return Output;
	}

#ifdef _DEBUG
	cerr << "Numpy array verified." << endl;
#endif	

	switch (descr->type_num)
	{
		case NPY_FLOAT :
			return PyArray_Convert<float,float>(pyArray->data,pyArray->dimensions[0]);
		case NPY_DOUBLE :
			return PyArray_Convert<float,double>(pyArray->data,pyArray->dimensions[0]);
		case NPY_INT :
			return PyArray_Convert<float,int>(pyArray->data,pyArray->dimensions[0]);
		case NPY_LONG :
			return PyArray_Convert<float,long>(pyArray->data,pyArray->dimensions[0]);
		
		default :
		cerr << "Error. Unsupported element type in NumPy array object." << endl;
			return Output;
	}
}
#endif

/*			 			Vamp API Specific Types				  		

Vamp::Plugin::OutputList 
PyTypeInterface::PyValue_To_OutputList(PyObject* pyList) const
{
	Vamp::Plugin::OutputList list;
	Vamp::Plugin::OutputDescriptor od;
	
	// Type checking
	if (! PyList_Check(pyList) ) {
		Py_CLEAR(pyList);
		// cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
		// << "] Expected List return type." << endl;
		return list;
	}
	
	//These will all be borrowed references (no need to DECREF)
	PyObject *pyDict;
		
	//Parse Output List
	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
		//Get i-th Vamp output descriptor (Borrowed Reference)
		pyDict = PyList_GET_ITEM(pyList,i);
		od = PyValue_To_OutputDescriptor(pyDict);
		list.push_back(od);
	}
	return list;
}

Vamp::Plugin::ParameterList 
PyTypeInterface::PyValue_To_ParameterList(PyObject* pyList) const
{
	Vamp::Plugin::ParameterList list;
	Vamp::Plugin::ParameterDescriptor pd;
	
	// Type checking
	if (! PyList_Check(pyList) ) {
		Py_CLEAR(pyList);
		// cerr << "ERROR: In Python plugin [" << m_class << "::" << method 
		// << "] Expected List return type." << endl;
		return list;
	}
	
	//These will all be borrowed references (no need to DECREF)
	PyObject *pyDict;
		
	//Parse Output List
	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyList); ++i) {
		//Get i-th Vamp output descriptor (Borrowed Reference)
		pyDict = PyList_GET_ITEM(pyList,i);
		pd = PyValue_To_ParameterDescriptor(pyDict);
		list.push_back(pd);
	}
	return list;
}

Vamp::Plugin::OutputDescriptor
PyTypeInterface::PyValue_To_OutputDescriptor(PyObject* pyDict) const
{
	//We only care about dictionaries holding output descriptors
	if (!PyDict_Check(pyDict)) 
		return Vamp::Plugin::OutputDescriptor();
                       
	Py_ssize_t pyPos = 0;
	PyObject *pyKey, *pyValue;
	initMaps();
	Vamp::Plugin::OutputDescriptor od;

	//Python Dictionary Iterator:
	while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue))
	{
		std::string key = PyValue_To_String(pyKey);
		SetValue(od,key,pyValue);
		if (m_error) {
			_lastError().location += "parameter: '" 
			+ key +"' descriptor: '" + od.identifier + "'";
		}
	}
	if (!m_errorQueue.empty()) m_error = true;
	return od;
}

Vamp::Plugin::ParameterDescriptor
PyTypeInterface::PyValue_To_ParameterDescriptor(PyObject* pyDict) const
{
	//We only care about dictionaries holding output descriptors
	if (!PyDict_Check(pyDict)) 
		return Vamp::Plugin::ParameterDescriptor();
                       
	Py_ssize_t pyPos = 0;
	PyObject *pyKey, *pyValue;
	initMaps();
	Vamp::Plugin::ParameterDescriptor pd;

	//Python Dictionary Iterator:
	while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue))
	{
		std::string key = PyValue_To_String(pyKey);
		SetValue(pd,key,pyValue);
		if (m_error) {
			_lastError().location += "parameter: '" 
			+ key +"' descriptor: '" + pd.identifier + "'";
		}
	}
	if (!m_errorQueue.empty()) m_error = true;
	return pd;
}
 
Vamp::Plugin::Feature
PyTypeInterface::PyValue_To_Feature(PyObject* pyDict) const
{
	//We only care about dictionaries holding output descriptors
	if (!PyDict_Check(pyDict)) 
		return Vamp::Plugin::Feature();
                       
	Py_ssize_t pyPos = 0;
	PyObject *pyKey, *pyValue;
	initMaps();
	Vamp::Plugin::Feature feature;

	//Python Dictionary Iterator:
	while (PyDict_Next(pyDict, &pyPos, &pyKey, &pyValue))
	{
		std::string key = PyValue_To_String(pyKey);
		float isr = 22050.0;
		Feature_SetValue(feature,key,pyValue,isr);
		if (m_error) {
			_lastError().location += "key: '" + key + "'";
			// _lastError().location += "parameter: '" 
			// + key +"' descriptor: '" + pd.identifier + "'";
		}
	}
	if (!m_errorQueue.empty()) m_error = true;
	return feature;
}
*/

/// FeatureSet (an int map of OutputLists)
Vamp::Plugin::FeatureSet
PyTypeInterface::PyValue_To_FeatureSet(PyObject* pyValue) const
{
	Vamp::Plugin::FeatureSet rFeatureSet; /// map<int>
	if (pyValue == NULL) {
		cerr << "NULL FeatureSet" << endl;
		return rFeatureSet;
	}

	cerr << "PyValue_To_FeatureSet" << endl;
	//Convert PyFeatureSet 
	if (PyFeatureSet_CheckExact(pyValue)) { 
		cerr << "FeatureSet Return type" << endl;
		Py_ssize_t pyPos = 0;
		//Borrowed References
		PyObject *pyKey, *pyDictValue;
		int key;

			//Python Dictionary Iterator:
			while (PyDict_Next(pyValue, &pyPos, &pyKey, &pyDictValue))
			{
				/// DictValue -> Vamp::FeatureList
				key = (int) PyInt_AS_LONG(pyKey);
				cerr << "FeatureSet key = " << key << endl;
				/// Error checking is done at value assignment
				PyValue_To_rValue(pyDictValue,rFeatureSet[key]);
			}
			if (!m_errorQueue.empty()) m_error = true;
			return rFeatureSet;
	}
	cerr << "not FeatureSet Return type" << endl;
	
		
	//Check return type
	if (pyValue == NULL || !PyList_Check(pyValue) ) {
		if (pyValue == NULL) {
			// cerr << "ERROR: In Python plugin [" << m_class << "::" << method << "] Unexpected result." << endl;
			if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
		} else {
			// cerr << "ERROR: In Python plugin [" << m_class << "::" << method << "] Expected List return type." << endl;
		}
		Py_CLEAR(pyValue);
		return Vamp::Plugin::FeatureSet();
	}
		
	// This will be borrowed reference
	PyObject *pyFeatureList;

	//Parse Output List for each element (FeatureSet)
	for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pyValue); ++i) {
		//Get i-th FeatureList (Borrowed Reference)
		pyFeatureList = PyList_GET_ITEM(pyValue,i);
		PyValue_To_rValue(pyFeatureList,rFeatureSet[i]);
	}
	// Py_CLEAR(pyOutputList);
	return rFeatureSet;
}

Vamp::RealTime
PyTypeInterface::PyValue_To_RealTime(PyObject* pyValue) const
{
// We accept integer sample counts (for backwards compatibility)
// or PyRealTime objects and convert them to Vamp::RealTime
	
	if (PyRealTime_CheckExact(pyValue))
	{
#ifdef _DEBUG
		cerr << "Converting from PyRealTime" << endl;
#endif
		/// just create a copy of the wrapped object
		return Vamp::RealTime(
			*PyRealTime_AS_REALTIME(pyValue));
	}
	// assume integer sample count
	long sampleCount = PyLong_AsLong(pyValue);
	if (PyErr_Occurred()) 
	{
		PyErr_Print(); PyErr_Clear();
		setValueError("Error while converting integer to RealTime.",m_strict);
		return Vamp::RealTime();
	}
#ifdef _DEBUG
	Vamp::RealTime rt = 
		Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );
	cerr << "RealTime: " << (long)sampleCount << ", ->" << rt.toString() << endl;
	return rt;
#else
	return Vamp::RealTime::frame2RealTime(sampleCount,m_inputSampleRate );	
#endif

}


/// OutputDescriptor
void
PyTypeInterface::SetValue(Vamp::Plugin::OutputDescriptor& od, std::string& key, PyObject* pyValue) const
{
	switch (outKeys[key])
	{
		case o::not_found:
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
			// implements specific conversion!
			od.sampleType = (Vamp::Plugin::OutputDescriptor::SampleType) sampleKeys[PyValue_To_String(pyValue)];
			break;
		case o::sampleRate:
			_convert(pyValue,od.sampleRate);
			break;
		case o::hasDuration:
			_convert(pyValue,od.hasDuration);
			break;
		default:
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
			cerr << "Unknown key in Vamp Feature: " << key << endl; 
			found = false;
			break;
		case hasTimestamp:
			_convert(pyValue,feature.hasTimestamp);
			break;				
		case timeStamp:
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
			found = false;
	}
	return found;
}

/*
/// Feature (it's like a Descriptor)
bool
PyTypeInterface::GetValue(Vamp::Plugin::Feature& feature, std::string& key, PyObject* pyValue) const
{
	bool found = true;
	switch (ffKeys[key])
	{
		case unknown :
			cerr << "Unknown key in Vamp Feature: " << key << endl; 
			found = false;
			break;
		case hasTimestamp:
			_convert(pyValue,feature.hasTimestamp);
			// pyValue = PyValue_From_CValue(feature.hasTimestamp)
			break;				
		case timeStamp:
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
			_convert(pyValue,feature.label); //vector<string>
			break;
		default:
			found = false;
	}
	return found;
}
*/

/*			   			  	Error handling		   			  		*/

void
PyTypeInterface::setValueError (std::string message, bool strict) const
{
	m_error = true;
	m_errorQueue.push(ValueError(message,strict));
}

/// return a reference to the last error or creates a new one.
PyTypeInterface::ValueError&
PyTypeInterface::_lastError() const 
{
	m_error = false;
	if (!m_errorQueue.empty()) return m_errorQueue.back();
	else {
		m_errorQueue.push(ValueError("Type conversion error.",m_strict));
		return m_errorQueue.back();
	}
}

/// return the last error message and clear the error flag
const PyTypeInterface::ValueError&
PyTypeInterface::lastError() const 
{
	// PyTypeInterface *self = const_cast<PyTypeInterface*> (this);
	m_error = false;
	if (!m_errorQueue.empty()) return m_errorQueue.back();
	else return m_noError;
}

/// iterate over the error message queue and pop the oldest item
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
	if (!pyType) return std::string ("< unknown type >");
	PyObject *pyString = PyObject_Str(pyType);
	if (!pyString)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_CLEAR(pyType);
		return std::string ("< unknown type >");
	}
	char *cstr = PyString_AS_STRING(pyString);
	if (!cstr)
	{
		if (PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
		Py_DECREF(pyType);
		Py_CLEAR(pyString);
		cerr << "Warning: Object type name could not be found." << endl;
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
	ffKeys["timeStamp"] = timeStamp;
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
