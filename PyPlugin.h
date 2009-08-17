/* -*- c-basic-offset: 8 indent-tabs-mode: t -*- */
/*
    Vamp

    An API for audio analysis and feature extraction plugins.

    Centre for Digital Music, Queen Mary, University of London.
    Copyright 2006 Chris Cannam.
  
    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Except as contained in this notice, the names of the Centre for
    Digital Music; Queen Mary, University of London; and Chris Cannam
    shall not be used in advertising or otherwise to promote the sale,
    use or other dealings in this Software without prior written
    authorization.
*/

	/**
	 * This plugin abstracts appropriate Python Scripts as a Vamp plugin.
	*/

#ifndef _PYTHON_WRAPPER_PLUGIN_H_
#define _PYTHON_WRAPPER_PLUGIN_H_

#include "vamp-sdk/Plugin.h"
#include <Python.h>

#include "Mutex.h"

//fields in OutputDescriptor
namespace o {
enum eOutDescriptors {
	not_found,
	identifier,
	name,
	description,
	unit, 
	hasFixedBinCount,
	binCount,
	binNames,
	hasKnownExtents,
	minValue,
	maxValue,
	isQuantized,
	quantizeStep,
	sampleType,	
	sampleRate,
	hasDuration,
	endNode
	}; 
}

namespace p {
enum eParmDescriptors {
	not_found,
	identifier,
	name,
	description,
	unit, 
	minValue,
	maxValue,
	defaultValue,
	isQuantized,
	quantizeStep
	};
}

enum eSampleTypes {
	OneSamplePerStep,
	FixedSampleRate,
	VariableSampleRate
	};

enum eFeatureFields {
	unknown,
	hasTimestamp,
	timeStamp,
	hasDuration,
	duration,
	values,
	label
	};

enum eProcessType {
	not_implemented,
	legacyProcess,
	numpyProcess
	};

class PyPlugin : public Vamp::Plugin
{
public:
	PyPlugin(std::string plugin,float inputSampleRate, PyObject *pyClass);
	virtual ~PyPlugin();

	bool initialise(size_t channels, size_t stepSize, size_t blockSize);
	void reset();

	InputDomain getInputDomain() const;
	size_t getPreferredBlockSize() const;
	size_t getPreferredStepSize() const; 
	size_t getMinChannelCount() const; 
	size_t getMaxChannelCount() const;

	std::string getIdentifier() const;
	std::string getName() const;
	std::string getDescription() const;
	std::string getMaker() const;
	int getPluginVersion() const;
	std::string getCopyright() const;
	
	OutputList getOutputDescriptors() const;
	ParameterList getParameterDescriptors() const;
	float getParameter(std::string paramid) const;
	void setParameter(std::string paramid, float newval);
    
	FeatureSet process(const float *const *inputBuffers,
			   Vamp::RealTime timestamp);

	FeatureSet getRemainingFeatures();

protected:
	PyObject *m_pyClass;
	PyObject *m_pyInstance;
	size_t m_stepSize;
	size_t m_blockSize;
	size_t m_channels;
	std::string m_plugin;
	std::string m_class;
	std::string m_path;
	int m_processType;
	PyObject *m_pyProcess;
	InputDomain m_inputDomain;
	
	bool initMaps() const;
	std::vector<std::string> PyList_To_StringVector (PyObject *inputList) const;
	std::vector<float> PyList_As_FloatVector (PyObject *inputList) const;

	static Mutex m_pythonInterpreterMutex;
};


#endif
