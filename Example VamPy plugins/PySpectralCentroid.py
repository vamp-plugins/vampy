'''PySpectralCentroid.py - Example plugin demonstrates 
how to write a C style plugin using VamPy in pure Python.
This plugin also introduces the use of the builtin vampy
extension module.

The plugin has frequency domain input and is using the
legacy interface: the FFT outpout is passed as a list
of complex numbers.

Outputs: 
1) Spectral centroid

Note: This is not the adviced way of writing Vampy plugins now,
since the interfaces provided for Numpy are at least 5 times
faster. However, this is still a nice and easy to understand
example, which also shows how can one write a reasonable
plugin without having Numpy installed.

Warning: Earlier versions of this plugin are now obsolete.
(They were using the legacy interface of Vampy 1 which
did not distinquish between time and frequency domain inputs.)

Centre for Digital Music, Queen Mary University of London.
Copyright (C) 2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
for licence information.)

'''

# import the names we use from vampy 
from vampy import Feature,FeatureSet,ParameterDescriptor
from vampy import OutputDescriptor,FrequencyDomain,OneSamplePerStep

from math import sqrt

class PySpectralCentroid: 
	
	def __init__(self,inputSampleRate): 
		self.m_imputSampleRate = 0.0 
		self.m_stepSize = 0
		self.m_blockSize = 0
		self.m_channels = 0
		self.previousSample = 0.0
		self.m_inputSampleRate = inputSampleRate
		self.threshold = 0.00
		
	def initialise(self,channels,stepSize,blockSize):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		return True
	
	def getMaker(self):
		return 'Vampy Example Plugins'
	
	def getName(self):
		return 'Spectral Centroid (using legacy process interface)'
		
	def getIdentifier(self):
		return 'vampy-sc3'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return FrequencyDomain
			
	def getOutputDescriptors(self):
		
		cod = OutputDescriptor()
		cod.identifier='vampy-sc3'
		cod.name='Spectral Centroid'
		cod.description='Spectral Centroid (Brightness)'
		cod.unit=''
		cod.hasFixedBinCount=True
		cod.binCount=1
		cod.hasKnownExtents=False
		cod.isQuantized=True
		cod.quantizeStep=1.0
		cod.sampleType=OneSamplePerStep
		return cod

	def getParameterDescriptors(self):
		thd = ParameterDescriptor()
		thd.identifier='threshold'
		thd.name='Noise threshold'
		thd.description='Return null or delete this function if not needed.'
		thd.unit='v'
		thd.minValue=0.0
		thd.maxValue=0.5
		thd.defaultValue=0.05
		thd.isQuantized=False
		return thd

	def setParameter(self,paramid,newval):
		if paramid == 'threshold' :
			self.threshold = newval
		return
		
	def getParameter(self,paramid):
		if paramid == 'threshold' :
			return self.threshold
		else:
			return 0.0
			
	def process(self,inputbuffers,timestamp):
		
		# this is a 1 channel frequency domain plugin, therefore
		# inputbuffers contain (block size / 2) + 1 complex numbers
		# corresponding to the FFT output from DC to Nyquist inclusive
		
		cplxArray = inputbuffers[0][:-1]

		prev = self.previousSample
		numLin = 0.0
		denom = 0.0
		centroid = 0.0		

		output = FeatureSet()

		pw = 0
		for i in xrange(1,len(cplxArray)) : 
			pw = pw + abs(cplxArray[i])
		
		if pw > self.threshold : 
			for i in range(1,(len(cplxArray))) :
				
				re = cplxArray[i].real
				im = cplxArray[i].imag
				freq = i * self.m_inputSampleRate / self.m_blockSize
				power = sqrt (re*re + im*im) / (self.m_blockSize/2)
				denom = denom + power
				numLin = numLin + freq * power
				
			if denom != 0 :
				centroid = numLin / denom 
				
		else :
			centroid = 0.0
			
		output[0] = Feature()
		output[0].values = centroid
		output[0].label = str(centroid)
		
		return output
