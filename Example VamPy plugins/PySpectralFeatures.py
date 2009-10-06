'''PySpectralFeatures.py - Example plugin demonstrates 
how to calculate and return simple low-level spectral 
descriptors (curves) using Numpy and the buffer interface.

Outputs: 
1) Spectral Centroid
2) Spectral Creast Factor
3) Spectral Band-width
4) Spectral Difference (first order)

Centre for Digital Music, Queen Mary University of London.
Copyright (C) 2009 Gyorgy Fazekas, QMUL. (See Vamp sources 
for licence information.)

'''

from numpy import *
from vampy import *

class PySpectralFeatures: 
	
	def __init__(self,inputSampleRate):

		# flags:
		self.vampy_flags = vf_DEBUG | vf_BUFFER | vf_REALTIME

		self.m_inputSampleRate = inputSampleRate
		self.m_stepSize = 0
		self.m_blockSize = 0
		self.m_channels = 0
		self.threshold = 0.05
		return None
		
	def initialise(self,channels,stepSize,blockSize):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		self.prevMag = zeros((blockSize/2))
		return True
		
	def reset(self):
		# reset any initial conditions
		self.prevMag = zeros((blockSize/2))
		return None
	
	def getMaker(self):
		return 'Vampy Example Plugins'
	
	def getName(self):
		return 'Vampy Spectral Features'
		
	def getIdentifier(self):
		return 'vampy-sf3'

	def getDescription(self):
		return 'A collection of low-level spectral descriptors.'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return FrequencyDomain
			
	def getOutputDescriptors(self):

		#Generic values are the same for all
		Generic = OutputDescriptor()
		Generic.hasFixedBinCount=True
		Generic.binCount=1
		Generic.hasKnownExtents=False
		Generic.isQuantized=False
		Generic.sampleType = OneSamplePerStep
		Generic.unit = 'Hz'		
		
		#Spectral centroid etc...
		SC = OutputDescriptor(Generic)
		SC.identifier = 'vampy-sc'
		SC.name = 'Spectral Centroid'
		SC.description ='Spectral Centroid (Brightness)'
				
		SCF = OutputDescriptor(Generic)
		SCF.identifier = 'vampy-scf'
		SCF.name = 'Spectral Crest Factor'
		SCF.description = 'Spectral Crest (Tonality)'
		SCF.unit = 'v'

		BW = OutputDescriptor(Generic)
		BW.identifier = 'vampy-bw'
		BW.name = 'Band Width'
		BW.description = 'Spectral Band Width'
		
		SD = OutputDescriptor(Generic)
		SD.identifier = 'vampy-sd'
		SD.name = 'Spectral Difference'
		SD.description = 'Eucledian distance of successive magnitude spectra.'
			
		#return a tuple, list or OutputList(SC,SCF,BW)
		return OutputList(SC,SCF,BW,SD)

	def getParameterDescriptors(self):
		
		threshold = ParameterDescriptor()
		threshold.identifier='threshold'
		threshold.name='Noise threshold'
		threshold.description='Noise threshold'
		threshold.unit='v'
		threshold.minValue=0
		threshold.maxValue=1
		threshold.defaultValue=0.05
		threshold.isQuantized=False
		
		return ParameterList(threshold)

	def setParameter(self,paramid,newval):
		if paramid == 'threshold' :
			self.threshold = newval
		return
		
	def getParameter(self,paramid):
		if paramid == 'threshold' :
			return self.threshold
		else:
			return 0.0


    # using the numpy memory buffer interface: 
	# flag : vf_BUFFER (or implement processN)
	# NOTE: Vampy can now pass numpy arrays directly using 
	# the flag vf_ARRAY (see MFCC plugin for example)
	def process(self,membuffer,timestamp):

		fftsize = self.m_blockSize
		sampleRate = self.m_inputSampleRate

		#for time domain plugins use the following line:
		#audioSamples = frombuffer(membuffer[0],float32)

		#for frequency domain plugins use:
		complexSpectrum =  frombuffer(membuffer[0],complex64,-1,8)

		# meaning of the parameters above:
		# complex64 : data type of the created numpy array
		# -1 : convert the whole buffer 
		#  8 : skip the DC component (2*32bit / 8bit = 8byte)

		magnitudeSpectrum = abs(complexSpectrum) / (fftsize*0.5)
		#phaseSpectrum = angle(complexSpectrum)
		
		freq = array(range(1,len(complexSpectrum)+1)) \
		* sampleRate / fftsize
		
		# return features in a FeatureSet()
		output_featureSet = FeatureSet()

		tpower = sum(magnitudeSpectrum)

		if tpower > self.threshold : 
			centroid = sum(freq * magnitudeSpectrum) / tpower 
			crest = max(magnitudeSpectrum)  / tpower
			bw = sum( abs(freq - centroid) * magnitudeSpectrum ) / tpower
			normMag = magnitudeSpectrum / tpower			
			sd = sqrt(sum(power((normMag - self.prevMag),2)))
			self.prevMag = normMag
		else :
			centroid = 0.0
			crest = 0.0
			bw = 0.0
			sd = 0.0
			
		# Any value resulting from the process can be returned.
		# It is no longer necessary to wrap single values into lists
		# and convert numpy.floats to python floats,
		# however a FeatureList() (or python list) can be returned
		# if more than one feature is calculated per frame.
		# The feature values can be e.g. int, float, list or array.
		
		output_featureSet[0] = Feature(centroid)
		output_featureSet[1] = Feature(crest)
		output_featureSet[2] = Feature(bw)
		output_featureSet[3] = Feature(sd)

		return output_featureSet
