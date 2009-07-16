'''PySpectralCentroid.py - Example plugin demonstrates''' 
'''how to write a C style plugin using VamPy.'''

from numpy import *

class PySpectralCentroid: 
	
	def __init__(self): 
		self.m_imputSampleRate = 0.0 
		self.m_stepSize = 0
		self.m_blockSize = 0
		self.m_channels = 0
		self.previousSample = 0.0
		self.threshold = 0.05
		
	def initialise(self,channels,stepSize,blockSize,inputSampleRate):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		self.m_inputSampleRate = inputSampleRate
		return True
	
	def getMaker(self):
		return 'VamPy Example Plugins'
	
	def getName(self):
		return 'Spectral Centroid (VamPy Legacy Interface)'
		
	def getIdentifier(self):
		return 'python-sf1'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return 'FrequencyDomain'
			
	def getOutputDescriptors(self):
		
		#descriptors are python dictionaries
		output0={
		'identifier':'vampy-sf1',
		'name':'Spectral Centroid',
		'description':'Spectral Centroid (Brightness)',
		'unit':' ',
		'hasFixedBinCount':True,
		'binCount':1,
		#'binNames':['1 Hz',1.5,'2 Hz',3,'4 Hz'],
		'hasKnownExtents':False,
		#'minValue':0.0,
		#'maxValue':0.0,
		'isQuantized':True,
		'quantizeStep':1.0,
		'sampleType':'OneSamplePerStep'
		#'sampleRate':48000.0
		}

		#return a list of dictionaries
		return [output0]

	def getParameterDescriptors(self):
		paramlist1={
		'identifier':'threshold',
		'name':'Noise threshold: ',
		'description':'Return null or delete this function if not needed.',
		'unit':'v',
		'minValue':0.0,
		'maxValue':0.5,
		'defaultValue':0.05,
		'isQuantized':False
		}
		return [paramlist1]

	def setParameter(self,paramid,newval):
		if paramid == 'threshold' :
			self.threshold = newval
		return
		
	def getParameter(self,paramid):
		if paramid == 'threshold' :
			return self.threshold
		else:
			return 0.0
			
	def process(self,inbuf,timestamp):
		inArray = array(inbuf[0])
		crossing = False
		prev = self.previousSample
		count = 0.0
		numLin = 0.0
		denom = 0.0
		centroid = 0.0
		

		re = array(inbuf[2:len(inArray):2])
		im = array(inbuf[3:len(inArray):2])
		#we have two outputs defined thus we have to declare
		#them as empty dictionaries in our output list
		#in order to be able to return variable rate outputs
		output0=[]
		output1=[]
		
		if sum(abs(inArray)) > self.threshold : 
			for i in range(1,(len(inArray)/2)) :
				
				re = inArray[i*2]
				im = inArray[i*2+1]
				freq = i * self.m_inputSampleRate / self.m_blockSize
				power = sqrt (re*re + im*im) / (self.m_blockSize/2)
				denom = denom + power
				numLin = numLin + freq * power
				
			if denom != 0 :
				centroid = numLin / denom 
				
		else :
			centroid = 0.0

		feature0={
		'hasTimestamp':False,		
		'values':[centroid],		#strictly must be a list
		'label':str(centroid)				
		}
		output0.append(feature0)
		
		#return a LIST of list of dictionaries
		return [output0]
