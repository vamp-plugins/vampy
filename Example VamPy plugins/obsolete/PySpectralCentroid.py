'''PySpectralCentroid.py - Example plugin demonstrates 
how to write a C style plugin using VamPy.

Obsolete warning: this plugin will no longer be supported 
since the legacy interface should pass a list of complex
numbers for frequency domain plugins and a list of floats
for time domin plugins.

'''

from numpy import *

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
		return 'Spectral Centroid (legacy process interface)'
		
	def getIdentifier(self):
		return 'vampy-sc2'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return 'FrequencyDomain'
			
	def getOutputDescriptors(self):
		
		output0={
		'identifier':'vampy-sf1',
		'name':'Spectral Centroid',
		'description':'Spectral Centroid (Brightness)',
		'unit':' ',
		'hasFixedBinCount':True,
		'binCount':1,
		'hasKnownExtents':False,
		'isQuantized':True,
		'quantizeStep':1.0,
		'sampleType':'OneSamplePerStep'
		}

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
		
		# re = array(inbuf[2:len(inArray):2])
		# im = array(inbuf[3:len(inArray):2])

		output0=[]
		output1=[]

		# pw = 0
		# for i in xrange(1,len(inbuf[0])) : 
		# 	pw = pw + abs(inbuf[0][i])
		
		if sum(abs(inArray)) > self.threshold : 
			for i in range(1,(len(inArray)/2)) :
			# for i in range(1,len(inbuf[0])) :
				
				re = inArray[i*2]
				im = inArray[i*2+1]
				# re = inbuf[0][i].real
				# im = inbuf[0][i].imag
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
		
		return [output0]
