'''PyZeroCrossing.py - Example plugin demonstrates''' 
'''how to call a python class using the VamPy Vamp plugin'''

class PyZeroCrossing: 
	
	def __init__(self): 
		self.m_imputSampleRate = 0.0 
		self.m_stepSize = 0
		self.m_blockSize = 0
		self.m_channels = 0
		self.previousSample = 0.0
		self.threshold = 0.01
		
	def initialise(self,channels,stepSize,blockSize,inputSampleRate):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		self.m_inputSampleRate = inputSampleRate
		return True
	
	def getMaker(self):
		return 'VamPy Example Plugins'
	
	def getName(self):
		return 'Vampy Zero Crossings'
		
	def getIdentifier(self):
		return 'python-zc'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return 'TimeDomain'
			
	def getOutputDescriptors(self):
		
		#descriptors are python dictionaries
		output0={
		'identifier':'vampy-counts',
		'name':'Number of Zero Crossings',
		'description':'Number of zero crossings per audio frame',
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

		output1={
		'identifier':'vampy-crossings',
		'name':'Zero Crossing Locations',
		'description':'The locations of zero crossing points',
		'unit':'discrete',
		'hasFixedBinCount':True,
		'binCount':0,
		'sampleType':'VariableSampleRate'
		#'sampleRate':48000.0
		}

		#return a list of dictionaries
		return [output0,output1]

	def getParameterDescriptors(self):
		paramlist1={
		'identifier':'threshold',
		'name':'Noise threshold: ',
		'description':'Return null or delete this function if not needed.',
		'unit':'v',
		'minValue':0.0,
		'maxValue':0.5,
		'defaultValue':0.005,
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
		crossing = False
		prev = self.previousSample
		count = 0.0;
		channel = inbuf[0]

		#we have two outputs defined thus we have to declare
		#them as empty dictionaries in our output list
		#in order to be able to return variable rate outputs
		output0=[]
		output1=[]

		if sum([abs(s) for s in channel]) > self.threshold : 

			for x in range(len(channel)-1) :
				crossing = False
				sample = channel[x]
				if sample <= 0.0 : 
					if prev > 0.0 : crossing = True
				else :
					if sample > 0.0 :
						if prev <= 0.0 : crossing = True		
			
				if crossing == True : 
					count = count + 1
					feature1={
					'hasTimestamp':True,	
					#for now return sample position and convert to RealTime in C code
					'timeStamp':long(timestamp + x),				
					'values':[count],			
					'label':str(count),				
					}				
					output1.append(feature1)
			
				prev = sample	
			self.previousSample = prev

		else :
			count = 0.0
			self.previousSample = channel[len(channel)-1]

		feature0={
		'hasTimestamp':False,		
		#'timeStamp':timestamp,				
		'values':[count],		#strictly must be a list
		'label':str(count)				
		}
		output0.append(feature0)
		
		#return a LIST of list of dictionaries
		return [output0,output1]
		
