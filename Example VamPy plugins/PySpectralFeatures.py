'''PySpectralFeatures.py - Example plugin demonstrates''' 
'''how to use the NumPy array interface and write Matlab style code.'''

from numpy import *

class PySpectralFeatures: 
	
	def __init__(self,inputSampleRate): 
		self.m_inputSampleRate = inputSampleRate
		self.m_stepSize = 0
		self.m_blockSize = 0
		self.m_channels = 0
		self.threshold = 0.00
		self.r = 2.0
		
	def initialise(self,channels,stepSize,blockSize):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		#self.prevMag = ones((blockSize/2)-1) / ((blockSize/2)-1)
		self.prevMag = zeros((blockSize/2)-1)
		self.prevMag[0] = 1

		return True
	
	def getMaker(self):
		return 'VamPy Example Plugins'
	
	def getName(self):
		return 'VamPy Spectral Features'
		
	def getIdentifier(self):
		return 'vampy-sf2'

	def getDescription(self):
		return 'A collection of low-level spectral descriptors.'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return 'FrequencyDomain'
			
	def getOutputDescriptors(self):

		#descriptors are python dictionaries
		#Generic values are the same for all
		Generic={
		'hasFixedBinCount':True,
		'binCount':1,
		'hasKnownExtents':False,
		'isQuantized':False,
		'sampleType':'OneSamplePerStep'
		}

		#Spectral centroid etc...
		SC=Generic.copy()
		SC.update({
		'identifier':'vampy-sc',
		'name':'Spectral Centroid',
		'description':'Spectral Centroid (Brightness)',
		'unit':'Hz'		
		})
				
		SCF=Generic.copy()
		SCF.update({
		'identifier':'vampy-scf',
		'name':'Spectral Crest Factor',
		'description':'Spectral Crest (Tonality)',
		'unit':'v'
		})

		BW=Generic.copy()
		BW.update({
		'identifier':'vampy-bw',
		'name':'Band Width',
		'description':'Spectral Band Width',
		'unit':'Hz',
		})		
		
		SE=Generic.copy()
		SE.update({
		'identifier':'vampy-se',
		'name':'Shannon Entropy',
		'description':'Shannon Entropy',
		'unit':'',
		})		

		RE=Generic.copy()
		RE.update({
		'identifier':'vampy-re',
		'name':'Renyi Entropy',
		'description':'Renyi Entropy',
		'unit':'',
		})

		KL=Generic.copy()
		KL.update({
		'identifier':'vampy-kl',
		'name':'Kullback Leibler divergence',
		'description':'KL divergence between successive spectra',
		'unit':'',
		})
			
		#return a list of dictionaries
		return [SC,SCF,BW,SE,RE,KL]

	def getParameterDescriptors(self):
		threshold={
		'identifier':'threshold',
		'name':'Noise threshold: ',
		'description':'',
		'unit':'v',
		'minValue':0.0,
		'maxValue':0.5,
		'defaultValue':0.05,
		'isQuantized':False
		}

		renyicoeff={
		'identifier':'r',
		'name':'Renyi entropy coeff: ',
		'description':'',
		'unit':'',
		'minValue':0.0,
		'maxValue':10.0,
		'defaultValue':2,
		'isQuantized':False		
		}

		return [threshold,renyicoeff]

	def setParameter(self,paramid,newval):
		if paramid == 'threshold' :
			self.threshold = newval
		if paramid == 'r' :
			self.r == newval
		return
		
	def getParameter(self,paramid):
		if paramid == 'threshold' :
			return self.threshold
		if paramid == 'r':
			return float(self.r)
		else:
			return 0.0
			
	def processN(self,membuffer,samplecount):
		fftsize = self.m_blockSize
		sampleRate = self.m_inputSampleRate

		#for time domain plugins use the following line:
		#audioSamples = frombuffer(membuffer[0],float32)
		#-1: do till the end, skip DC 2*32bit / 8bit = 8byte
		complexSpectrum =  frombuffer(membuffer[0],complex64,-1,8)
		magnitudeSpectrum = abs(complexSpectrum) / (fftsize/2)
		tpower = sum(magnitudeSpectrum)
		#phaseSpectrum = angle(complexSpectrum)

		freq = array(range(1,len(complexSpectrum)+1)) \
		* sampleRate / fftsize

		centroid = 0.0
		crest = 0.0
		bw = 0.0
		shannon = 0.0
		renyi = 0.0
		r = self.r
		KLdiv = 0.0
		flatness = 0.0
		exp=1.0 / (fftsize/2)
		#print exp

		#declare outputs
		output0=[]
		output1=[]
		output2=[]
		output3=[]
		output4=[]
		output5=[]
		
		if tpower > self.threshold : 

			centroid = sum(freq * magnitudeSpectrum) / tpower 
			crest = max(magnitudeSpectrum)  / tpower
			bw = sum( abs(freq - centroid) * magnitudeSpectrum ) / tpower
			#flatness = prod(abs(complexSpectrum))  
			#print flatness
			normMag = magnitudeSpectrum / tpower #make it sum to 1			
			shannon = - sum( normMag * log2(normMag) )
			renyi = (1/1-r) * log10( sum( power(normMag,r)))
			KLdiv = sum( normMag * log2(normMag / self.prevMag) )
			self.prevMag = normMag
 				
		output0.append({
		'hasTimestamp':False,		
		'values':[float(centroid)],		
		#'label':str(centroid)				
		})

		output1.append({
		'hasTimestamp':False,		
		'values':[float(crest)],		
		#'label':str(crest)				
		})
	
		output2.append({
		'hasTimestamp':False,		
		'values':[float(bw)],		
		#'label':str(bw)				
		})

		output3.append({
		'hasTimestamp':False,		
		'values':[float(shannon)],		
		#'label':str(shannon)				
		})

		output4.append({
		'hasTimestamp':False,		
		'values':[float(renyi)],		
		#'label':str(renyi)				
		})

		output5.append({
		'hasTimestamp':False,		
		'values':[float(KLdiv)],		#strictly must be a list
		#'label':str(renyi)				
		})

		#return a LIST of list of dictionaries
		return [output0,output1,output2,output3,output4,output5]
