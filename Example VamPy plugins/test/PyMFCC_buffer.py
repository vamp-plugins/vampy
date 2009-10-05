'''PyMFCC_buffer.py - This example Vampy plugin demonstrates
how to return sprectrogram-like features.

This plugin uses the numpy BUFFER interface and 
frequency domain input. Flag: vf_BUFFER

Centre for Digital Music, Queen Mary University of London.
Copyright 2006 Gyorgy Fazekas, QMUL. 
(See Vamp API for licence information.)

Constants for Mel frequency conversion and filter 
centre calculation are taken from the GNU GPL licenced 
Freespeech library. Copyright (C) 1999 Jean-Marc Valin
'''

import sys,numpy
from numpy import log,exp,floor,sum
from numpy import *       
from numpy.fft import *
import vampy
from vampy import *

class melScaling(object):

	def __init__(self,sampleRate,inputSize,numBands,minHz = 0,maxHz = None):
		'''Initialise frequency warping and DCT matrix. 
		Parameters:
		sampleRate: audio sample rate
		inputSize: length of magnitude spectrum (half of FFT size assumed)
		numBands: number of mel Bands (MFCCs)
		minHz: lower bound of warping  (default = DC)
		maxHz: higher bound of warping (default = Nyquist frequency)
		'''
		self.sampleRate = sampleRate
		self.NqHz = sampleRate / 2.0
		self.minHz = minHz
		if maxHz is None : maxHz = self.NqHz
		self.maxHz = maxHz
		self.inputSize = inputSize
		self.numBands = numBands
		self.valid = False
		self.updated = False
		
	def update(self): 
		# make sure this will run only once if called from a vamp process
		
		if self.updated: return self.valid
		self.updated = True
		self.valid = False
		print 'Updating parameters and recalculating filters: '
		print 'Nyquist: ',self.NqHz
		
		if self.maxHz > self.NqHz : 
			raise Exception('Maximum frequency must be smaller than the Nyquist frequency')
		
		self.maxMel = 1000*log(1+self.maxHz/700.0)/log(1+1000.0/700.0)
		self.minMel = 1000*log(1+self.minHz/700.0)/log(1+1000.0/700.0)
		print 'minHz:%s\nmaxHz:%s\nminMel:%s\nmaxMel:%s\n' %(self.minHz,self.maxHz,self.minMel,self.maxMel)
		self.filterMatrix = self.getFilterMatrix(self.inputSize,self.numBands)
		self.DCTMatrix = self.getDCTMatrix(self.numBands)
		self.filterIter = self.filterMatrix.__iter__()
		self.valid = True
		return self.valid
		
		# try :
		# 	self.maxMel = 1000*log(1+self.maxHz/700.0)/log(1+1000.0/700.0)
		# 	self.minMel = 1000*log(1+self.minHz/700.0)/log(1+1000.0/700.0)
		# 	self.filterMatrix = self.getFilterMatrix(self.inputSize,self.numBands)
		# 	self.DCTMatrix = self.getDCTMatrix(self.numBands)
		# 	self.filterIter = self.filterMatrix.__iter__()
		# 	self.valid = True
		# 	return True
		# except :
		# 	print "Invalid parameter setting encountered in MelScaling class."
		# 	return False
		# return True
		
	def getFilterCentres(self,inputSize,numBands):
		'''Calculate Mel filter centres around FFT bins.
		This function calculates two extra bands at the edges for
		finding the starting and end point of the first and last 
		actual filters.'''
		centresMel = numpy.array(xrange(numBands+2)) * (self.maxMel-self.minMel)/(numBands+1) + self.minMel
		centresBin = numpy.floor(0.5 + 700.0*inputSize*(exp(centresMel*log(1+1000.0/700.0)/1000.0)-1)/self.NqHz)
		return numpy.array(centresBin,int)
		
	def getFilterMatrix(self,inputSize,numBands):
		'''Compose the Mel scaling matrix.'''
		filterMatrix = numpy.zeros((numBands,inputSize))
		self.filterCentres = self.getFilterCentres(inputSize,numBands)
		for i in xrange(numBands) :
			start,centre,end = self.filterCentres[i:i+3]
			self.setFilter(filterMatrix[i],start,centre,end)
		return filterMatrix.transpose()

	def setFilter(self,filt,filterStart,filterCentre,filterEnd):
		'''Calculate a single Mel filter.'''
		k1 = numpy.float32(filterCentre-filterStart)
		k2 = numpy.float32(filterEnd-filterCentre)
		up = (numpy.array(xrange(filterStart,filterCentre))-filterStart)/k1
		dn = (filterEnd-numpy.array(xrange(filterCentre,filterEnd)))/k2
		filt[filterStart:filterCentre] = up
		filt[filterCentre:filterEnd] = dn
						
	def warpSpectrum(self,magnitudeSpectrum):
		'''Compute the Mel scaled spectrum.'''
		return numpy.dot(magnitudeSpectrum,self.filterMatrix)
		
	def getDCTMatrix(self,size):
		'''Calculate the square DCT transform matrix. Results are 
		equivalent to Matlab dctmtx(n) but with 64 bit precision.'''
		DCTmx = numpy.array(xrange(size),numpy.float64).repeat(size).reshape(size,size)
		DCTmxT = numpy.pi * (DCTmx.transpose()+0.5) / size
		DCTmxT = (1.0/sqrt( size / 2.0)) * cos(DCTmx * DCTmxT)
		DCTmxT[0] = DCTmxT[0] * (sqrt(2.0)/2.0)
		return DCTmxT
		
	def dct(self,data_matrix):
		'''Compute DCT of input matrix.'''
		return numpy.dot(self.DCTMatrix,data_matrix)
		
	def getMFCCs(self,warpedSpectrum,cn=True):
		'''Compute MFCC coefficients from Mel warped magnitude spectrum.'''
		mfccs=self.dct(numpy.log(warpedSpectrum))
		if cn is False : mfccs[0] = 0.0
		return mfccs
	

class PyMFCC_buffer(melScaling): 
	
	def __init__(self,inputSampleRate):
		
		# flags for setting some Vampy options
		self.vampy_flags = vf_DEBUG | vf_BUFFER | vf_REALTIME

		self.m_inputSampleRate = int(inputSampleRate)
		self.m_stepSize = 512
		self.m_blockSize = 2048
		self.m_channels = 1
		self.numBands = 40
		self.cnull = 1
		self.two_ch = False
		melScaling.__init__(self,int(self.m_inputSampleRate),self.m_blockSize/2,self.numBands)
		
	def initialise(self,channels,stepSize,blockSize):
		self.m_channels = channels
		self.m_stepSize = stepSize		
		self.m_blockSize = blockSize
		self.window = numpy.hamming(blockSize)
		melScaling.__init__(self,int(self.m_inputSampleRate),self.m_blockSize/2,self.numBands)
		return True
	
	def getMaker(self):
		return 'Vampy Test Plugins'
		
	def getCopyright(self):
		return 'Plugin By George Fazekas'
	
	def getName(self):
		return 'Vampy Buffer MFCC Plugin'
		
	def getIdentifier(self):
		return 'vampy-mfcc-test-buffer'

	def getDescription(self):
		return 'A simple MFCC plugin. (using the Buffer interface)'
	
	def getMaxChannelCount(self):
		return 2
		
	def getInputDomain(self):
		return FrequencyDomain
		
	def getPreferredBlockSize(self):
		return 2048
		
	def getPreferredStepSize(self):
		return 512
			
	def getOutputDescriptors(self):
		
		Generic = OutputDescriptor() 
		Generic.hasFixedBinCount=True
		Generic.binCount=int(self.numBands)-self.cnull
		Generic.hasKnownExtents=False
		Generic.isQuantized=True
		Generic.sampleType = OneSamplePerStep 
		
		# note the inheritance of attributes (use is optional)
		MFCC = OutputDescriptor(Generic)
		MFCC.identifier = 'mfccs'
		MFCC.name = 'MFCCs'
		MFCC.description = 'MFCC Coefficients'
		MFCC.binNames=map(lambda x: 'C '+str(x),range(self.cnull,int(self.numBands)))
		MFCC.unit = None
		if self.two_ch and self.m_channels == 2 :
			MFCC.binCount = self.m_channels * (int(self.numBands)-self.cnull)
		else :
			MFCC.binCount = self.numBands-self.cnull
				
		warpedSpectrum = OutputDescriptor(Generic)
		warpedSpectrum.identifier='warped-fft'
		warpedSpectrum.name='Mel Scaled Spectrum'
		warpedSpectrum.description='Mel Scaled Magnitide Spectrum'
		warpedSpectrum.unit='Mel'
		if self.two_ch and self.m_channels == 2 :
			warpedSpectrum.binCount = self.m_channels * int(self.numBands) 
		else :
			warpedSpectrum.binCount = self.numBands
		
		melFilter = OutputDescriptor(Generic)
		melFilter.identifier = 'mel-filter-matrix'
		melFilter.sampleType='FixedSampleRate'
		melFilter.sampleRate=self.m_inputSampleRate/self.m_stepSize
		melFilter.name='Mel Filter Matrix'
		melFilter.description='Returns the created filter matrix in getRemainingFeatures.'
		melFilter.unit = None
				
		return OutputList(MFCC,warpedSpectrum,melFilter)
		

	def getParameterDescriptors(self):

		melbands = ParameterDescriptor()
		melbands.identifier='melbands'
		melbands.name='Number of bands (coefficients)'
		melbands.description='Set the number of coefficients.'
		melbands.unit = ''
		melbands.minValue = 2
		melbands.maxValue = 128
		melbands.defaultValue = 40
		melbands.isQuantized = True
		melbands.quantizeStep = 1
				
		cnull = ParameterDescriptor()
		cnull.identifier='cnull'
		cnull.name='Return C0'
		cnull.description='Select if the DC coefficient is required.'
		cnull.unit = None
		cnull.minValue = 0
		cnull.maxValue = 1
		cnull.defaultValue = 0
		cnull.isQuantized = True
		cnull.quantizeStep = 1
		
		two_ch = ParameterDescriptor(cnull)
		two_ch.identifier='two_ch'
		two_ch.name='Process channels separately'
		two_ch.description='Process two channel files separately.'
		two_ch.defaultValue = False
				
		minHz = ParameterDescriptor()
		minHz.identifier='minHz'
		minHz.name='minimum frequency'
		minHz.description='Set the lower frequency bound.'
		minHz.unit='Hz'
		minHz.minValue = 0
		minHz.maxValue = 24000
		minHz.defaultValue = 0
		minHz.isQuantized = True
		minHz.quantizeStep = 1.0
		
		maxHz = ParameterDescriptor()
		maxHz.identifier='maxHz'
		maxHz.description='Set the upper frequency bound.'
		maxHz.name='maximum frequency'
		maxHz.unit='Hz'
		maxHz.minValue = 100
		maxHz.maxValue = 24000
		maxHz.defaultValue = 11025
		maxHz.isQuantized = True
		maxHz.quantizeStep = 100
		
		return ParameterList(melbands,minHz,maxHz,cnull,two_ch)
		

	def setParameter(self,paramid,newval):
		self.valid = False
		if paramid == 'minHz' :
			if newval < self.maxHz and newval < self.NqHz :
				self.minHz = float(newval)
			print 'minHz: ', self.minHz
		if paramid == 'maxHz' :
			print 'trying to set maxHz to: ',newval
			if newval < self.NqHz and newval > self.minHz+1000 :
				self.maxHz = float(newval)
			else :
				self.maxHz = self.NqHz
			print 'set to: ',self.maxHz
		if paramid == 'cnull' :
			self.cnull = int(not int(newval))
		if paramid == 'melbands' :
			self.numBands = int(newval)
		if paramid == 'two_ch' :
			self.two_ch = bool(newval)
			
		return 
				
	def getParameter(self,paramid):
		if paramid == 'minHz' :
			return float(self.minHz)
		if paramid == 'maxHz' :
			return float(self.maxHz)
		if paramid == 'cnull' :
			return float(not int(self.cnull))
		if paramid == 'melbands' :
			return float(self.numBands)
		if paramid == 'two_ch' :
			return float(self.two_ch)
		else:
			return 0.0

	# numpy process using the buffer interface
	def process(self,inputbuffers,timestamp):

		if not self.update() : return None
		
		if self.m_channels == 2 and self.two_ch :
			return self.process2ch(inputbuffers,timestamp)
		
		fftsize = self.m_blockSize
		
		if self.m_channels > 1 :
			# take the mean of the two magnitude spectra
			complexSpectrum0 = frombuffer(inputbuffers[0],complex64,-1,0)
			complexSpectrum1 = frombuffer(inputbuffers[1],complex64,-1,0)
			magnitudeSpectrum0 = abs(complexSpectrum0)[0:fftsize/2]
			magnitudeSpectrum1 = abs(complexSpectrum1)[0:fftsize/2]
			magnitudeSpectrum = (magnitudeSpectrum0 + magnitudeSpectrum1) / 2
		else :
			complexSpectrum = frombuffer(inputbuffers[0],complex64,-1,0)
			magnitudeSpectrum = abs(complexSpectrum)[0:fftsize/2]
						
		# do the computation
		melSpectrum = self.warpSpectrum(magnitudeSpectrum)
		melCepstrum = self.getMFCCs(melSpectrum,cn=True)
		
		# output feature set (the builtin dict type can also be used)
		outputs = FeatureSet()
		outputs[0] = Feature(melCepstrum[self.cnull:])
		outputs[1] = Feature(melSpectrum)
		
		return outputs

	# process two channel files (stack the returned arrays)
	def process2ch(self,inputbuffers,timestamp):

		fftsize = self.m_blockSize
		
		complexSpectrum0 = frombuffer(inputbuffers[0],complex64,-1,0)
		complexSpectrum1 = frombuffer(inputbuffers[1],complex64,-1,0)
		
		magnitudeSpectrum0 = abs(complexSpectrum0)[0:fftsize/2]
		magnitudeSpectrum1 = abs(complexSpectrum1)[0:fftsize/2]
		
		# do the computations
		melSpectrum0 = self.warpSpectrum(magnitudeSpectrum0)
		melCepstrum0 = self.getMFCCs(melSpectrum0,cn=True)
		melSpectrum1 = self.warpSpectrum(magnitudeSpectrum1)
		melCepstrum1 = self.getMFCCs(melSpectrum1,cn=True)
		
		outputs = FeatureSet()
		
		outputs[0] = Feature(hstack((melCepstrum1[self.cnull:],melCepstrum0[self.cnull:])))
		
		outputs[1] = Feature(hstack((melSpectrum1,melSpectrum0)))
		
		return outputs


	def getRemainingFeatures(self):
		if not self.update() : return []
		frameSampleStart = 0
		
		output_featureSet = FeatureSet()

		# the filter is the third output (index starts from zero)
		output_featureSet[2] = flist = FeatureList()

		while True:
			f = Feature()
			f.hasTimestamp = True
			f.timestamp = frame2RealTime(frameSampleStart,self.m_inputSampleRate)
			try :
				f.values = self.filterIter.next()
			except StopIteration :
				break
			flist.append(f)
			frameSampleStart += self.m_stepSize

		return output_featureSet
		