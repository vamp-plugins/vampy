'''PyMFCC_oldstyle.py - This example Vampy plugin demonstrates
how to return sprectrogram-like features.

This plugin uses backward compatible syntax and 
no extension module.

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
		# print '\n\n>>Plugin initialised with sample rate: %s<<\n\n' %self.sampleRate
		# print 'minHz:%s\nmaxHz:%s\n' %(self.minHz,self.maxHz)
		
		
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
	

class PyMFCC_oldstyle(melScaling): 
	
	def __init__(self,inputSampleRate):
		self.vampy_flags = 1 # vf_DEBUG = 1
		self.m_inputSampleRate = inputSampleRate 
		self.m_stepSize = 1024
		self.m_blockSize = 2048
		self.m_channels = 1
		self.numBands = 40
		self.cnull = 1
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
		return 'Vampy Old Style MFCC Plugin'
		
	def getIdentifier(self):
		return 'vampy-mfcc-test-old'

	def getDescription(self):
		return 'A simple MFCC plugin. (using the old syntax)'
	
	def getMaxChannelCount(self):
		return 1
		
	def getInputDomain(self):
		return 'TimeDomain'
		
	def getPreferredBlockSize(self):
		return 2048
		
	def getPreferredStepSize(self):
		return 512
			
	def getOutputDescriptors(self):

		Generic={
		'hasFixedBinCount':True,
		'binCount':int(self.numBands)-self.cnull,
		'hasKnownExtents':False,
		'isQuantized':True,
		'sampleType':'OneSamplePerStep'
		}
		
		MFCC=Generic.copy()
		MFCC.update({
		'identifier':'mfccs',
		'name':'MFCCs',
		'description':'MFCC Coefficients',
		'binNames':map(lambda x: 'C '+str(x),range(self.cnull,int(self.numBands))),
		'unit':''		
		})
		
		warpedSpectrum=Generic.copy()
		warpedSpectrum.update({
		'identifier':'warped-fft',
		'name':'Mel Scaled Spectrum',
		'description':'Mel Scaled Magnitide Spectrum',
		'unit':'Mel'
		})
		
		melFilter=Generic.copy()
		melFilter.update({
		'identifier':'mel-filter',
		'name':'Mel Filter Matrix',
		'description':'Returns the created filter matrix.',
		'sampleType':'FixedSampleRate',
		'sampleRate':self.m_inputSampleRate/self.m_stepSize,
		'unit':''
		})
				
		return [MFCC,warpedSpectrum,melFilter]

	def getParameterDescriptors(self):
		melbands = {
		'identifier':'melbands',
		'name':'Number of bands (coefficients)',
		'description':'Set the number of coefficients.',
		'unit':'',
		'minValue':2.0,
		'maxValue':128.0,
		'defaultValue':40.0,
		'isQuantized':True,
		'quantizeStep':1.0
		}
		
		cnull = {
		'identifier':'cnull',
		'name':'Return C0',
		'description':'Select if the DC coefficient is required.',
		'unit':'',
		'minValue':0.0,
		'maxValue':1.0,
		'defaultValue':0.0,
		'isQuantized':True,
		'quantizeStep':1.0
		}
		
		minHz = {
		'identifier':'minHz',
		'name':'minimum frequency',
		'description':'Set the lower frequency bound.',
		'unit':'Hz',
		'minValue':0.0,
		'maxValue':24000.0,
		'defaultValue':0.0,
		'isQuantized':True,
		'quantizeStep':1.0
		}
				
		maxHz = {
		'identifier':'maxHz',
		'name':'maximum frequency',
		'description':'Set the upper frequency bound.',
		'unit':'Hz',
		'minValue':100.0,
		'maxValue':24000.0,
		'defaultValue':11025.0,
		'isQuantized':True,
		'quantizeStep':100.0
		}
		
		return [melbands,minHz,maxHz,cnull]

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
		else:
			return 0.0
			
	def processN(self,membuffer,frameSampleStart):
		
		# recalculate the filter and DCT matrices if needed
		if not self.update() : return []

		fftsize = self.m_blockSize
		audioSamples = frombuffer(membuffer[0],float32)

		complexSpectrum = fft(self.window*audioSamples,fftsize)
		#complexSpectrum =  frombuffer(membuffer[0],complex64,-1,8)

		magnitudeSpectrum = abs(complexSpectrum)[0:fftsize/2] / (fftsize/2)
		melSpectrum = self.warpSpectrum(magnitudeSpectrum)
		melCepstrum = self.getMFCCs(melSpectrum,cn=True)
		
		output_melCepstrum = [{
		'hasTimestamp':False,
		'values':melCepstrum[self.cnull:].tolist()
		}]

		output_melSpectrum = [{
		'hasTimestamp':False,		
		'values':melSpectrum.tolist()
		}]

		return [output_melCepstrum,output_melSpectrum,[]]


	def getRemainingFeatures(self):
		if not self.update() : return []
		frameSampleStart = 0
		output_melFilter = []

		while True:
			try :
				melFilter = self.filterIter.next()
				output_melFilter.append({
				'hasTimestamp':True,
				'timeStamp':frameSampleStart,		
				'values':melFilter.tolist()
				})
				frameSampleStart += self.m_stepSize
			except StopIteration :
				break;

		return [[],[],output_melFilter]


# ============================================
# Simple Unit Tests
# ============================================

def main():
	
	dct = melScaling(44100,2048,numBands=4)
	dct.update()
	print dct.DCTMatrix
	# print dct.getMFCCs(numpy.array([0.0,0.1,0.0,-0.1],numpy.float64))
	sys.exit(-1)

if __name__ == '__main__':
	main()
		
