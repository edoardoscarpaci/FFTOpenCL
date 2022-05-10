#pragma once
#include "AudioFile.h"

#include <complex>
#include <vector>
#include <string>

class WAV{
private:
	std::string filePath;
	AudioFile<float> audioFile;

public:
	WAV(const std::string& path){
		this->filePath = path;
		this->audioFile.load(filePath);
	};
	std::vector<std::complex<float>> getComplexSamples(int channel=0){
		std::cout <<"Channels: "<<audioFile.getNumChannels() << std::endl;
		
		int numSamples = audioFile.getNumSamplesPerChannel();
		std::vector<std::complex<float>> samples;
		for (int i = 0; i < numSamples ; i++){
			samples.emplace_back(std::complex<float>(audioFile.samples[channel][i],0)); 
		}

		return samples;
	}
};