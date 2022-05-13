#pragma once
#include "AudioFile.h"

#include <complex>
#include <vector>
#include <string>
#include <cmath>

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
		const double multiplier = std::pow(10.0, 6);
		int numSamples = audioFile.getNumSamplesPerChannel();
		std::vector<std::complex<float>> samples;
		for (int i = 0; i < numSamples ; i++){
			float sample = std::ceil(audioFile.samples[channel][i] * multiplier) /multiplier; 
			samples.emplace_back(std::complex<float>(sample,0) *1.f); 
		}

		return samples;
	}
};