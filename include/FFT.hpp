#pragma once
#include <complex>
#include <vector>

class FFT{

private:
	void fft(std::complex<float>* A, size_t N,int iter);

public:
	FFT();
	std::vector<std::complex<float>> computeFFT(std::vector<std::complex<float>> wave);
	
};