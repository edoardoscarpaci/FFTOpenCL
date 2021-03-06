#include "FFT.hpp"
#include <iostream>

FFT::FFT(){
}

void FFT::fft(std::complex<float>* A, size_t N,int iter){
	
	if (N<=1) return;

	std::complex<float>* even = new std::complex<float>[N/2];
	std::complex<float>* odd = new std::complex<float>[N/2];

	for(size_t i=0;i<N/2;i++){
		even[i] = A[i*2];
		odd[i] = A[i*2+1];
	}

	fft(even,N/2,iter-1);
	fft(odd,N/2,iter-1);
	for(int i=0;i<N/2;i++){	
		std::complex<float> t = std::exp(std::complex<float>(0, -2 * M_PI * i / N));
		std::complex<float> t_odd  = t * odd[i];
	

		A[i] = even[i]  + t_odd;
		A[i + N/2] = even[i] - t_odd;
	
	}

	delete[] even;
	delete[] odd;
}
std::vector<std::complex<float>> FFT::computeFFT(std::vector<std::complex<float>> wave){
	std::vector<std::complex<float>> A = wave;
	/*
	for(int i=0; i < wave.size() ;i++){
		A[i] *= 1;
	}*/

	fft(A.data(),A.size(),log2(A.size()));
	return A;
}