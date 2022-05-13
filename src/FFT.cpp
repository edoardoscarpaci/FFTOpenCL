#include "FFT.hpp"
#include <iostream>

FFT::FFT(){
}

void FFT::fft(std::complex<float>* A, size_t N){
	//std::cout << "Calling fft: " << N <<std::endl; 

	if (N<=1) return;


	std::complex<float>* even = new std::complex<float>[N/2];
	std::complex<float>* odd = new std::complex<float>[N/2];

	//std::cout << "Before Divide N: " << N <<std::endl; 


	for(size_t i=0;i<N/2;i++){
		even[i] = A[i*2];
		odd[i] = A[i*2+1];
	}
	//std::cout << "Before fft divide N: " << N <<std::endl; 

	fft(even,N/2);
	fft(odd,N/2);
	
	for(int i=0;i<N/2;i++){
		std::complex<float> t = exp(std::complex<float>(0, -2 * M_PI * i / N));
		//std::cout<<"Even:" << even[i]<<" Odd:"<< odd[i] <<std::endl; 
		//std::cout << "t: " << t <<std::endl; 
		t *= odd[i];
		A[i] = even[i]  + t;
		A[i + N/2] = even[i] - t;
	}

	delete even;
	delete odd;
}
std::vector<std::complex<float>> FFT::computeFFT(std::vector<std::complex<float>> wave){
	std::vector<std::complex<float>> A = wave;
	/*
	for(int i=0; i < wave.size() ;i++){
		A[i] *= 1;
	}*/

	fft(A.data(),A.size());
	return A;
}