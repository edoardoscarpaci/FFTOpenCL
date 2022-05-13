#include "FFT.hpp"
#include <iostream>

FFT::FFT(){
}

void FFT::fft(std::complex<float>* A, size_t N,int iter){
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

	fft(even,N/2,iter-1);
	fft(odd,N/2,iter-1);
	//std::cout<< "Iter: " << iter<<std::endl;
	for(int i=0;i<N/2;i++){	
		std::complex<float> t = std::exp(std::complex<float>(0, -2 * M_PI * i / N));
		
		if(t.real() > -0.000001 && t.real() < 0 ){
			//std::cout << "Changing t: " << t <<std::endl;
			t = std::complex<float>(0,t.imag());
		}
		std::complex<float> t_odd  = t * odd[i];
		//std::cout << "t_odd: " << t <<std::endl;
	

		A[i] = even[i]  + t_odd;
		A[i + N/2] = even[i] - t_odd;
		
		/*if(iter == 3){
			std::cout <<"["<<i<<"] "<<"Even: " << even[i] << " Odd: " << odd[i] <<" T: "<< t<<" T_odd:"<<t_odd<<std::endl;
			//std::cout <<"Results:["<<i<<"] "<<"Even: " << even[i]+t_odd << " Odd: " << even[i] - t_odd <<std::endl;
		}*/
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

	fft(A.data(),A.size(),log2(A.size()));
	return A;
}