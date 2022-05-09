#include "FFT.hpp"


FFT::FFT(){
}

void FFT::fft(std::complex<float>* A, size_t N){
	if (N<=1) return;


	std::complex<float> even[N/2];
	std::complex<float> odd[N/2];

	for(size_t i=0;i<N/2-1;i++){
		even[i] = A[i*2];
		odd[i] = A[i*2+1];
	}

	fft(even,N/2);
	fft(odd,N/2);

	for(int i=0;i<N/2;i++){
		std::complex<float> t = exp(std::complex<float>(0, -2 * M_PI * i / N)) * odd[i];
		A[i] = even[i]  + t;
		A[i + N/2] = even[i] - t;
	}


}
std::vector<std::complex<float>> FFT::computeFFT(std::vector<std::complex<float>> wave){
	std::vector<std::complex<float>> A = wave;
	
	for(int i=0; i < wave.size() ;i++){
		A[i] *= 1;
	}

	fft(A.data(),A.size());
	return A;
}