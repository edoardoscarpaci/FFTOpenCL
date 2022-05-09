#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <string>
#include <iostream>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include "FFT.hpp"
#include "WAV.hpp"

void error(const char * msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

int main(int argc, char *argv[])
{
	if (argc < 3) error("please specify number of elements and wav file path ");
	
	const int nels = atoi(argv[1]);
	const std::string path = argv[2];
	const size_t memsize = nels*sizeof(int);

	/* Initialize OpenCL */
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);

	FFT fft = FFT();
	WAV wavFile = WAV(path);

	std::vector<std::complex<float>> samples = wavFile.getComplexSamples();
	std::cout << "Here" <<std::endl;

	std::vector<std::complex<float>> A = fft.computeFFT(samples);

	for(auto sample :  samples){
		std::cout << sample<<std::endl;
	}

}