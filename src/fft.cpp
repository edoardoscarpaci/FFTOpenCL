#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <string>
#include <iostream>
#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include <math.h>
#include "FFT.hpp"
#include "WAV.hpp"

#include <functional>
#include <chrono>

void error(const char * msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

void timeFunction(std::function<uint32_t*(uint32_t)> f,uint32_t N){
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();
    f(N);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    //auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    //std::cout << ms_int.count() << "ms\n";
    std::cout <<"CPU function: " <<ms_double.count() << "ms ," <<1.0e-6*N*sizeof(int)/ms_double.count()<<"GB/s"<<std::endl;
} 

void timeFunctionfft(std::vector<std::complex<float>> wave){
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
	FFT fft = FFT();


	auto t1 = high_resolution_clock::now();
    fft.computeFFT(wave);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    //auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    //std::cout << ms_int.count() << "ms\n";
    std::cout <<"CPU function: " <<ms_double.count() << "ms ," <<1.0e-6*wave.size()*2*sizeof(int)/ms_double.count()<<"GB/s"<<std::endl;
}

uint32_t* bitReversal(uint32_t lenght){
	uint32_t t = log2(lenght);
	const unsigned half_n = 1 << (t-1);
	std::cout<<"half_n: " << half_n << std::endl;
	uint32_t* result = new uint32_t[1 << t];
	result[0] = 0;
	result[1] = half_n;
	for (unsigned n = 1; n < half_n; ++n) {
		const unsigned index = n << 1;
		result[index] = result[n] >> 1;
		result[index+1] = result[index] + half_n;
	}
	return result;
}

void verifyBitReversal(uint32_t* x, uint32_t N){
	uint32_t* trueBitReversal = bitReversal(N);
	for(int i=0; i<N;i++){
		if(x[i] != trueBitReversal[i]){
			std::cout << "Error in index "<< i << " with value: " << x[i] <<" True value: "<<trueBitReversal[i] << std::endl;
			return;
		}
	}
}

float* complexToFloat(std::complex<float>* x,int N){
	float * out  = new float[N*2];
	for(int i=0;i<N;i++){
		out[i*2] = x[i].real();
		out[i*2+1] = x[i].imag();
	}

	return out;
}
template <class T>
void verifyArray(T* x ,T* y,int N){
	for (int i=0;i<N;i++){
		if(x[i] != y[i]){
			std::cout<<"Different value at index {"<< i<< "} with value x: " << x[i] << " y: " << y[i]<<std::endl;
			return;
		}
	}
}

void print16Array(int * x , int N){
	for(int i=0; i<16;i++)
		std::cout << x[i] <<std::endl;
}

cl_event bitReversalInit(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init,
	cl_mem d_array, cl_int nels)
{
	cl_int err;
	err = clSetKernelArg(k, 0, sizeof(d_array), &d_array);
	ocl_check(err, "set kernel arg 0 for init_bit_reversal");
	
	err = clSetKernelArg(k, 1, sizeof(nels), &nels);
	ocl_check(err, "set kernel arg 1 for init_bit_reversal");

	/* Round up to the next multiple of preferred_multiple_init.
	 * ALTERNATIVE (exercise): split the kernel execution in 
	 * a chunk with a nice gws, and a chunk with the remaining elements
	 */
	size_t gws[] = { round_mul_up(nels, preferred_multiple_init) };
	cl_event init_evt;
	err = clEnqueueNDRangeKernel(q, k,
		1, NULL, gws, NULL,
		0, NULL, &init_evt);
	ocl_check(err, "launch kernel init_bit_reversal");
	return init_evt;
}


cl_event fft_gpu(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init,
	cl_mem input,cl_mem output, cl_int nels)
{
	cl_int err;
		err = clSetKernelArg(k, 0, sizeof(input), &input);
	ocl_check(err, "set kernel arg 0 for fft");

	err = clSetKernelArg(k, 1, sizeof(output), &output);
	ocl_check(err, "set kernel arg 1 for fft");
	
	
	err = clSetKernelArg(k, 2, sizeof(nels), &nels);
	ocl_check(err, "set kernel arg 2 for ftt");


	size_t gws[] = { round_mul_up(nels, preferred_multiple_init) };
	cl_event fft_gpu_evt;
	err = clEnqueueNDRangeKernel(q, k,
		1, NULL, gws, NULL,
		0, NULL, &fft_gpu_evt);
	ocl_check(err, "launch kernel init_bit_reversal");
	return fft_gpu_evt;
}

int main(int argc, char *argv[])
{
	if (argc < 2) error("wav file path and cut ");
	
	//const int nels = atoi(argv[1]);
	const std::string path = argv[1];
	//const size_t memsize = nels*sizeof(float)*2;

	const int cut = atoi(argv[2]);

	FFT fft = FFT();
	WAV wavFile = WAV(path);
	std::vector<std::complex<float>> samples = wavFile.getComplexSamples();
	
	std::cout <<"Samples Size before resize: "<< samples.size() <<std::endl;
	
	int prev = pow(2, floor(log(samples.size())/log(2)));
	if(cut < samples.size()){
		prev = cut;
	}
	samples.resize(prev);
	const int nels = samples.size();
	
	std::cout <<"Samples Size Resized: "<< nels <<std::endl;

	
	float* samples_float = complexToFloat(samples.data(),samples.size());

	const int memsize = nels * 2 * sizeof(float);	
	std::cout <<"Memsize: "<< memsize <<std::endl;
	

	/* Initialize OpenCL */
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("bitPermutation.ocl", ctx, d);

	cl_int err;
	cl_kernel init_bit_reversal = clCreateKernel(prog, "fft_1", &err);
	ocl_check(err, "create kernel fft_1");


	cl_mem input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, memsize, samples_float, &err);
	ocl_check(err, "create input array");
	
	cl_mem output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "create output array");


	size_t preferred_multiple_init;
	clGetKernelWorkGroupInfo(init_bit_reversal, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);



	cl_event init_evt = fft_gpu(que, init_bit_reversal, preferred_multiple_init, input, output, nels);
	cl_event read_evt;

	std::cout <<"After fftgpu Here " << std::endl;


	float* h_array = (float*)clEnqueueMapBuffer(que, output, CL_TRUE, CL_MAP_READ,
		0, memsize,
		1, &init_evt, &read_evt,
		&err);
	ocl_check(err, "map buffer");
	
	


	std::vector<std::complex<float>> A = fft.computeFFT(samples);

	std::cout <<"After fft Here " << std::endl;


	verifyArray<float>(h_array,complexToFloat(A.data(),A.size()),nels);
	double runtime_init = runtime_ms(init_evt);
	double runtime_read = runtime_ms(read_evt);

	printf("init: %gms, %gGB/s\n", runtime_init, 1.0e-6*memsize/runtime_init);
	printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);
	
	timeFunctionfft(samples);


	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que, output, h_array,
		0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer");

	err = clWaitForEvents(1, &unmap_evt);
	ocl_check(err, "wait for unmap");

	
	clReleaseEvent(read_evt);
	clReleaseEvent(init_evt);
	clReleaseKernel(init_bit_reversal);
	clReleaseMemObject(output);
	clReleaseMemObject(input);
	

}