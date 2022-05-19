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
#include <iomanip>

#include <fftw3.h>

#include <functional>
#include <chrono>
#include "FFTgpu.hpp"
#include <fstream>

void error(const char * msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

bool cmpf(float A, float B, float epsilon = 0.01f)
{
    return (fabs(A - B) < epsilon);
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

double timeFunctionfft(std::vector<std::complex<float>> wave){
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
	return ms_double.count();
}

double timeFunctionfftwf(fftwf_plan plan,int memsize){
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
	FFT fft = FFT();


	auto t1 = high_resolution_clock::now();
	fftwf_execute(plan);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    //auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    //std::cout << ms_int.count() << "ms\n";
    std::cout <<"FFTW CPU function: " <<ms_double.count() << "ms ," <<1.0e-6*memsize/ms_double.count()<<"GB/s"<<std::endl;
	return ms_double.count();
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


void fillcomplexfftw(float* a , fftwf_complex* b,int N){
	for(int i=0;i<N;i++){
		b[i][0] = a[i*2];
		b[i][1] = a[i*2+1];
	}
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


void verifyArrayFloat(float* x ,float* y,int N){
	for (int i=0;i<N;i++){
		if(!cmpf(x[i],y[i])){
			std::cout<<"Different value at index {"<<i<< "} with value x: " << x[i] << " y: " << y[i]<<std::endl;
			return;
		}
	}
}

int generatePermutation(int index,int logLength){
	int a = index;
	int b = 0;
	int j = 0;	

	while(j++ < logLength){
		b = (b << 1)| (a & 1);
		a >>= 1;
	}
	return b;
}

void print16Array(int * x , int N){
	for(int i=0; i<16;i++)
		std::cout << x[i] <<std::endl;
}


cl_event bitReverseArray(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init,
	cl_mem input,cl_mem output, cl_int nels)
{
	cl_int err;
	err = clSetKernelArg(k, 0, sizeof(input), &input);
	ocl_check(err, "set kernel arg 0 for bit_reverse_array");
	
	err = clSetKernelArg(k, 1, sizeof(output), &output);
	ocl_check(err, "set kernel arg 0 for bit_reverse_array");
	
	int logLenght = log2(nels);
	
	err = clSetKernelArg(k, 2, sizeof(logLenght), &logLenght);
	ocl_check(err, "set kernel arg 1 for bit_reverse_array");

	/* Round up to the next multiple of preferred_multiple_init.
	 * ALTERNATIVE (exercise): split the kernel execution in 
	 * a chunk with a nice gws, and a chunk with the remaining elements
	 */
	size_t gws[] = { round_mul_up(nels, preferred_multiple_init) };
	cl_event init_evt;
	err = clEnqueueNDRangeKernel(q, k,
		1, NULL, gws, NULL,
		0, NULL, &init_evt);
	ocl_check(err, "launch kernel bit_reverse_array");
	return init_evt;
}


cl_event fft_gpu(cl_command_queue q, cl_kernel k, size_t preferred_multiple_init,
	cl_mem input,cl_mem output, cl_int nels, int iter, cl_int num_events_to_wait, cl_event *to_wait)
{
	cl_int err;
	err = clSetKernelArg(k, 0, sizeof(input), &input);
	ocl_check(err, "set kernel arg 0 for fft");

	err = clSetKernelArg(k, 1, sizeof(output), &output);
	ocl_check(err, "set kernel arg 1 for fft");
	
	
	err = clSetKernelArg(k, 2, sizeof(nels), &nels);
	ocl_check(err, "set kernel arg 2 for ftt");

	err = clSetKernelArg(k, 3, sizeof(int),&iter );
	ocl_check(err, "set kernel arg 3 for ftt");


	size_t gws[] = { round_mul_up(nels/2, preferred_multiple_init) };
	//printf("gws %d\n",gws[0]);
	cl_event fft_gpu_evt;
	
	if(num_events_to_wait > 0)
		err = clEnqueueNDRangeKernel(q, k,
			1, NULL, gws, NULL,
			num_events_to_wait, to_wait, &fft_gpu_evt );
	else
		err = clEnqueueNDRangeKernel(q, k,
			1, NULL, gws, NULL,
			0, NULL, &fft_gpu_evt );
	ocl_check(err, "launch kernel fft_k");
	return fft_gpu_evt;
}

int main(int argc, char *argv[])
{
	if (argc < 3) error("wav file path and cut ");
	
	const std::string path = argv[1];
	const int cut = atoi(argv[2]);
	const std::string kernel = argv[3];
	bool writeToFile = false;
	std::string pathToWrite;
	if(argc == 5){
		pathToWrite = argv[4];
		std::cout << pathToWrite<<std::endl;
		writeToFile= true;

	}
	//const size_t memsize = nels*sizeof(float)*2;


	FFT fft = FFT();
	WAV wavFile = WAV(path);
	std::vector<std::complex<float>> samples = wavFile.getComplexSamples();
	
	std::cout <<"Samples Size before resize: "<< samples.size() <<std::endl;
	
	int prev = pow(2, floor(log(samples.size())/log(2)));
	if((size_t)cut < samples.size()){
		prev = cut;
	}

	samples.resize(prev);
	const int nels = samples.size();
	
	std::cout <<"Samples Size Resized: "<< nels <<std::endl;
	const int memsize = nels * 2 * sizeof(float);	
	std::cout <<"Memsize: "<< memsize <<std::endl;
	
	float* samples_float = complexToFloat(samples.data(),samples.size());


	/* Initialize OpenCL */
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("bitPermutation.ocl", ctx, d);
/*
	cl_int err;
	cl_kernel init_bit_reversal = clCreateKernel(prog, kernel.c_str(), &err);
	ocl_check(err, "create kernel fft_k");


	cl_mem input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, memsize, samples_float, &err);
	ocl_check(err, "create input array");
	
	cl_mem output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "create output array");


	size_t preferred_multiple_init;
	clGetKernelWorkGroupInfo(init_bit_reversal, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);
	const int iter = log2(nels);

	cl_event init_evt[iter];
	for(int i=1;i<=iter;i++){
		init_evt[i-1] = fft_gpu(que, init_bit_reversal, preferred_multiple_init, input, output, nels,i, i-1,init_evt);
		cl_mem tmp = input;
		input = output;
		output = tmp;
	}
	cl_mem tmp = input;
	input = output;
	output = tmp;
	
	cl_event read_evt;
	float* h_array = (float*)clEnqueueMapBuffer(que, output, CL_TRUE, CL_MAP_READ,
		0, memsize,
		1, init_evt+(iter-1), &read_evt,
		&err);
	ocl_check(err, "map buffer");
		

	
	double runtime_init = total_runtime_ms(init_evt[0],init_evt[iter-1]);
	double runtime_read = runtime_ms(read_evt);

	for (int pass = 0; pass < iter; ++pass) {
		double runtime_pass = runtime_ms(init_evt[pass]);
		printf("fft_pass_%d: %gms, %gGB/s, %gGE/s\n",pass,runtime_pass,1.0e-6*memsize/runtime_pass,1.0e-6*2*nels/runtime_pass);
	} 

	printf("fft: %gms, %gGB/s\n", runtime_init, 1.0e-6*memsize/runtime_init);
	printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);

	printf("combined: %gms, %gGB/s\n", runtime_read+runtime_init, 1.0e-6*memsize/(runtime_read+runtime_init));
	*/


	//verifyArray<float>(h_array,complexToFloat(A.data(),A.size()),nels);

	std::vector<std::complex<float>> A = fft.computeFFT(samples);

	FFTGpu* fft_gpu;

	if(kernel == "fft_1" || kernel== "fft_2"){
		fft_gpu = new FFTGpu1_2(p,d,ctx,que,prog,kernel);
	}
	else if(kernel == "fft_3"){
		fft_gpu = new FFTGpu3(p,d,ctx,que,prog);
	}
	else if(kernel == "fft_4"){
		fft_gpu = new FFTGpu4(p,d,ctx,que,prog);
	}
	
	float *h_array = fft_gpu->fft(samples_float,nels);
	
		
	/*for(int i=0;i<A.size();i++){
		printf("[%d] cpu:(%f,%f) gpu:(%f,%f)\n",i,A[i].real(),A[i].imag(),h_array[i*2],h_array[i*2+1]);
	}*/

	verifyArrayFloat(h_array,complexToFloat(A.data(),A.size()),nels);

	
	fftwf_complex *in = new fftwf_complex[nels];
	fftwf_complex *out = new fftwf_complex[nels];
	fillcomplexfftw(samples_float,in,nels);

	fftwf_plan  plan_fftw = fftwf_plan_dft_1d(nels, in ,out, FFTW_FORWARD, FFTW_ESTIMATE);

	double combined_fft = fft_gpu->evaluateSpeed();
	double cpu_time = timeFunctionfft(samples);
	double fftw_time = timeFunctionfftwf(plan_fftw,memsize);

	printf("Speedup CPU: %fx \n" ,cpu_time / (combined_fft));
	printf("Speedup FFTW: %fx \n" ,fftw_time / (combined_fft));

	if(writeToFile){
		FILE* f = fopen(pathToWrite.c_str(),"w");

		double combined_fft = fft_gpu->evaluateSpeed();
		double cpu_time = timeFunctionfft(samples);
		double fftw_time = timeFunctionfftwf(plan_fftw,memsize);

  	  	fprintf(f,"CPU function: %fms\n",cpu_time);
  	  	fprintf(f,"GPU function: %fms\n",combined_fft);
  	  	fprintf(f,"FFTW CPU function: %fms\n",fftw_time);

		fprintf(f,"Speedup CPU: %fx \n" ,cpu_time / (combined_fft));
		fprintf(f,"Speedup FFTW: %fx \n" ,fftw_time / (combined_fft));
	}

	delete[] in;
	delete[] out;

}