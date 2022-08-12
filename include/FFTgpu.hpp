#include "ocl_boiler.h"
#include <vector>
#include <string>
#include <cmath>

class FFTGpu{

protected:
	std::vector<cl_event> events;
	cl_platform_id platform;
	cl_device_id device;
	cl_context ctx;
	cl_command_queue queue;
	cl_program prog;

public:
	FFTGpu(cl_platform_id p,cl_device_id device,cl_context ctx,cl_command_queue queue,cl_program prog) {
		this->platform = p;
		this->device = device;
		this->ctx = ctx;
		this->queue = queue;
		this->prog = prog;
	}
	virtual ~FFTGpu(){
		for(auto event :events){
			clReleaseEvent(event);
		}
	}
	virtual float* fft(float* samples,cl_int nels) = 0;
	virtual inline std::vector<cl_event> getEvents(){return events;};
	virtual double evaluateSpeed(FILE* f) = 0;
};



class FFTGpu1_2:public virtual FFTGpu{
private:
	cl_mem input;
	cl_mem output;
	size_t preferred_multiple_init;
	cl_kernel fft_kernel;
	std::string kernel_version;
	cl_int nels;
	float* h_array;


private:
	void inizializeMemory(float* samples,cl_int nels){
		const int memsize = sizeof(float) *2 *nels;
		cl_int err;
		input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize, samples, &err);
		ocl_check(err, "create input array");
	
		output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
			ocl_check(err, "create output array");
		};

	void issueKernel(cl_int nels,cl_int iter){
		cl_int err;
		err = clSetKernelArg(fft_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for fft");

		err = clSetKernelArg(fft_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for fft");


		err = clSetKernelArg(fft_kernel, 2, sizeof(nels), &nels);
		ocl_check(err, "set kernel arg 2 for ftt");

		err = clSetKernelArg(fft_kernel, 3, sizeof(int),&iter );
		ocl_check(err, "set kernel arg 3 for ftt");


		size_t gws[] = { round_mul_up(nels/2, preferred_multiple_init) };
		//printf("gws %d\n",gws[0]);
		cl_event fft_gpu_evt;

		if(events.size() > 0)
			err = clEnqueueNDRangeKernel(queue, fft_kernel,
				1, NULL, gws, NULL,
				events.size(), events.data(), &fft_gpu_evt );
		else
			err = clEnqueueNDRangeKernel(queue, fft_kernel,
				1, NULL, gws, NULL,
				0, NULL, &fft_gpu_evt );
		ocl_check(err, "launch kernel fft_k");
		events.emplace_back(fft_gpu_evt);
	}

public:
	FFTGpu1_2(cl_platform_id p,cl_device_id device,cl_context ctx,cl_command_queue queue,cl_program prog,std::string version)
	:FFTGpu(p,device,ctx,queue,prog){
		
		this->kernel_version = version;
		cl_int err;
		fft_kernel = clCreateKernel(prog, version.c_str(), &err);
		clGetKernelWorkGroupInfo(fft_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);
		
	}

	virtual float* fft(float* samples,cl_int nels) override{
	 	this->nels = nels;
		inizializeMemory(samples,nels);
		const int iter = log2(nels);
		const int memsize = sizeof(float)*2*nels;
		
		for(int i=1;i<=iter;i++){
			issueKernel(nels,i);
			cl_mem tmp = input;
			input = output;
			output = tmp;
		}
		cl_mem tmp = input;
		input = output;
		output = tmp;

		cl_event read_evt;
		cl_int err;

		h_array = (float*)clEnqueueMapBuffer(queue, output, CL_TRUE, CL_MAP_READ,
			0, memsize,
			events.size(), events.data(), &read_evt,
			&err);
		ocl_check(err, "map buffer");
		events.emplace_back(read_evt);

		return h_array;
	}

	virtual double evaluateSpeed(FILE* f){
		const int memsize = sizeof(float)*2*nels;
		
		
		for (int i=0;i<events.size()-1;i++) {
			double runtime_pass = runtime_ms(events[i]);
			printf("fft_pass_%d: %gms, %gGB/s, %gGE/s\n",i,runtime_pass,1.0e-6*memsize/runtime_pass,1.0e-6*2*nels/runtime_pass);
		} 

		double runtime_fft = total_runtime_ms(events[0],events[events.size()-2]);
		double runtime_read = runtime_ms(events[events.size()-1]);

		printf("fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
		printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);

		printf("combined: %gms, %gGB/s\n", runtime_read+runtime_fft, 1.0e-6*memsize/(runtime_read+runtime_fft));
		

		if(f != NULL){
  	  		fprintf(f,"fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
			fprintf(f,"read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);

			fprintf(f,"combined: %gms %gGB/s\n",runtime_read+runtime_fft, 1.0e-6*memsize/(runtime_read+runtime_fft));


		}
		return runtime_read+runtime_fft;
	}
	
	virtual ~FFTGpu1_2(){
		cl_event unmap_evt;
		cl_int err;
		err = clEnqueueUnmapMemObject(queue, output, h_array,
			0, NULL, &unmap_evt);
		ocl_check(err, "unmap buffer");

		err = clWaitForEvents(1, &unmap_evt);
		ocl_check(err, "wait for unmap");


		clReleaseKernel(fft_kernel);
		clReleaseMemObject(output);
		clReleaseMemObject(input);
	}
};




class FFTGpu3:public virtual FFTGpu{
private:
	cl_mem input;
	cl_mem output;
	size_t preferred_multiple_init;
	cl_kernel fft_kernel;
	cl_kernel bit_reverse_kernel;

	std::string kernel_version;
	cl_int nels;
	float* h_array;
private:
	void swapIO(){
		cl_mem tmp = input;
		input = output;
		output = tmp;
	}

	void inizializeMemory(float* samples){
		const int memsize = sizeof(float) * 2 *nels;
		cl_int err;
		input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize, samples, &err);
		ocl_check(err, "create input array");
	
		output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
			ocl_check(err, "create output array");
		};


	void setMemoryFFTKernel(cl_int iter){
		cl_int err;
		err = clSetKernelArg(fft_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for fft");

		err = clSetKernelArg(fft_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for fft");


		err = clSetKernelArg(fft_kernel, 2, sizeof(nels), &nels);
		ocl_check(err, "set kernel arg 2 for ftt");

		err = clSetKernelArg(fft_kernel, 3, sizeof(int),&iter);
		ocl_check(err, "set kernel arg 3 for ftt");

	}

	void setMemoryBitReverseKernel(cl_int logLenght){
		cl_int err;
		err = clSetKernelArg(bit_reverse_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for bit_reverse");

		err = clSetKernelArg(bit_reverse_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for bit_reverse");

		err = clSetKernelArg(bit_reverse_kernel, 2, sizeof(logLenght), &logLenght);
		ocl_check(err, "set kernel arg 2 for bit_reverse");
	}

	void issueKernel(cl_kernel kernel,cl_int gws_elements){
		cl_int err;
		size_t gws[] = { round_mul_up(gws_elements, preferred_multiple_init) };
		//printf("gws %d\n",gws[0]);
		cl_event fft_gpu_evt;

		if(events.size() > 0)
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, NULL,
				events.size(), events.data(), &fft_gpu_evt );
		else
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, NULL,
				0, NULL, &fft_gpu_evt );
		ocl_check(err, "launch kernel fft_3");
		events.emplace_back(fft_gpu_evt);
	}

public:
	FFTGpu3(cl_platform_id p,cl_device_id device,cl_context ctx,cl_command_queue queue,cl_program prog)
	:FFTGpu(p,device,ctx,queue,prog){
		
		cl_int err;
		fft_kernel = clCreateKernel(prog, "fft_3", &err);
		ocl_check(err, "create kernel fft_3");

		bit_reverse_kernel = clCreateKernel(prog, "bitReverse", &err);
		ocl_check(err, "create kernel bitReverse");


		clGetKernelWorkGroupInfo(fft_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);
		
	}

	virtual float* fft(float* samples,cl_int nels) override{
	 	this->nels = nels;
		const int memsize = sizeof(float)*2*nels;
		const int iter = log2(nels);
		inizializeMemory(samples);
	
		setMemoryBitReverseKernel(iter);
		issueKernel(bit_reverse_kernel,nels);

		swapIO();
		for(int i=1;i<=iter;i++){
			setMemoryFFTKernel(i);
			issueKernel(fft_kernel,nels/2);
			swapIO();
		}
		swapIO();

		cl_event read_evt;
		cl_int err;

		h_array = (float*)clEnqueueMapBuffer(queue, output, CL_TRUE, CL_MAP_READ,
			0, memsize,
			events.size(), events.data(), &read_evt,
			&err);
		ocl_check(err, "map buffer");
		events.emplace_back(read_evt);

		return h_array;
	}

	virtual double evaluateSpeed(FILE* f){
		const int memsize = sizeof(float)*2*nels;
		
		printf("\n");
		for (int i=1;i<events.size()-1;i++) {
			double runtime_pass = runtime_ms(events[i]);
			printf("fft_pass_%d: %gms, %gGB/s, %gGE/s\n",i-1,runtime_pass,1.0e-6*memsize/runtime_pass,1.0e-6*2*nels/runtime_pass);
		} 
		double runtime_fft = total_runtime_ms(events[0],events[events.size()-2]);
		double runtime_read = runtime_ms(events[events.size()-1]);
		double runtime_bit_reverse = runtime_ms(events[0]);

		printf("\n");
		printf("fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
		printf("bit_reverse: %gms, %gGB/s\n", runtime_bit_reverse, 1.0e-6*memsize/runtime_bit_reverse);
		printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);
		
		double combined = runtime_bit_reverse+ runtime_fft+runtime_read;
		printf("\n");
		
		printf("combined: %gms, %gGB/s %gGE/s\n",combined, 1.0e-6*memsize/combined, 1.0e-6*2*nels/combined);
		printf("\n");


		if(f != NULL){
  	  		fprintf(f,"fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
			fprintf(f,"bit_reverse: %gms, %gGB/s\n", runtime_bit_reverse, 1.0e-6*memsize/runtime_bit_reverse);
			fprintf(f,"read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);


			fprintf(f,"combined: %gms %gGB/s\n",combined, 1.0e-6*memsize/combined);


		}
		
		return combined;
	}
	
	virtual ~FFTGpu3(){
		cl_event unmap_evt;
		cl_int err;
		err = clEnqueueUnmapMemObject(queue, output, h_array,
			0, NULL, &unmap_evt);
		ocl_check(err, "unmap buffer");

		err = clWaitForEvents(1, &unmap_evt);
		ocl_check(err, "wait for unmap");


		clReleaseKernel(fft_kernel);
		clReleaseKernel(bit_reverse_kernel);
		clReleaseMemObject(output);
		clReleaseMemObject(input);
	}
};



class FFTGpu4:public virtual FFTGpu{
private:
	cl_mem input;
	cl_mem output;
	size_t preferred_multiple_init;
	cl_kernel fft_kernel;
	cl_kernel bit_reverse_kernel;
	cl_kernel permutate_array;
	std::string kernel_version;
	cl_int nels;
	float* h_array;

private:
	void swapIO(){
		cl_mem tmp = input;
		input = output;
		output = tmp;
	}

	void inizializeMemory(float* samples){
		const int memsize = sizeof(float) * 2 *nels;
		cl_int err;
		input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize, samples, &err);
		ocl_check(err, "create input array");
	
		output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
			ocl_check(err, "create output array");
		};


	void setMemoryFFTKernel(cl_int logLength,cl_int iter){
		cl_int err;
		err = clSetKernelArg(fft_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for fft4");

		err = clSetKernelArg(fft_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for ftt4");

		err = clSetKernelArg(fft_kernel, 2, sizeof(nels),&nels);
		ocl_check(err, "set kernel arg 2 for ftt4");

		err = clSetKernelArg(fft_kernel, 3, sizeof(iter),&iter);
		ocl_check(err, "set kernel arg 3 for ftt4");

		err = clSetKernelArg(fft_kernel, 4, sizeof(logLength),&logLength);
		ocl_check(err, "set kernel arg 4 for ftt4");

	}

	void setMemoryPermutateArray(cl_int iter){
		cl_int err;
		err = clSetKernelArg(permutate_array, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for fft4");

		err = clSetKernelArg(permutate_array, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for ftt4");

		err = clSetKernelArg(permutate_array, 2, sizeof(nels),&nels);
		ocl_check(err, "set kernel arg 2 for ftt4");

		err = clSetKernelArg(permutate_array, 3, sizeof(iter),&iter);
		ocl_check(err, "set kernel arg 3 for ftt4");

	}

	void setMemoryBitReverseKernel(cl_int logLenght){
		cl_int err;
		err = clSetKernelArg(bit_reverse_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for bit_reverse");

		err = clSetKernelArg(bit_reverse_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for bit_reverse");

		err = clSetKernelArg(bit_reverse_kernel, 2, sizeof(logLenght), &logLenght);
		ocl_check(err, "set kernel arg 2 for bit_reverse");
	}

	void issueKernel(cl_kernel kernel,cl_int gws_elements){
		cl_int err;
		size_t gws[] = { round_mul_up(gws_elements, preferred_multiple_init) };
		//printf("gws %d\n",gws[0]);
		cl_event fft_gpu_evt;

		if(events.size() > 0)
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, NULL,
				events.size(), events.data(), &fft_gpu_evt );
		else
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, NULL,
				0, NULL, &fft_gpu_evt );
		ocl_check(err, "launch kernel fft_4");
		events.emplace_back(fft_gpu_evt);
	}

public:
	FFTGpu4(cl_platform_id p,cl_device_id device,cl_context ctx,cl_command_queue queue,cl_program prog)
	:FFTGpu(p,device,ctx,queue,prog){
		
		cl_int err;
		fft_kernel = clCreateKernel(prog, "fft_4", &err);
		ocl_check(err, "create kernel fft_4");

		bit_reverse_kernel = clCreateKernel(prog, "bitReverse", &err);
		ocl_check(err, "create kernel bitReverse");

		permutate_array = clCreateKernel(prog, "permutateArray", &err);
		ocl_check(err, "create kernel permutateArray");

		clGetKernelWorkGroupInfo(fft_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_multiple_init), &preferred_multiple_init, NULL);
		
	}

	virtual float* fft(float* samples,cl_int nels) override{
	 	this->nels = nels;
		const int memsize = sizeof(float)*2*nels;
		const int iter = log2(nels);
		inizializeMemory(samples);

		setMemoryBitReverseKernel(iter);
		issueKernel(bit_reverse_kernel,nels);
		swapIO();


		for(int i=1;i<=iter;i++){
			setMemoryPermutateArray(i);
			issueKernel(permutate_array,nels);
			swapIO();

			setMemoryFFTKernel(iter,i);
			issueKernel(fft_kernel,nels/2);
			swapIO();
		}
		swapIO();

		cl_event read_evt;
		cl_int err;
		std::cout<<"Mapping Buffer"<<std::endl;

		h_array = (float*)clEnqueueMapBuffer(queue, output, CL_TRUE, CL_MAP_READ,
			0, memsize,
			events.size(), events.data(), &read_evt,
			&err);
		ocl_check(err, "map buffer");
		events.emplace_back(read_evt);

		return h_array;
	}

	virtual double evaluateSpeed(FILE* f){
		const int memsize = sizeof(float)*2*nels;
		
		printf("\n");
		double runtime_bit_reverse = runtime_ms(events[0]);
		double runtime_read = runtime_ms(events[events.size()-1]);
		double first_reverse_pass = runtime_ms(events[1]);
		double runtime_reverse = 0;
		double runtime_fft =0;

		printf("\n");
		printf("reverse_pass%d: %gms, %gGB/s, %gGE/s\n",0,first_reverse_pass,1.0e-6*memsize/first_reverse_pass,1.0e-6*2*nels/first_reverse_pass);
		for (int i=1;i<(events.size()/2)-1;i++) {
			double fft_pass = runtime_ms(events[i*2]);
			double reverse_pass = runtime_ms(events[i*2+1]);
			runtime_fft += fft_pass;
			runtime_reverse += reverse_pass; 
			printf("fft_pass_%d: %gms, %gGB/s, %gGE/s\n",i-1,fft_pass,1.0e-6*memsize/fft_pass,1.0e-6*2*nels/fft_pass);
			printf("permutate_pass_%d: %gms, %gGB/s, %gGE/s\n",i-1,reverse_pass,1.0e-6*memsize/reverse_pass,1.0e-6*2*nels/reverse_pass);
			
		}
		
		printf("\n");
		printf("permutate : %gms, %gGB/s\n", runtime_reverse, 1.0e-6*memsize/runtime_reverse);

		printf("fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
		printf("bit_reverse: %gms, %gGB/s\n", runtime_bit_reverse, 1.0e-6*memsize/runtime_bit_reverse);
		printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);
		
		double combined = runtime_bit_reverse + runtime_fft + runtime_read + runtime_reverse;
		printf("\n");
		
		printf("combined: %gms, %gGB/s %gGE/s\n",combined, 1.0e-6*memsize/combined, 1.0e-6*2*nels/combined);
		printf("\n");
		
		if(f != NULL){
  	  		fprintf(f,"fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
			fprintf(f,"bit_reverse: %gms, %gGB/s\n", runtime_bit_reverse, 1.0e-6*memsize/runtime_bit_reverse);
			fprintf(f,"permutate : %gms, %gGB/s\n", runtime_reverse, 1.0e-6*memsize/runtime_reverse);
			fprintf(f,"read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);


			fprintf(f,"combined: %gms %gGB/s\n",combined, 1.0e-6*memsize/combined);

		}


		return combined;
	}
	
	virtual ~FFTGpu4(){
		cl_event unmap_evt;
		cl_int err;
		err = clEnqueueUnmapMemObject(queue, output, h_array,
			0, NULL, &unmap_evt);
		ocl_check(err, "unmap buffer");

		err = clWaitForEvents(1, &unmap_evt);
		ocl_check(err, "wait for unmap");


		clReleaseKernel(fft_kernel);
		clReleaseKernel(bit_reverse_kernel);
		clReleaseMemObject(output);
		clReleaseMemObject(input);
	}
};

class FFTCompact3 :public virtual FFTGpu{
private:
	cl_mem input;
	cl_mem output;
	cl_kernel fft_kernel;
	std::string kernel_version;
	
	cl_int nels;
	float* h_array;
	size_t maxLws;

private:
	void swapIO(){
		cl_mem tmp = input;
		input = output;
		output = tmp;
	}

	void inizializeMemory(float* samples){
		const int memsize = sizeof(float) * 2 *nels;
		cl_int err;
		input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, memsize, samples, &err);
		ocl_check(err, "create input array");
	
		output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
			ocl_check(err, "create output array");
	};


	void setMemoryFFTKernel(cl_int maxIter){
		cl_int err;
		err = clSetKernelArg(fft_kernel, 0, sizeof(input), &input);
		ocl_check(err, "set kernel arg 0 for compact_fft3");

		err = clSetKernelArg(fft_kernel, 1, sizeof(output), &output);
		ocl_check(err, "set kernel arg 1 for compact_fft3");

		err = clSetKernelArg(fft_kernel, 2, sizeof(nels),&nels);
		ocl_check(err, "set kernel arg 2 for compact_fft3");

		err = clSetKernelArg(fft_kernel, 3, sizeof(maxIter),&maxIter);
		ocl_check(err, "set kernel arg 3 for compact_fft3");

		err = clSetKernelArg(fft_kernel, 4, nels * 2 * sizeof(float) , NULL);
		ocl_check(err, "set kernel arg 4 for compact_fft3");

	}

	void issueKernel(cl_kernel kernel,cl_int gws_elements){
		cl_int err;
		
		
		size_t gws[] = { round_mul_up(gws_elements, maxLws) };

		printf("Issuing kernel with elements %d, gws %zd and  maxLws %zd\n",gws_elements,gws[0],maxLws );

		cl_event fft_gpu_evt;			
		size_t lws[1] = {maxLws};
		
		if(events.size() > 0)
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, lws,
				events.size(), events.data(), &fft_gpu_evt );
		else
			err = clEnqueueNDRangeKernel(queue, kernel,
				1, NULL, gws, lws,
				0, NULL, &fft_gpu_evt );
		ocl_check(err, "launch kernel fft_4");
		events.emplace_back(fft_gpu_evt);
	}

public:
	FFTCompact3(cl_platform_id p,cl_device_id device,cl_context ctx,cl_command_queue queue,cl_program prog,size_t maxLws)
	:FFTGpu(p,device,ctx,queue,prog){
		
		cl_int err;
		fft_kernel = clCreateKernel(prog, "compact_fft_3", &err);
		ocl_check(err, "create kernel compact_fft_3");


		this->maxLws = maxLws;
	}

	virtual float* fft(float* samples,cl_int nels) override{
	 	this->nels = nels;
		const int memsize = sizeof(float)*2*nels;
		const int maxIter = log2(nels);
		
		inizializeMemory(samples);
		setMemoryFFTKernel(maxIter);
		
		issueKernel(fft_kernel,nels/2);
		//swapIO();

		cl_event read_evt;
		cl_int err;
		std::cout<<"Mapping Buffer"<<std::endl;

		h_array = (float*)clEnqueueMapBuffer(queue, output, CL_TRUE, CL_MAP_READ,
			0, memsize,
			events.size(), events.data(), &read_evt,
			&err);
		ocl_check(err, "map buffer");
		events.emplace_back(read_evt);

		return h_array;
	}

	virtual double evaluateSpeed(FILE* f){
		const int memsize = sizeof(float)*2*nels;
		
		printf("\n");
		double runtime_fft = runtime_ms(events[0]);
		double runtime_read = runtime_ms(events[1]);

		printf("fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
		printf("read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);
		
		double combined = runtime_fft + runtime_read ;
		printf("\n");
		
		printf("combined: %gms, %gGB/s %gGE/s\n",combined, 1.0e-6*memsize/combined, 1.0e-6*2*nels/combined);
		printf("\n");
		
		if(f != NULL){
  	  		fprintf(f,"fft: %gms, %gGB/s\n", runtime_fft, 1.0e-6*memsize/runtime_fft);
			fprintf(f,"read: %gms, %gGB/s\n", runtime_read, 1.0e-6*memsize/runtime_read);
			fprintf(f,"combined: %gms %gGB/s\n",combined, 1.0e-6*memsize/combined);
		}


		return combined;
	}
	
	virtual ~FFTCompact3(){
		cl_event unmap_evt;
		cl_int err;
		err = clEnqueueUnmapMemObject(queue, output, h_array,
			0, NULL, &unmap_evt);
		ocl_check(err, "unmap buffer");

		err = clWaitForEvents(1, &unmap_evt);
		ocl_check(err, "wait for unmap");


		clReleaseKernel(fft_kernel);
		clReleaseMemObject(output);
		clReleaseMemObject(input);
	}
	
};
