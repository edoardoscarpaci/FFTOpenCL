#!/bin/bash
makeFile='false';
size=8;
benchmarkMode='false'
n_iter=4
max_exp=20
file_path=""
while getopts mk:s:bn:f: flag
do
    case "${flag}" in
        m) makeFile='true';;
        k) kernel_name=${OPTARG};;
		s) size=${OPTARG};;
		b) benchmarkMode='true';;
		n) n_iter=${OPTARG};;
		e) max_exp=${OPTARG};;
		f) file_path=${OPTARG};;
    esac
done

shopt -s xpg_echo

if $benchmarkMode; then
	for i in $(seq 3 10);do \
	mkdir -p ~/Desktop/benchmarks/compact_fft3 && mkdir -p ~/Desktop/benchmarks/compact_fft3/${j} && cd /home/edo/Projects/FFTOpenCL/src && OCL_PLATFORM=2 ../build/fft "/home/edo/Downloads/file_example_WAV_10MG.wav" $((2**${i})) fft_3 ~/Desktop/benchmarks/compact_fft_3/${j}/${i}
	done;
else
	if $makeFile; then
		cd /home/edo/Projects/FFTOpenCL/ \
		&& rm -f build/* && make \
		&& cd src && OCL_PLATFORM=2 ../build/fft "/home/edo/Downloads/file_example_WAV_10MG.wav" ${size} ${kernel_name} ${file_path}
	else 	
		cd /home/edo/Projects/FFTOpenCL/src && OCL_PLATFORM=2 /home/edo/Projects/FFTOpenCL/build/fft "/home/edo/Downloads/file_example_WAV_10MG.wav" ${size} ${kernel_name}
	fi;	
fi;