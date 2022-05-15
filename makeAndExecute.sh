#!/bin/bash
makeFile='false';
size=8;
benchmarkMode='false'
n_iter=4
max_exp=20
while getopts mk:s:bn: flag
do
    case "${flag}" in
        m) makeFile='true';;
        k) kernel_name=${OPTARG};;
		s) size=${OPTARG};;
		b) benchmarkMode='true';;
		n) n_iter=${OPTARG};;
		e) max_exp=${OPTARG};;
    esac
done

shopt -s xpg_echo

if $benchmarkMode; then
 	for j in $(seq 0 4);do \
	for i in $(seq 3 $max_exp);do \
	echo ${i} $(cd /home/edo/Projects/FFTOpenCL/src && OCL_PLATFORM=2 ../build/fft "/home/edo/Downloads/file_example_WAV_10MG.wav" $((2**${i})) ${kernel_name} | grep "Speedup") \
	>> ~/Desktop/benchmarks/${kernel_name}_${j};
	done;done;

else
	if $makeFile; then
		cd /home/edo/Projects/FFTOpenCL/ \
		&& rm -f build/* && make \
		&& cd src && OCL_PLATFORM=2 ../build/fft /home/edo/Downloads/file_example_WAV_10MG.wav $size ${kernel_name}
	else 
		cd /home/edo/Projects/FFTOpenCL/src && OCL_PLATFORM=2 /home/edo/Projects/FFTOpenCL/build/fft /home/edo/Downloads/file_example_WAV_10MG.wav $size ${kernel_name}
	fi;
fi;