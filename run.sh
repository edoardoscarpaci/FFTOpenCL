#! /bin/sh
size=8;
input_file="../sample.wav"
kernel_name="fft_3"
platform="2"

while getopts k:s:f:p: flag
do
    case "${flag}" in
        k) kernel_name=${OPTARG};;
		s) size=${OPTARG};;
		f) input_file=${OPTARG};;
		p) platform=${OPTARG};;

    esac
done


cd src && OCL_PLATFORM=${platform} ../build/fft ${input_file} ${size} ${kernel_name} 