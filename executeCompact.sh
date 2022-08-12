for j in $(seq 0 4);do \
for i in $(seq 3 10);do \
mkdir -p ~/Desktop/benchmarks/compact_fft3 && mkdir -p ~/Desktop/benchmarks/compact_fft3/${j} && cd /home/edo/Projects/FFTOpenCL/src && OCL_PLATFORM=2 ../build/fft "/home/edo/Downloads/file_example_WAV_10MG.wav" $((2**${i})) compact_fft_3 ~/Desktop/benchmarks/compact_fft3/${j}/${i}
done;done;