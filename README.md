# FFTOpenCL

É possibile eseguire gli algoritmi usando il file run.sh.
Le opzioni sono 
	-s per la dimensione dei sample deve essere una potenza di 2,la dimensione dei sample massima é definita dal file in input.
	-k nome del kernel da eseguire fft_1,ftt_2,fft_3,fft_4
	-f é il path del file da usare come input.

Se non sono specificate:
	-s la dimensione di default sará 1024.
	-f sará il file sample.wav
	-k sará fft_3
