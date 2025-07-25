/*
 *  maximilian
 *  platform independent synthesis library using portaudio or rtaudio
 *
 *  Created by Mick Grierson on 29/12/2009.
 *  Copyright 2009 Mick Grierson & Strangeloop Limited. All rights reserved.
 *	Thanks to the Goldsmiths Creative Computing Team.
 *	Special thanks to Arturo Castro for the PortAudio implementation.
 * 
 *	Permission is hereby granted, free of charge, to any person
 *	obtaining a copy of this software and associated documentation
 *	files (the "Software"), to deal in the Software without
 *	restriction, including without limitation the rights to use,
 *	copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the
 *	Software is furnished to do so, subject to the following
 *	conditions:
 *	
 *	The above copyright notice and this permission notice shall be
 *	included in all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,	
 *	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *	OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *	NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *	HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *	WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *	OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef _FFT
#define _FFT

#include "maxiMalloc.h"

#ifdef ARDUINO 
#undef __APPLE_CC__
#endif

#undef M_PI

#include <vector>
#ifdef __APPLE_CC__
#include <Accelerate/Accelerate.h>
#endif



class fft {
const float	M_PI =	3.14159265358979323846f; /* pi */
	
public:
//    fft();
//    fft(int fftSize);
    fft(){};
//    fft(fft const &other);
	~fft();
	
    void setup(int fftSize);
	int n; //fftSize
	int half; //halfFFTSize
	
//    float            *in_real, *out_real, *in_img, *out_img;
    std::vector<float> in_real,out_real,in_img,out_img;
    
    float * getReal();
    float * getImg();
    
#ifdef __APPLE_CC__
	int log2n; //log2(n);
    FFTSetup        setupReal = NULL;
    COMPLEX_SPLIT   A;
    bool issetup = false;
    std::vector<float> realp, imagp;
	std::vector<float> polar;
    
    void calcFFT_vdsp(float *data, float *window);
    void cartToPol_vdsp(float *magnitude,float *phase);
	void powerSpectrum_vdsp(int start, float *data, float *window, float *magnitude,float *phase);
    
    void polToCart_vdsp(float *magnitude,float *phase);
    void calcIFFT_vdsp(float *finalOut, float *window);
    void inverseFFTComplex_vdsp(int start, float *finalOut, float *window, float *real, float *imaginary);
	void inversePowerSpectrum_vdsp(int start, float *finalOut, float *window, float *magnitude,float *phase);
	void convToDB_vdsp(float *in, float *out);
#endif
	
	/* Calculate the power spectrum */
    void calcFFT(int start, float *data, float *window);
    void cartToPol(float *magnitude,float *phase);
	void powerSpectrum(int start, float *data, float *window, float *magnitude, float *phase);
	/* ... the inverse */
    void polToCart(float *magnitude,float *phase);
    void calcIFFT(int start, float *finalOut, float *window);
    void inverseFFTComplex(int start, float *finalOut, float *window, float *real, float *imaginary);
	void inversePowerSpectrum(int start, float *finalOut, float *window, float *magnitude,float *phase);
	void convToDB(float *in, float *out);
    
	static void genWindow(int whichFunction, int NumSamples, float *window);
	
};


#endif	
