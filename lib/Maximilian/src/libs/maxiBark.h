/*
 *  maxiBark.cpp
 *  Bark scale loudness
 *
 *  Created by Jakub on 01/12/2014.
 *  Copyright 2014 Goldsmiths Creative Computing. All rights reserved.
 *
 */

#pragma once
//#pragma pack(16)
#include "../maximilian_types.h"

#include "maxiFFT.h"
#include <math.h>
#include <iostream>
//#include <algorithm>
#include <cstdlib>
#ifdef __APPLE_CC__
#include <Accelerate/Accelerate.h>
#endif

using namespace std;

//convert to Bark scale (Zwicker, 1961)
inline maxi_float_t hzToBark(maxi_float_t hz) {
    return 13.0f * atanf(hz/1315.8f) + 3.5f*atan(powf((hz/7518.0f),2));
}

inline maxi_float_t binToHz(unsigned int bin, unsigned int sR, unsigned int bS) {
    return bin*sR/bS;
}

// is T used anywhere?
template <class T>

class maxiBarkScaleAnalyser {
public:
    int NUM_BARK_BANDS;
    
    void setup(unsigned int sR, unsigned int bS) {
        this->sampleRate = sR;
        this->bufferSize = bS;
        specSize = bS/2;
        NUM_BARK_BANDS = 24;
        for (int i=0; i<specSize; i++) {
            barkScale[i] = hzToBark(binToHz(i, sR, bS));
        }
        
        bbLimits[0] = 0;
        int currentBandEnd = barkScale[specSize-1]/NUM_BARK_BANDS;
        int currentBand = 1;
        
        for(int i = 0; i<specSize; i++){
            while(barkScale[i] > currentBandEnd) {
                bbLimits[currentBand] = i;
                currentBand++;
                currentBandEnd = currentBand*barkScale[specSize-1]/NUM_BARK_BANDS;
            }
        }
        
        bbLimits[NUM_BARK_BANDS] = specSize-1;
    };
    
    maxi_float_t* specificLoudness(float* normalisedSpectrum) {
        for (int i = 0; i < NUM_BARK_BANDS; i++){
            maxi_float_t sum = 0;
            for (int j = bbLimits[i] ; j < bbLimits[i+1] ; j++) {
                
                sum += normalisedSpectrum[j];
            }
            specific[i] = pow(sum,0.23);
        }
        
        return specific;
    };
    
    maxi_float_t* relativeLoudness(float* normalisedSpectrum) {
        for (int i = 0; i < NUM_BARK_BANDS; i++){
            maxi_float_t sum = 0;
            for (int j = bbLimits[i] ; j < bbLimits[i+1] ; j++) {
                
                sum += normalisedSpectrum[j];
            }
            specific[i] = pow(sum,0.23);
        }
        
        maxi_float_t max = 0;
        for (int i = 0; i < NUM_BARK_BANDS; i++){
            if (specific[i] > max) max = specific[i];
        }
        
        for (int i = 0; i < NUM_BARK_BANDS; i++){
            relative[i] = specific[i]/max;
        }
        
        return relative;
    };
    
    maxi_float_t* totalLoudness(float* normalisedSpectrum) {
        for (int i = 0; i < NUM_BARK_BANDS; i++){
            maxi_float_t sum = 0;
            for (int j = bbLimits[i] ; j < bbLimits[i+1] ; j++) {
                
                sum += normalisedSpectrum[j];
            }
            specific[i] = pow(sum,0.23);
        }
        
        total[0] = 0;
        
        for (int i = 0; i < 24; i++){
            total[0] += specific[i];
        }
        
        return total;
    };
    
private:
    int bbLimits[24];
    unsigned int sampleRate, bufferSize, specSize;
    maxi_float_t barkScale[2048];
    maxi_float_t specific[24];
    maxi_float_t relative[24];
    maxi_float_t total[1];
    
};

typedef maxiBarkScaleAnalyser<maxi_float_t> maxiBark;






