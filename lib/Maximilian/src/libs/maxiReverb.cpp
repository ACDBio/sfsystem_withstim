#include "maxiReverb.h"

maxiReverbFilters::maxiReverbFilters()
{
    a = 0.0f;
    output = 0.0f;
    delay_index = 0;
    feedback = 0.8;
    gain_cof = 0.85;
    delay_line.resize(44100, 0);
}

maxi_float_t maxiReverbFilters::twopoint(maxi_float_t input)
{
    a = 0.5f * (input + a);
    return a;
}

maxi_float_t maxiReverbFilters::comb1(maxi_float_t input,maxi_float_t size)
{
    delay_size = size;
    output = delay_line[delay_index];
    delay_line[delay_index] = input + (feedback * output);
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}


maxi_float_t maxiReverbFilters::combff(maxi_float_t input, maxi_float_t size)
{
    delay_size = size;
    output = input + delay_line[delay_index];
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::combfb(maxi_float_t input, maxi_float_t size, maxi_float_t fb)
{
    // holding delay size allows me to tap line
    delay_size = size;
    output = input + (fb * delay_line[delay_index]);
    delay_line[delay_index] = output;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::lpcombfb(maxi_float_t input, maxi_float_t size, maxi_float_t fb, maxi_float_t cutoff)
{
    // used for freeverb emulation
    // low pass between delay output + feedback
    delay_size = size;
    output = input + (fb * mf.lopass(delay_line[delay_index],(1.0f-cutoff)));
    delay_line[delay_index] = output;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
    return 0.0f;
    
}

maxi_float_t maxiReverbFilters::allpass(maxi_float_t input,maxi_float_t size)
{
    delay_size = size;
    input += delay_line[delay_index] * gain_cof;
    output = delay_line[delay_index] + (input * (-gain_cof));
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    
    return output;
}

maxi_float_t maxiReverbFilters::allpass(maxi_float_t input,maxi_float_t size,maxi_float_t fb)
{
    delay_size = size;
    input += delay_line[delay_index] * fb;
    output = delay_line[delay_index] + (input * (-fb));
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::allpasstap(maxi_float_t input,maxi_float_t size,int tap)
{
    delay_size = size;
    input += delay_line[delay_index] * gain_cof;
    
    int t = delay_index + tap;
    if(t > delay_size -1){
        t -= delay_size;
    }
    output = delay_line[t];
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    
    return output;
}

maxi_float_t maxiReverbFilters::gettap(int tap)
{
    int t = delay_index + tap;
    if(t > delay_size -1){
        t -= delay_size;
    }
    output = delay_line[t];
    return output;
}

maxi_float_t maxiReverbFilters::onetap(maxi_float_t input, maxi_float_t size)
{
    delay_size = size;
    output = delay_line[delay_index];
    delay_line[delay_index] = input;
    delay_index != size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::tapd(maxi_float_t input,maxi_float_t size, maxi_float_t * taps,int numtaps)
{
    output = 0.0f;
    delay_size = size;
    for(int i = 0; i < numtaps ; i++)
    {
        float t = (int)(taps[i] * (size-1));
        int o = delay_index + t;
        if(o > delay_size -1)
            o -= delay_size;
        output += delay_line[o];
    }
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::tapdwgain(maxi_float_t input,maxi_float_t size, maxi_float_t * taps,int numtaps,maxi_float_t * gain)
{
    output = 0.0f;
    delay_size = size;
    for(int i = 0; i < numtaps ; i++)
    {
        float t = (int)(taps[i] * (delay_size-1));
        int o = delay_index + t;
        if(o > delay_size -1)
            o -= delay_size;
        output += gain[i] * delay_line[o];
    }
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}

maxi_float_t maxiReverbFilters::tapdpos(maxi_float_t input,int size, int * taps,int numtaps)
{
    output = 0.0f;
    delay_size = size;
    for(int i = 0; i < numtaps ; i++)
    {
        output += delay_line[taps[i]];
    }
    delay_line[delay_index] = input;
    delay_index != delay_size - 1 ? delay_index++ : delay_index = 0;
    return output;
}


////////////////////////////////////////////////////////////////////////////

maxiReverbBase::maxiReverbBase()
{    
    int dblsize = sizeof(maxi_float_t);
    int numfilterssize = dblsize * numfilters;
    
    // Initialize all necessary memory
    memset(fbcomb, 0,numfilterssize);
    memset(fbap, 0, numfilterssize);
    memset(combgainweight, 0,numfilterssize);
    memset(apgainweight, 0, numfilterssize);
    memset(taps, 0, numfilterssize);
    memset(tapsgain, 0, numfilterssize);
    memset(stereooutput,0,dblsize*2);
    memset(feedbackcombfb, 0,dblsize*2);
    memset(lpcombcutoff,0,dblsize*2);
    // used to calculate delay times
    numsamplesms = (float)maxiSettings::sampleRate/1000.0f;
    output = 0.0;
    accumulator = 0.0;
    numtaps = numfilters;
    for(int i = 0 ; i < numtaps; i++){
        taps[i] = 1.0f/10.0f;
        tapsgain[i] = 0.1;
    }
    taps[0] = 0.5f;
    taps[1] = 0.2;
    tapdellength = maxiSettings::sampleRate/10;
    
    // set up various parameters with reasonable values
    for(int i = 0; i < numfilters; i++){
        fbcomb[i] = 13 * (i+1);
        fbap[i] = 13 * (i+1);
        combgainweight[i] = 1.0;
        apgainweight[i] = 1.0;
        lpcombcutoff[i] = 0.2;
        feedbackcombfb[i] = 0.5;
    }
    fbcomb[0] = 50;
    fbcomb[1] = 100;
    fbcomb[2] = 150;
    
}


maxi_float_t maxiReverbBase::apcombcombo(maxi_float_t input, maxi_float_t gain_coef)
{
    // Implementation of Schroeders first Reverb structure
    // input -> 8 parrallel comb filters -> 4 serial combs -> output + input
    // http://www.music.miami.edu/programs/mue/mue2003/research/jfrenette/chapter_2/chapter_2.html
    // earlyref mimics early reflections, 3 taps
    maxi_float_t t = earlyref.tapdwgain(input, tapdellength, taps, 3, tapsgain);
    maxi_float_t combresult = parallelcomb(t,0,4);
    maxi_float_t allpassresult = serialallpass(combresult,0, 2);
    output =  allpassresult * gain_coef ;
    return output + t;
    //return output;
}


maxi_float_t maxiReverbBase::serialallpass(maxi_float_t input,int firstfilter, int numfilters)
{
    //output = input;
    maxi_float_t t = input;
    limitnumfilters(&numfilters);
    for(int i = 0; i < numfilters; i++){
        t = fArrayAllP[i].allpass(t,fbap[i]);
    }
    output = t;
    return output;
}

maxi_float_t maxiReverbBase::serialallpass(maxi_float_t input,int firstfilter, int numfilters,maxi_float_t feedback)
{
    //output = input;
    maxi_float_t t = input;
    limitnumfilters(&numfilters);
    for(int i = 0; i < numfilters; i++){
        t = fArrayAllP[i].allpass(t,fbap[i],feedback);
    }
    output = t;
    return output;
}

maxi_float_t maxiReverbBase::parallelcomb(maxi_float_t input,int firstfilter, int numfilters)
{
    // set to prime nums
    accumulator = 0.0;
    limitnumfilters(&numfilters);
    for(int i = firstfilter; i < numfilters ; i++){
        accumulator += fArrayTwo[i].combfb(input, fbcomb[i], 0.85);
    }
    return accumulator;
}

maxi_float_t maxiReverbBase::parallellpcomb(maxi_float_t input,int firstfilter,int numfilters)
{
    // set to prime nums
    accumulator = 0.0;
    limitnumfilters(&numfilters);
    for(int i = firstfilter; i < numfilters ; i++)
    {
        accumulator += fArrayTwo[i].lpcombfb(input, fbcomb[i], combgainweight[i], lpcombcutoff[i]);
    }
    return accumulator;
}

void maxiReverbBase::limitnumfilters(int * num)
{
    if(*num > numfilters-1){
        *num = numfilters-1;
    } else if(*num < 0){
        *num = 0;
    }
}

void maxiReverbBase::setcombtimesms(maxi_float_t *times, int numset)
{
    limitnumfilters(&numset);
    for(int i = 0; i < numset; i++){
        fbcomb[i] = mstodellength(times[i]);
    }
}

void maxiReverbBase::setcombtimes(int *times, int numset)
{
    limitnumfilters(&numset);
    for(int i = 0; i < numset ; i++){
        fbcomb[i] = times[i];
    }
}

void maxiReverbBase::setcombfeedback(maxi_float_t *feedback, int numset)
{
    limitnumfilters(&numset);
    for(int i = 0 ; i < numset; i++){
        feedbackcombfb[i] = feedback[i];
    }
}

void maxiReverbBase::setlpcombcutoff(maxi_float_t *cutoff,int numset)
{
    limitnumfilters(&numset);
    for(int i = 0 ; i < numset; i++){
        lpcombcutoff[i] = cutoff[i];
    }
}

void maxiReverbBase::setlpcombcutoffall(maxi_float_t cutoff)
{
    if(cutoff >1.0f) cutoff = 1.0;
    if(cutoff < 0.f ) cutoff = 0.0f;
    for(int i = 0 ; i < numfilters; i++)
    {
        lpcombcutoff[i] = cutoff;
    }
}

void maxiReverbBase::setaptimesms(maxi_float_t *times, int numset)
{
    limitnumfilters(&numset);
    for(int i = 0; i < numset; i++){
        
        fbap[i] = mstodellength(times[i]);
    }
}

void maxiReverbBase::setaptimes(int *times, int numset)
{
    limitnumfilters(&numset);
    for(int i = 0; i < numset; i++){
        fbap[i] = times[i];
    }
}


int maxiReverbBase::mstodellength(maxi_float_t ms)
{
    return (int)(numsamplesms * ms);
}


void maxiReverbBase::setcombweights(maxi_float_t *weights, int numset)
{
    setweights(weights,numset,combgainweight);
}

void maxiReverbBase::setcombweightsall(maxi_float_t feedback)
{
    // can expand this
    if(feedback > 1.0f) feedback = 1.0;
    if(feedback < 0.0f) feedback = 0.0;
    for(int i = 0; i < numfilters; i++){
        combgainweight[i] = feedback;
    }
}
void maxiReverbBase::setapweights(maxi_float_t *weights, int numset)
{
    setweights(weights, numset, apgainweight);
}

void maxiReverbBase::setweights(maxi_float_t *weights, int numset,maxi_float_t * filter)
{
    // used 2.16 from miami
    limitnumfilters(&numset);
    for(int i = 0 ; i < numset; i++){
        filter[i] = weights[i];
        
    }
}

/////////////////////////////////////////////////////////////////

maxiSatReverb::maxiSatReverb() : maxiReverbBase() {
    int ctimes[4] = {778,901,1011,1123};
    setcombtimes(ctimes, 4);
    maxi_float_t cgain[4] = {0.805,0.827,0.783,0.764};
    setcombweights(cgain, 4);
    int atimes[3] = {125,42,12};
    setaptimes(atimes, 3);
    maxi_float_t again[3] = {0.7,0.7,0.7};
    setapweights(again, 3);
}

maxi_float_t maxiSatReverb::play(maxi_float_t input)
{
    // Structure created by Chowning (1971) : https://ccrma.stanford.edu/~jos/pasp/Example_Schroeder_Reverberators.html
    // 4 parallel combs -> 3 serial all pass
    maxi_float_t a = parallelcomb(input,0, 4);
    maxi_float_t b = serialallpass(a,0, 3);
    return b;
}

maxi_float_t* maxiSatReverb::playStereo(maxi_float_t input)
{
    // same as above but with stereo widening
    maxi_float_t a = parallelcomb(input,0, 4);
    maxi_float_t b = serialallpass(a,0, 3);
    stereooutput[0] = b;
    stereooutput[1] = -b;
    return stereooutput;
}

//////////////////////////////////////////////////////////////////

maxiFreeVerb::maxiFreeVerb() : maxiReverbBase() {
    int ctimes[8] = {1557,1617,1491,1422,1277,1356,1188,1116};
    setcombtimes(ctimes, 8);
    maxi_float_t cgain[8];
    maxi_float_t cutoff[8];
    for(int i = 0 ; i < 8; i++){
        cgain[i] = 0.84;
        cutoff[i] = 0.2;
    }
    setcombweights(cgain, 8);
    setlpcombcutoff(cutoff, 8);
    int atimes[4] = {225,556,441,341};
    setaptimes(atimes, 4);
    maxi_float_t again[4] = {0.5,0.5,0.5,0.5};
    setapweights(again, 4);    
}

maxi_float_t maxiFreeVerb::play(maxi_float_t input)
{
    // structure created by Shroeder/Moorer
    // https://ccrma.stanford.edu/~jos/pasp/Freeverb.html
    // delay lengths/ feedback amount etc adapted from this page
    maxi_float_t a = parallellpcomb(input,0, 8);
    maxi_float_t b = serialallpass(a,0, 4);
    return b;
}

maxi_float_t maxiFreeVerb::play(maxi_float_t input,maxi_float_t roomsize,maxi_float_t absorbtion)
{
    // extends controllable freeverb parameters
    setcombweightsall((roomsize*0.10f) + 0.84f);
    setlpcombcutoffall(absorbtion);
    maxi_float_t a = parallellpcomb(input,0, 8);
    maxi_float_t b = serialallpass(a,0, 44);
    return b;
}


//////////////////////////////////////////////////////////////////////////

maxiFreeVerbStereo::maxiFreeVerbStereo() : maxiReverbBase() {
    const int num_combs = 16;
    const int num_ap = 8;
    int stereospread = 23;
    int ctimes[num_combs] = {1557,1617,1491,1422,1277,1356,1188,1116};
    for(int i = num_combs/2; i < num_combs; i++){
        ctimes[i] = ctimes[i-(num_combs/2)] + stereospread;
    }
    setcombtimes(ctimes, num_combs);
    maxi_float_t cgain[num_combs];
    maxi_float_t cutoff[num_combs];
    for(int i = 0 ; i < num_combs; i++){
        cgain[i] = 0.84;
        cutoff[i] = 0.2;
    }
    setcombweights(cgain, num_combs);
    setlpcombcutoff(cutoff, num_combs);
    int atimes[num_ap] = {225,556,441,341};
    for(int i = num_ap/2 ; i < num_ap; i++){
        atimes[i] = atimes[i-(num_ap/2)] + stereospread;
    }
    setaptimes(atimes, num_ap);
    maxi_float_t again[num_ap] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    setapweights(again, num_ap);    
}

maxi_float_t* maxiFreeVerbStereo::playStereo(maxi_float_t input, maxi_float_t roomsize, maxi_float_t absorbtion)
{
    // Two seperate freeverb implens, each with different feedback/delay/filter lengths
    setcombweightsall((roomsize*0.10f) + 0.84f);
    setlpcombcutoffall(absorbtion);
    maxi_float_t l1 = parallelcomb(input, 0, 8);
    maxi_float_t l2 = serialallpass(l1,0, 4);
    maxi_float_t r1 = parallelcomb(input, 8, 8);
    maxi_float_t r2 = serialallpass(r1,4,4);
    
    stereooutput[0] = l2;
    stereooutput[1] = r2;
    
    return stereooutput;
}

//////////////////////////////////////////////////////////////////////////////

maxiDattaroReverb::maxiDattaroReverb() : maxiReverbBase() {
    memset(dattorogains,0,sizeof(maxi_float_t)*4);
    memset(dattarotapspos,0,sizeof(int)*numdattarotappos);
    memset(dattorotap,0,sizeof(maxi_float_t)*numdattarotaps);
    memset(dattarofixdellengths,0,sizeof(int)*4);
    memset(maxideltimes,0,sizeof(int)*4);
    sigl = 0.0;
    sigr = 0.0;
    
    // original dattaro del tap lengths for 29.8 khz SR
    int orig[numdattarotappos] = { 266,2974,1913,1996,1990,187,1066,353,3627,1228,2673,2111,335,121 };
    // 1 ms at 29.8 khz SR;
    float dms = 29.8;
    float cms = (float)maxiSettings::sampleRate/1000.0f;
    for(int i = 0 ; i < numdattarotappos; i++){
        float prevdelengthms = (float)orig[i]/dms;
        int newdellength = floor(prevdelengthms * cms);
        dattarotapspos[i] = newdellength;
    }
    // orig dattaro del lengths
    const int numfixeddelays = 5;
    int origdel[numfixeddelays] = { 4217,3163,4453,3720 };
    for(int i = 0 ; i < numfixeddelays; i++){
        float prevdellengthms = (float)origdel[i]/dms;
        int newdellength = floor(prevdellengthms * cms);
        dattarofixdellengths[i] = newdellength;
    }
    // initial delay
    dattarofixdellengths[4] = 3100;
    // set gains
    const int numdattarogains = 5;
    maxi_float_t presetgains[numdattarogains] = { 0.75,0.625,0.7,0.5,0.3 };
    //    int presetgains[numdattarogains] = { 0.999,0.9999,0.7,0.5,0.99999999 };
    for(int i = 0 ; i < numdattarogains; i++){
        dattorogains[i] = presetgains[i];
    }
    
    // set fb ap
    const int numinitap = 8;
    int initaplengthsorig[numinitap] = { 142,107,379,277,908,2656,672,1800 };
    for(int i = 0 ; i < numinitap; i++){
        float prevlength = (float)initaplengthsorig[i]/dms;
        int newlength = floor(prevlength * cms);
        fbap[i] = newlength;
    }    
}

maxi_float_t* maxiDattaroReverb::playStereo(maxi_float_t input) {
    // implementation of reverb designed by Jon Dattoro
    // https://ccrma.stanford.edu/~dattorro/EffectDesignPart1.pdf
    // Sigificantly more complex than the previous examples.
    // 1.0f The inital reflctions are subject to more processing
    // 2.f Interaction and feedback between two channels
    //    for great stereo reverb
    // 3. Low pass filters used throughout the structure
    //    to alter response
    // 4. Output is composed of taps from various points
    //    in the structure, rather than being taken
    //    off from the end. The way this is done really
    //    flattens the frequency response associated with combs
    //
    // Delay lengths/feedback amount etc are all set in the setReverb function.
    // All above params adapated from the paper and subjected to further tuning
    // from myself.
    
    maxi_float_t a = maxiDelays[4].onetap(input, dattarofixdellengths[4]);
    
    maxi_float_t b = fArrayLP[0].lopass(input, 0.8);
    maxi_float_t c = serialallpass(b, 0, 2,dattorogains[0]);
    maxi_float_t d = serialallpass(c, 2, 2,dattorogains[1]);
    maxi_float_t tsigl = sigl;
    
    sigl = d + dattorogains[4] * sigr ;
    sigr = d + dattorogains[4] * tsigl ;
    
    sigl = fArrayAllP[4].allpass(sigl,fbap[4],dattorogains[2]);
    sigl = maxiDelays[0].onetap(sigl, dattarofixdellengths[0]);
    dattorotap[0] = maxiDelays[0].gettap(dattarotapspos[0]);
    dattorotap[1] = maxiDelays[0].gettap(dattarotapspos[1]);
    dattorotap[11] = maxiDelays[0].gettap(dattarotapspos[11]);
    sigl = fArrayLP[1].lopass(sigl, 0.4);
    sigl = fArrayAllP[5].allpass(sigl,fbap[5],dattorogains[3]);
    dattorotap[2] = fArrayAllP[5].gettap(dattarotapspos[2]);
    dattorotap[12]= fArrayAllP[5].gettap(dattarotapspos[12]);
    sigl = maxiDelays[1].onetap(sigl, dattarofixdellengths[1]);
    dattorotap[3] = maxiDelays[1].gettap(dattarotapspos[3]);
    dattorotap[13] = maxiDelays[1].gettap(dattarotapspos[13]);
    
    sigr = fArrayAllP[6].allpass(sigr,fbap[6],dattorogains[2]);
    sigr = maxiDelays[2].onetap(sigr, dattarofixdellengths[2]);
    dattorotap[4] = maxiDelays[2].gettap(dattarotapspos[4]);
    dattorotap[7] = maxiDelays[2].gettap(dattarotapspos[7]);
    dattorotap[8] = maxiDelays[2].gettap(dattarotapspos[8]);
    sigr = fArrayLP[2].lopass(sigr, 0.4);
    sigr = fArrayAllP[7].allpass(sigr, fbap[7],dattorogains[3]);
    dattorotap[5] = fArrayAllP[7].gettap(dattarotapspos[5]);
    dattorotap[9] = fArrayAllP[7].gettap(dattarotapspos[9]);
    sigr = maxiDelays[3].onetap(sigr, dattarofixdellengths[3]);
    dattorotap[6] = maxiDelays[3].gettap(dattarotapspos[6]);
    dattorotap[10] = maxiDelays[3].gettap(dattarotapspos[10]);
    
    stereooutput[0] = dattorotap[0] + dattorotap[1] - dattorotap[2] +
    dattorotap[3] - dattorotap[4] - dattorotap[5] - dattorotap[6];
    stereooutput[1] = dattorotap[7] + dattorotap[8] - dattorotap[9] +
    dattorotap[10] - dattorotap[11] - dattorotap[12] - dattorotap[13];
    
    return stereooutput;
}
