#include <Arduino.h>
#include <Adafruit_I2CDevice.h>
#include "ADS1256.h"
#include "Wire.h"
#include <Adafruit_NeoPixel.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <GyverEncoder.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiClientSecure.h>

#include <WebSocketsServer.h>
#include <string>
#include "AudioTools.h"
#include "AudioLibs/MaximilianDSP.h"
//wave_freq - ch1, wave_freq - ch2, panner_freq, panner_div, phasor1_freq, phasor1min, phasor1_max, phasor2_freq, phasor2min, phasor2_max


WiFiMulti WiFiMulti;
WebSocketsServer webSocket = WebSocketsServer(80);





//#define Serial. Serial1

//pin config
#define MOSI 23
#define MISO 19
#define SCK 18
#define CS_ADS1256 5
#define CS_SD 17
#define DRDY_ADS1256 26
#define RESET_ADS1256 25
#define LED_PIN 13  
#define SDA 21
#define SCL 22
#define ENC_CLK 35
#define ENC_DT 32
#define ENC_SW 33



#define I2S_BCK 14;
#define I2S_WS 15;
#define I2S_DATA 27;


//other definitions
#define NUM_LEDS 8

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C


//Sound design system definition
I2SStream out;
Maximilian maximilian(out);
maxiOsc myOsc1,myAutoPanner, myOsc2;//
vector<float> myStereoOutput1(2,0);
vector<float> myStereoOutput2(2,0);
maxiMix myOutputs;//this is the stereo mixer channel.
maxiOsc myPhasor1, myPhasor2, noise, myOsc3, myOsc4;

int wave_1_freq;
int wave_2_freq;
int panner_freq;
int panner_div;
int phasor1_freq;
int phasor1_max;
int phasor1_min;
int phasor2_freq;
int phasor2_max;
int phasor2_min;
float maxivolume;

int wave_1_type;
int wave_2_type;
  // 0. noise
  // 1. sinewave
  // 2. square
  // 3. triangle
  // notimp 4. triangle
  // notimp 5. impulse

uint32_t color1;
uint32_t color2;
uint32_t color3;
uint32_t color4;
uint32_t color5;
uint32_t color6;
uint32_t color7;
uint32_t color8;

void config_audio(){
    auto config = out.defaultConfig(TX_MODE);
        config.i2s_format = I2S_LSB_FORMAT;
        config.pin_bck = I2S_BCK;
        config.pin_ws = I2S_WS;
        config.pin_data = I2S_DATA;
        out.begin(config);
    maximilian.begin(config);
    maximilian.setVolume(0.0);
}

void play(float *output){
  if ((wave_1_type==0) & (wave_2_type==0)) {
    myOutputs.stereo(myOsc1.noise(),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.noise(),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==1) & (wave_2_type==0)) {
    myOutputs.stereo(myOsc1.sinewave(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.noise(),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==2) & (wave_2_type==0)) {
    myOutputs.stereo(myOsc1.square(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.noise(),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==3) & (wave_2_type==0)) {
    myOutputs.stereo(myOsc1.triangle(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.noise(),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }



  if ((wave_1_type==0) & (wave_2_type==1)) {
    myOutputs.stereo(myOsc1.noise(),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.sinewave(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==1) & (wave_2_type==1)) {
    myOutputs.stereo(myOsc1.sinewave(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.sinewave(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==2) & (wave_2_type==1)) {
    myOutputs.stereo(myOsc1.square(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.sinewave(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==3) & (wave_2_type==1)) {
    myOutputs.stereo(myOsc1.triangle(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.sinewave(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }


  if ((wave_1_type==0) & (wave_2_type==2)) {
    myOutputs.stereo(myOsc1.noise(),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.square(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
    if ((wave_1_type==1) & (wave_2_type==2)) {
    myOutputs.stereo(myOsc1.sinewave(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.square(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==2) & (wave_2_type==2)) {
    myOutputs.stereo(myOsc1.square(wave_2_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.square(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==3) & (wave_2_type==2)) {
    myOutputs.stereo(myOsc1.triangle(wave_2_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.square(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }


  if ((wave_1_type==0) & (wave_2_type==3)) {
    myOutputs.stereo(myOsc1.noise(),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.triangle(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==1) & (wave_2_type==3)) {
    myOutputs.stereo(myOsc1.sinewave(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.triangle(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==2) & (wave_2_type==3)) {
    myOutputs.stereo(myOsc1.square(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.triangle(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
  if ((wave_1_type==3) & (wave_2_type==3)) {
    myOutputs.stereo(myOsc1.triangle(wave_1_freq),myStereoOutput1,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);//Stereo, Quad or 8 Channel. Specify the input to be mixed, the output[numberofchannels], and the pan (0-1,equal power).
    myOutputs.stereo(myOsc2.triangle(wave_2_freq),myStereoOutput2,(myAutoPanner.sinewave(panner_freq)+1)/panner_div);
    output[0]=(myStereoOutput1[0]+myStereoOutput2[0])*myOsc3.sinewave(myPhasor1.phasorBetween(phasor1_freq,phasor1_min,phasor1_max));//When working with mixing, you need to specify the outputs explicitly
    output[1]=(myStereoOutput1[1]+myStereoOutput2[1])*myOsc4.sinewave(myPhasor2.phasorBetween(phasor2_freq, phasor2_min, phasor2_max));
  }
}

//sound design system definition - end


//other vars
uint8_t clientID;

//ADS1256 setup
ADS1256 ads;


//LED setup
Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);
uint8_t led_brightness = 100; //The brightness is a value between 0 (min brightness) and 255 (max brightness).

//OLED setup
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//Encoder setup
Encoder enc(ENC_CLK, ENC_DT, ENC_SW);
//Encoder enc(ENC_CLK, ENC_DT, ENC_SW, TYPE1);


//Custom functions
void turnOffDisplay() {
  display.ssd1306_command(SSD1306_DISPLAYOFF);
}

void turnOnDisplay() {
  display.ssd1306_command(SSD1306_DISPLAYON);
}

void hexdump(const void *mem, uint32_t len, uint8_t cols = 16) {
	const uint8_t* src = (const uint8_t*) mem;
	Serial.printf("\n[HEXDUMP] Address: 0x%08X len: 0x%X (%d)", (ptrdiff_t)src, len, len);
	for(uint32_t i = 0; i < len; i++) {
		if(i % cols == 0) {
			Serial.printf("\n[0x%08X] 0x%08X: ", (ptrdiff_t)src, i);
		}
		Serial.printf("%02X ", *src);
		src++;
	}
	Serial.printf("\n");
}

bool data_transfer;
int n_datapoints;
bool set_data_transfer_buffer;
uint8_t delay_length;
bool set_delay;
bool set_delay_and_data_transfer_buffer_step1;
bool set_delay_and_data_transfer_buffer_step2;
bool receive_outputcontrol_data;
bool run_led_cycle;
int leddelay;
bool use_leddelay;
bool only_pos_enc_mode;

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  clientID = num;
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            break;
        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[%u] Connected from %d.%d.%d.%d url: %s\n", num, ip[0], ip[1], ip[2], ip[3], payload);
                delay_length = 10;
                n_datapoints = 1;
                data_transfer = false;
                set_data_transfer_buffer = false;
                set_delay = false;
                only_pos_enc_mode = false;
				// send message to client
				webSocket.sendTXT(num, "Connected");
            }
            break;
        case WStype_TEXT:
            Serial.printf("[%u] get Text: %s\n", num, payload);
            // send message to client
            //webSocket.sendTXT(num, "Text message received!");

            if (strcmp((const char*)payload, "stop_led_cycle") == 0) {
              //run_led_cycle=false;
              color1 = strip.Color(0, 0, 0); // Red
              color2 = strip.Color(0, 0, 0); // Green
              color3 = strip.Color(0, 0, 0); // Blue
              color4 = strip.Color(0, 0, 0); // White
              color5 = strip.Color(0, 0, 0); // Cyan
              color6 = strip.Color(0, 0, 0); // Magenta
              color7 = strip.Color(0, 0, 0); // Yellow
              color8 = strip.Color(0, 0, 0); // Red
              delay(500);
              strip.fill(strip.Color(0, 0, 0)); // If not working - check the JST connection - it should be resoldered.
              delay(500);
              strip.show(); // Update the LEDs
              maximilian.setVolume(0.0); //turn off the sounds

            };
            if (strcmp((const char*)payload, "receive_output_control_data") == 0) {
              data_transfer = false;
              set_data_transfer_buffer = false;
              set_delay = false;
              set_delay_and_data_transfer_buffer_step1 = false;
              set_delay_and_data_transfer_buffer_step2 = false;
              receive_outputcontrol_data = true;
            };
            if (strcmp((const char*)payload, "use_only_pos_enc_mode") == 0) {
              only_pos_enc_mode  = true;
              webSocket.sendTXT(num, "Encoder data will be used regardless of rotation direction");
            };
            if (strcmp((const char*)payload, "use_directional_enc_mode") == 0) {
              only_pos_enc_mode  = false;
              webSocket.sendTXT(num, "Encoder direction of rotation will be considered");
            };
            if (receive_outputcontrol_data){
              int l1r, l1g, l1b, l2r, l2g, l2b, l3r, l3g, l3b, l4r, l4g, l4b, l5r, l5g, l5b, l6r, l6g, l6b, l7r, l7g, l7b, l8r, l8g ,l8b, ld, w1f, w2f, pf, pd, ph1f, ph1dif, ph1min, ph2f, ph2dif, ph2min, vol, w1t, w2t;
              if (sscanf((const char*)payload, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &l1r, &l1g, &l1b, &l2r, &l2g, &l2b, &l3r, &l3g, &l3b, &l4r, &l4g, &l4b, &l5r, &l5g, &l5b, &l6r, &l6g, &l6b, &l7r, &l7g, &l7b, &l8r, &l8g ,&l8b, &ld,  &w1f, &w2f, &pf, &pd, &ph1f, &ph1dif, &ph1min, &ph2f, &ph2dif, &ph2min, &vol, &w1t, &w2t) ==  38) { 
                //setting up the led colors
                
                color1 = strip.Color(l1r, l1g, l1b); 
                color2 = strip.Color(l2r, l2g, l2b); 
                color3 = strip.Color(l3r, l3g, l3b); 
                color4 = strip.Color(l4r, l4g, l4b); 
                color5 = strip.Color(l5r, l5g, l5b); 
                color6 = strip.Color(l6r, l6g, l6b); 
                color7 = strip.Color(l7r, l7g, l7b); 
                color8 = strip.Color(l8r, l8g ,l8b); 
                leddelay = ld;
                run_led_cycle = true;
                use_leddelay = true;
                if (leddelay>10000){ //threshold in ms to turn blinking off
                use_leddelay=false;
                }
                wave_1_freq=w1f;
                wave_2_freq=w2f;
                panner_freq=pf;
                panner_div=pd;
                phasor1_freq=ph1f;
                phasor1_max=ph1min + ph1dif;
                phasor1_min=ph1min;
                phasor2_freq=ph2f;
                phasor2_max=ph2min + ph2dif;
                phasor2_min=ph2min;
                wave_1_type=w1t;
                wave_2_type=w2t;
                maxivolume=vol / 100.0;
                maximilian.setVolume(maxivolume);
              }
            }
            if (strcmp((const char*)payload, "start_data_transfer_from_ads") == 0) {
              data_transfer = true;
              set_data_transfer_buffer = false;
              set_delay = false;
              set_delay_and_data_transfer_buffer_step1 = false;
              set_delay_and_data_transfer_buffer_step2 = false;
              receive_outputcontrol_data = false;
            };
            if (strcmp((const char*)payload, "stop_data_transfer_from_ads") == 0) {
              data_transfer = false;
            };
            if (set_data_transfer_buffer) {
              String n_datapoints_txt=(const char*)payload;
              //Serial.println('Changed buffer size:');
              //Serial.println(n_datapoints_txt);
              n_datapoints = n_datapoints_txt.toInt();
              set_data_transfer_buffer = false;
            };
            if (strcmp((const char*)payload, "set_data_transfer_buffer_size") == 0) {
              set_data_transfer_buffer = true;
              data_transfer = false;
              set_delay = false;
              set_delay_and_data_transfer_buffer_step1 = false;
              set_delay_and_data_transfer_buffer_step2 = false;
              receive_outputcontrol_data = false;
            };
            if (strcmp((const char*)payload, "set_delay") == 0) {
              set_delay = true;
              set_data_transfer_buffer = false;
              data_transfer = false;
              set_delay_and_data_transfer_buffer_step1 = false;
              set_delay_and_data_transfer_buffer_step2 = false;
              receive_outputcontrol_data = false;
            };
            if (set_data_transfer_buffer) {
              String delay_txt=(const char*)payload;
              //Serial.println('Changed delay:');
              //Serial.println(delay_txt);
              delay_length = delay_txt.toInt();
              set_delay = false;
            };
            if (strcmp((const char*)payload, "set_delay_and_data_transfer_buffer_size") == 0) {
              set_delay = false;
              set_data_transfer_buffer = false;
              data_transfer = false;
              set_delay_and_data_transfer_buffer_step1 = true;
              set_delay_and_data_transfer_buffer_step2 = false;
              receive_outputcontrol_data = false;
            };
            if (set_delay_and_data_transfer_buffer_step1) {
              webSocket.sendTXT(num, "Awaiting delay and data transfer buffer size in shape with space separator");
              set_delay_and_data_transfer_buffer_step1 = false;
              set_delay_and_data_transfer_buffer_step2 = true;
            };
            if (set_delay_and_data_transfer_buffer_step2){
              int dl, n_dp;
              if (sscanf((const char*)payload, "%d,%d", &dl, &n_dp) ==  2) { // Check if both values were successfully parsed
                delay_length = dl;
                n_datapoints = n_dp;
                Serial.println(delay_length); // This will print  10
                Serial.println(n_datapoints);
                webSocket.sendTXT(num, "Delay and data transfer buffer size set up");
                set_delay_and_data_transfer_buffer_step2 = false;
              }
            };       
            if (strncmp((const char*)payload, "display_text:", 13) == 0) {
                // Assuming the text to display is sent after the command
                // You may need to adjust this based on how you send the text

                char* payload_str = (char*)payload;
                strtok(payload_str, ":"); // Skip the "display_text" part
                int text_size = atoi(strtok(NULL, ":")); // Get the text size
                String textToDisplay = String(strtok(NULL, ":"));

                turnOnDisplay();

                
                //Serial.println(textToDisplay);
                //Serial.println(text_size);
                display.clearDisplay();
                display.setTextSize(text_size);
                display.setTextColor(WHITE);
                display.setCursor(0, 0);
                display.println(textToDisplay);

                // Update the display
                display.display();
            };
            if (strcmp((const char*)payload, "turn_off_display") == 0) {
                // Assuming the text to display is sent after the command
                // You may need to adjust this based on how you send the text
                turnOffDisplay();
            };
            break;
        case WStype_BIN:
            Serial.printf("[%u] get binary length: %u\n", num, length);
            hexdump(payload, length);

            // send message to client
            // webSocket.sendBIN(num, payload, length);
            break;
		case WStype_ERROR:			
		case WStype_FRAGMENT_TEXT_START:
		case WStype_FRAGMENT_BIN_START:
		case WStype_FRAGMENT:
		case WStype_FRAGMENT_FIN:
			break;
    }

}
//WIFI functions - end
void maximilianCopyTask(void *pvParameters) {
    for (;;) {
        maximilian.copy();
    }
}


void run_ledloop(){
    strip.setPixelColor(0, color1);
    strip.setPixelColor(1, color2);
    strip.setPixelColor(2, color3);
    strip.setPixelColor(3, color4);
    strip.setPixelColor(4, color5);
    strip.setPixelColor(5, color6);
    strip.setPixelColor(6, color7);
    strip.setPixelColor(7, color8);
    strip.show();
    if (use_leddelay==true){
    delay(leddelay);
    strip.fill(strip.Color(0, 0, 0)); // If not working - check the JST connection - it should be resoldered.
    strip.show(); // Update the LEDs
    delay(leddelay); // Wait for 1 second  }
    }
}
void ledloopRunTask(void *pvParameters) {
    for (;;) {
        run_ledloop();
    }
}

void setup() {
  Wire.setPins(SDA, SCL);
  Wire.begin();
  Serial.begin(115200); 
  enc.setType(TYPE1); 
  config_audio();
  //some random default values
  wave_1_freq=220;
  wave_2_freq=220;
  panner_freq=1;
  panner_div=2;
  phasor1_freq=220;
  phasor1_max=0;
  phasor1_min=220;
  phasor2_freq=0;
  phasor2_max=220;
  phasor2_min=1;
  maxivolume=0.0;
  // ADS1256 init
  ads.init( CS_ADS1256, DRDY_ADS1256, RESET_ADS1256, 1700000 );
	//Serial.println( ads.speedSPI );
  // ADS1256 init - end
  // LED init
  strip.begin();
  strip.setBrightness(led_brightness); // Set brightness to 50%
  strip.show(); // Initialize all pixels to 'off'
    // Set colors for each LED
  color1 = strip.Color(255, 0, 0); // Red
  color2 = strip.Color(0, 255, 0); // Green
  color3 = strip.Color(0, 0, 255); // Blue
  color4 = strip.Color(255, 255, 255); // White
  color5 = strip.Color(0, 255, 255); // Cyan
  color6 = strip.Color(255, 0, 255); // Magenta
  color7 = strip.Color(255, 255, 0); // Yellow
  color8 = strip.Color(255, 0, 0); // Red

    // Light up each LED with a specific color
  strip.setPixelColor(0, color1);
  strip.setPixelColor(1, color2);
  strip.setPixelColor(2, color3);
  strip.setPixelColor(3, color4);
  strip.setPixelColor(4, color5);
  strip.setPixelColor(5, color6);
  strip.setPixelColor(6, color7);
  strip.setPixelColor(7, color8);

  strip.show(); // Update the LEDs
  delay(1000); // Wait for 1 second
  Serial.println("LED modules initialized...");
  // LED init - end
  // Display init
  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Welcome!...");
  display.display();
  delay(3000);
  turnOffDisplay();
  delay(1000);
  // Display init - end

  // RTC init - end
  strip.fill(strip.Color(0, 0, 0)); // If not working - check the JST connection - it should be resoldered.
  strip.show(); // Update the LEDs
  delay(1000); // Wait for 1 second  
  Serial.setDebugOutput(true);

  Serial.println();
  Serial.println();
  Serial.println();

  for(uint8_t t = 3; t > 0; t--) {
      Serial.printf("[SETUP] BOOT WAIT %d...\n", t);
      Serial.flush();
      delay(1000);
  }

  WiFiMulti.addAP("pop-os", "therizinosauria");

  while(WiFiMulti.run() != WL_CONNECTED) {
      delay(100);
  }
 
  Serial.println("Connected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
color1 = strip.Color(0, 0, 0); // Red
color2 = strip.Color(0, 0, 0); // Green
color3 = strip.Color(0, 0, 0); // Blue
color4 = strip.Color(0, 0, 0); // White
color5 = strip.Color(0, 0, 0); // Cyan
color6 = strip.Color(0, 0, 0); // Magenta
color7 = strip.Color(0, 0, 0); // Yellow
color8 = strip.Color(0, 0, 0); // Red
wave_1_type = 0;
wave_2_type = 0;

xTaskCreatePinnedToCore(
        maximilianCopyTask,   /* Function to implement the task */
        "MaximilianCopyTask", /* Name of the task */
        5000,                /* Stack size in words */
        NULL,                /* Task input parameter */
        1,                   /* Priority of the task */
        NULL, /* Task handle */
        1                    /* Core where the task should run */
    );
xTaskCreatePinnedToCore(
        ledloopRunTask,   /* Function to implement the task */
        "ledloopRunTask", /* Name of the task */
        5000,                /* Stack size in words */
        NULL,                /* Task input parameter */
        1,                   /* Priority of the task */
        NULL, /* Task handle */
        0                    /* Core where the task should run */
    );
}


const int MAX_SAMPLES = 1000; // maximum number of samples to store
int ch1_vals[MAX_SAMPLES]; // arrays to store the sampled values
int ch2_vals[MAX_SAMPLES];
int ch3_vals[MAX_SAMPLES];
int ch4_vals[MAX_SAMPLES];
int ch5_vals[MAX_SAMPLES];
int ch6_vals[MAX_SAMPLES];
int ch7_vals[MAX_SAMPLES];
int ch8_vals[MAX_SAMPLES];
int ch9_vals[MAX_SAMPLES]; //encoder
int sample_count = 0; // variable to keep track of the number of samples taken



//int enc_val_prev = 0;
int enc_val_n = 0;
//int enc_val_diff = 0;
int enc_cum_val_whilenodtransfer=0;
int enc_is_click = 0;
int enc_is_holded =  0;
void loop() {
  enc.tick();
  //Serial.println(444);
  if (enc.isRight()) enc_val_n++;     
  if (enc.isRightH()) enc_val_n += 5;  
  if (only_pos_enc_mode){
    if (enc.isLeft()) enc_val_n++; 
    if (enc.isLeftH()) enc_val_n += 5;  
  } else {
    if (enc.isLeft()) enc_val_n--;     
    if (enc.isLeftH()) enc_val_n -= 5;
  }
  if (enc.isClick()) enc_is_click = 1;
  if (enc.isHolded()) enc_is_holded = 1;
  
  enc_cum_val_whilenodtransfer+=enc_val_n;
  enc_val_n = 0;

  webSocket.loop();
  if (data_transfer && sample_count < n_datapoints) {
    enc.tick();
    if (enc.isRight()) enc_val_n++;     
    if (enc.isRightH()) enc_val_n += 5;  
    if (only_pos_enc_mode){
      if (enc.isLeft()) enc_val_n++; 
      if (enc.isLeftH()) enc_val_n += 5;  
    } else {
      if (enc.isLeft()) enc_val_n--;     
      if (enc.isLeftH()) enc_val_n -= 5;
    }

    if (enc.isClick()) enc_is_click = 1;
    if (enc.isHolded()) enc_is_holded = 1;
    // Serial.println(111);
    //Serial.println(enc_val_n);
    // Serial.println(222);
    // Serial.println(enc_val_prev);
    //enc_val_diff=enc_val_n-enc_val_prev;
    int ch1_val=ads.adcValues[ 0 ];
    int ch2_val=ads.adcValues[ 1 ];
    int ch3_val=ads.adcValues[ 2 ];
    int ch4_val=ads.adcValues[ 3 ];
    int ch5_val=ads.adcValues[ 4 ];
    int ch6_val=ads.adcValues[ 5 ];
    int ch7_val=ads.adcValues[ 6 ];
    int ch8_val=ads.adcValues[ 7 ];
    int enc_val=enc_val_n+enc_cum_val_whilenodtransfer;

    enc_cum_val_whilenodtransfer=0;
    enc_val_n=0;

    // store the sampled values in the arrays
    ch1_vals[sample_count] = ch1_val;
    ch2_vals[sample_count] = ch2_val;
    ch3_vals[sample_count] = ch3_val;
    ch4_vals[sample_count] = ch4_val;
    ch5_vals[sample_count] = ch5_val;
    ch6_vals[sample_count] = ch6_val;
    ch7_vals[sample_count] = ch7_val;
    ch8_vals[sample_count] = ch8_val;
    ch9_vals[sample_count] = enc_val;

    // increment the sample count
    sample_count++;
    
    //enc_val_prev=enc_val_n;
    //some delay
    delay(delay_length);

    // check if enough samples have been taken
    if (sample_count == n_datapoints) {
      // create a JSON message with the sampled values
      String jsonMessage = "{\"ch1\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch1_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch2\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch2_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch3\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch3_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch4\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch4_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch5\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch5_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch6\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch6_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch7\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch7_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch8\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch8_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"ch9\":[";
      for (int i = 0; i < n_datapoints; i++) {
        jsonMessage += ch9_vals[i];
        if (i < n_datapoints - 1) {
          jsonMessage += ",";
        }
      }
      jsonMessage += "],\"enc_is_clicked\":[";
      jsonMessage += enc_is_click;
      jsonMessage += "],\"enc_is_holded\":[";
      jsonMessage += enc_is_holded;
      jsonMessage += "]}";
      // send data to client
      webSocket.sendTXT(clientID, jsonMessage);

      // reset the sample count
      sample_count = 0;
      enc_is_click = 0;
      enc_is_holded =  0;
    }
  }
}

