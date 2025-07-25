/**
 * @brief We just set up the codec. Because I2C was not defined we need to
 * initilize it ourself After this you can set up and use i2s
 * @author phil schatzmann
 */

#include "AudioBoard.h"

AudioBoard board(AudioDriverES8388);

void setup() {
  // Setup logging
  Serial.begin(115200);
  LOGLEVEL_AUDIODRIVER = AudioDriverInfo;

  // start I2C for the communication with the codec
  Wire.begin();
  // configure codec
  CodecConfig cfg;
  cfg.input_device = input_device_LINE1;
  cfg.output_device = output_device_ALL;
  cfg.i2s.bits = BIT_LENGTH_16BITS;
  cfg.i2s.rate = RATE_44K;
  // cfg.i2s.fmt = I2S_NORMAL;
  // cfg.i2s.mode = MODE_SLAVE;
  board.begin(cfg);
}

void loop() {}
