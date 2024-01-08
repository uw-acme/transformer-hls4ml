//Numpy array shape [8]
//Min -0.437500000000
//Max 0.062500000000
//Number of zeros 1

#ifndef B4_H_
#define B4_H_

#ifndef __SYNTHESIS__
bias4_t b4[8];
#else
bias4_t b4[8] = {-0.1250, -0.0625, -0.4375, -0.1250, -0.4375, 0.0000, -0.3750, 0.0625};
#endif

#endif
