//Numpy array shape [8]
//Min -0.125000000000
//Max 0.375000000000
//Number of zeros 1

#ifndef B28_H_
#define B28_H_

#ifndef __SYNTHESIS__
bias28_t b28[8];
#else
bias28_t b28[8] = {0.3750, 0.2500, 0.0625, 0.1875, 0.2500, 0.3125, -0.1250, 0.0000};
#endif

#endif
