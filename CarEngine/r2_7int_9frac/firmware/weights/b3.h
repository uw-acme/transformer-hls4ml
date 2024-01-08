//Numpy array shape [4]
//Min -0.250000000000
//Max 0.000000000000
//Number of zeros 3

#ifndef B3_H_
#define B3_H_

#ifndef __SYNTHESIS__
bias3_t b3[4];
#else
bias3_t b3[4] = {0.00, 0.00, -0.25, 0.00};
#endif

#endif
