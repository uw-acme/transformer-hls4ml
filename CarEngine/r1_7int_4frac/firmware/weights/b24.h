//Numpy array shape [32]
//Min -0.250000000000
//Max 0.000000000000
//Number of zeros 19

#ifndef B24_H_
#define B24_H_

#ifndef __SYNTHESIS__
bias24_t b24[32];
#else
bias24_t b24[32] = {-0.25, 0.00, 0.00, 0.00, 0.00, 0.00, -0.25, -0.25, -0.25, 0.00, 0.00, -0.25, -0.25, 0.00, -0.25, -0.25, 0.00, 0.00, 0.00, 0.00, -0.25, 0.00, 0.00, -0.25, 0.00, 0.00, -0.25, -0.25, -0.25, 0.00, 0.00, 0.00};
#endif

#endif
