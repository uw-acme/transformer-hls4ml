//Numpy array shape [4]
//Min -0.250000000000
//Max 0.000000000000
//Number of zeros 3

#ifndef B17_H_
#define B17_H_

#ifndef __SYNTHESIS__
bias17_t b17[4];
#else
bias17_t b17[4] = {0.00, 0.00, -0.25, 0.00};
#endif

#endif
