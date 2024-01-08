//Numpy array shape [4]
//Min -0.750000000000
//Max 1.000000000000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[4];
#else
bias2_t b2[4] = {1.00, -0.50, -0.75, 0.25};
#endif

#endif
