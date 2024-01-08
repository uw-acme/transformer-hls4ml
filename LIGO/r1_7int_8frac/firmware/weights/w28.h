//Numpy array shape [8, 1]
//Min -0.500000000000
//Max 1.750000000000
//Number of zeros 0

#ifndef W28_H_
#define W28_H_

#ifndef __SYNTHESIS__
weight28_t w28[8];
#else
weight28_t w28[8] = {0.50, -0.50, 1.50, -0.25, -0.25, 1.25, -0.50, 1.75};
#endif

#endif
