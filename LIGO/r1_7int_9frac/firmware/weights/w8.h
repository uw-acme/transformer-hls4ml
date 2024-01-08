//Numpy array shape [4, 4]
//Min -0.750000000000
//Max 1.500000000000
//Number of zeros 1

#ifndef W8_H_
#define W8_H_

#ifndef __SYNTHESIS__
weight8_t w8[16];
#else
weight8_t w8[16] = {0.50, -0.25, 0.75, 0.25, 0.25, 1.00, 0.00, 0.50, 0.25, 0.25, 0.50, -0.25, -0.75, -0.25, 1.50, -0.50};
#endif

#endif
