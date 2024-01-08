//Numpy array shape [2, 4]
//Min -1.500000000000
//Max 0.500000000000
//Number of zeros 1

#ifndef W2_H_
#define W2_H_

#ifndef __SYNTHESIS__
weight2_t w2[8];
#else
weight2_t w2[8] = {0.00, 0.50, -1.00, 0.50, 0.25, -0.50, -1.50, -1.25};
#endif

#endif
