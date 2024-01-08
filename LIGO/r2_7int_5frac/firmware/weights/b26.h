//Numpy array shape [8]
//Min -0.250000000000
//Max 1.250000000000
//Number of zeros 1

#ifndef B26_H_
#define B26_H_

#ifndef __SYNTHESIS__
bias26_t b26[8];
#else
bias26_t b26[8] = {0.25, -0.25, 1.25, 0.00, -0.25, 1.25, -0.25, 1.25};
#endif

#endif
