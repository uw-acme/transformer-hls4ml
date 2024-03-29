//Numpy array shape [3, 16]
//Min -0.136648178101
//Max 0.136648178101
//Number of zeros 6

#ifndef QUERY_BIAS9_H_
#define QUERY_BIAS9_H_

#ifndef __SYNTHESIS__
q_multi_head_attention_16_default_t query_bias9[48];
#else
q_multi_head_attention_16_default_t query_bias9[48] = {0.102, 0.000, -0.102, 0.122, -0.102, 0.102, 0.031, -0.118, 0.102, -0.102, 0.102, 0.000, 0.102, -0.081, -0.014, 0.000, 0.137, -0.137, -0.062, 0.137, -0.137, -0.137, -0.137, -0.137, -0.137, 0.137, 0.137, 0.137, 0.000, 0.137, 0.137, -0.074, 0.008, 0.088, -0.088, -0.088, -0.088, 0.088, 0.088, 0.088, 0.020, 0.088, 0.000, 0.088, -0.088, 0.000, 0.088, -0.088};
#endif

#endif
