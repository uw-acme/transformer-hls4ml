//Numpy array shape [3, 16]
//Min -0.136648178101
//Max 0.136648178101
//Number of zeros 6

#ifndef QUERY_BIAS9_H_
#define QUERY_BIAS9_H_

#ifndef __SYNTHESIS__
q_multi_head_attention_16_default_t query_bias9[48];
#else
q_multi_head_attention_16_default_t query_bias9[48] = {0.10, 0.00, -0.10, 0.12, -0.10, 0.10, 0.03, -0.12, 0.10, -0.10, 0.10, 0.00, 0.10, -0.08, -0.01, 0.00, 0.14, -0.14, -0.06, 0.14, -0.14, -0.14, -0.14, -0.14, -0.14, 0.14, 0.14, 0.14, 0.00, 0.14, 0.14, -0.07, 0.01, 0.09, -0.09, -0.09, -0.09, 0.09, 0.09, 0.09, 0.02, 0.09, 0.00, 0.09, -0.09, 0.00, 0.09, -0.09};
#endif

#endif
