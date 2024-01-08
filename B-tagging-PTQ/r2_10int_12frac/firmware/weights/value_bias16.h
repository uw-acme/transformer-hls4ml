//Numpy array shape [2, 32]
//Min -0.103085711598
//Max 0.116700582206
//Number of zeros 0

#ifndef VALUE_BIAS16_H_
#define VALUE_BIAS16_H_

#ifndef __SYNTHESIS__
multi_head_attention_4_default_t value_bias16[64];
#else
multi_head_attention_4_default_t value_bias16[64] = {-0.014621675014, -0.038671098650, -0.035837825388, -0.007020583842, -0.075805529952, -0.006258431822, 0.029008645564, 0.022649679333, 0.018928626552, -0.003723106813, 0.020174497738, -0.064632445574, 0.022085770965, 0.006202498917, -0.035345647484, 0.012323886156, 0.004390594084, -0.045142348856, -0.030745035037, 0.042343430221, 0.014461318962, -0.053680874407, -0.045609749854, -0.023291042075, -0.014660818502, 0.026717958972, -0.014629715122, -0.036834571511, -0.014998143539, -0.016028756276, 0.024970198050, 0.025524558499, -0.001813435345, -0.039600081742, 0.025408631191, 0.023853017017, 0.019041756168, 0.037357747555, -0.031539473683, 0.034425653517, 0.116700582206, 0.002923989668, 0.008760214783, 0.063198015094, 0.056828547269, -0.052624456584, -0.027210438624, -0.000464360870, 0.096776247025, 0.034626606852, 0.003793915734, -0.016118507832, -0.006116684061, -0.002715825336, 0.035671990365, -0.103085711598, -0.048824731261, 0.049029622227, -0.050795715302, 0.003344255500, 0.047555722296, 0.004355589394, 0.018880678341, -0.025305518880};
#endif

#endif