#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 15
#define N_INPUT_2_1 6
#define seq_out_2 15
#define feature_out_2 6
#define N_LAYER_1_4 15
#define N_LAYER_2_4 8
#define N_LAYER_1_6 15
#define N_LAYER_2_6 6
#define seq_out_9 15
#define feature_out_9 6
#define N_LAYER_1_11 15
#define N_LAYER_2_11 8
#define N_LAYER_1_13 15
#define N_LAYER_2_13 6
#define seq_out_16 15
#define feature_out_16 6
#define N_LAYER_1_18 15
#define N_LAYER_2_18 8
#define N_LAYER_1_20 15
#define N_LAYER_2_20 6
#define N_SIZE_1_23 90
#define N_LAYER_24 32
#define N_LAYER_26 16
#define N_LAYER_28 8
#define N_LAYER_30 3

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<26,14> input_16accum_t;
typedef ap_fixed<17,7> input_t;
typedef ap_fixed<26,14> q_multi_head_attention_45accum_t;
typedef ap_fixed<26,14> q_multi_head_attention_45table_t;
typedef ap_fixed<17,7> q_multi_head_attention_45_default_t;
typedef ap_fixed<17,7> layer2_t;
typedef ap_fixed<26,14> add_90accum_t;
typedef ap_fixed<17,7> layer3_t;
typedef ap_fixed<26,14> q_dense_150accum_t;
typedef ap_fixed<26,14> q_dense_150table_t;
typedef ap_fixed<17,7> layer4_t;
typedef ap_fixed<11,7> weight4_t;
typedef ap_fixed<11,7> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<26,14> q_dense_150_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_150_quantized_relutable_t;
typedef ap_fixed<17,7> layer5_t;
typedef ap_fixed<26,14> q_dense_151accum_t;
typedef ap_fixed<26,14> q_dense_151table_t;
typedef ap_fixed<17,7> layer6_t;
typedef ap_fixed<11,7> weight6_t;
typedef ap_fixed<11,7> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<26,14> add_91accum_t;
typedef ap_fixed<17,7> layer8_t;
typedef ap_fixed<26,14> q_multi_head_attention_46accum_t;
typedef ap_fixed<26,14> q_multi_head_attention_46table_t;
typedef ap_fixed<17,7> q_multi_head_attention_46_default_t;
typedef ap_fixed<17,7> layer9_t;
typedef ap_fixed<26,14> add_92accum_t;
typedef ap_fixed<17,7> layer10_t;
typedef ap_fixed<26,14> q_dense_152accum_t;
typedef ap_fixed<26,14> q_dense_152table_t;
typedef ap_fixed<17,7> layer11_t;
typedef ap_fixed<11,7> weight11_t;
typedef ap_fixed<11,7> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_fixed<26,14> q_dense_152_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_152_quantized_relutable_t;
typedef ap_fixed<17,7> layer12_t;
typedef ap_fixed<26,14> q_dense_153accum_t;
typedef ap_fixed<26,14> q_dense_153table_t;
typedef ap_fixed<17,7> layer13_t;
typedef ap_fixed<11,7> weight13_t;
typedef ap_fixed<11,7> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<26,14> add_93accum_t;
typedef ap_fixed<17,7> layer15_t;
typedef ap_fixed<26,14> q_multi_head_attention_47accum_t;
typedef ap_fixed<26,14> q_multi_head_attention_47table_t;
typedef ap_fixed<17,7> q_multi_head_attention_47_default_t;
typedef ap_fixed<17,7> layer16_t;
typedef ap_fixed<26,14> add_94accum_t;
typedef ap_fixed<17,7> layer17_t;
typedef ap_fixed<26,14> q_dense_154accum_t;
typedef ap_fixed<26,14> q_dense_154table_t;
typedef ap_fixed<17,7> layer18_t;
typedef ap_fixed<11,7> weight18_t;
typedef ap_fixed<11,7> bias18_t;
typedef ap_uint<1> layer18_index;
typedef ap_fixed<26,14> q_dense_154_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_154_quantized_relutable_t;
typedef ap_fixed<17,7> layer19_t;
typedef ap_fixed<26,14> q_dense_155accum_t;
typedef ap_fixed<26,14> q_dense_155table_t;
typedef ap_fixed<17,7> layer20_t;
typedef ap_fixed<11,7> weight20_t;
typedef ap_fixed<11,7> bias20_t;
typedef ap_uint<1> layer20_index;
typedef ap_fixed<26,14> add_95accum_t;
typedef ap_fixed<17,7> layer22_t;
typedef ap_fixed<17,7> model_default_t;
typedef ap_fixed<26,14> q_dense_156accum_t;
typedef ap_fixed<26,14> q_dense_156table_t;
typedef ap_fixed<17,7> layer24_t;
typedef ap_fixed<11,7> weight24_t;
typedef ap_fixed<11,7> bias24_t;
typedef ap_uint<1> layer24_index;
typedef ap_fixed<26,14> q_dense_156_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_156_quantized_relutable_t;
typedef ap_fixed<17,7> layer25_t;
typedef ap_fixed<26,14> q_dense_157accum_t;
typedef ap_fixed<26,14> q_dense_157table_t;
typedef ap_fixed<17,7> layer26_t;
typedef ap_fixed<11,7> weight26_t;
typedef ap_fixed<11,7> bias26_t;
typedef ap_uint<1> layer26_index;
typedef ap_fixed<26,14> q_dense_157_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_157_quantized_relutable_t;
typedef ap_fixed<17,7> layer27_t;
typedef ap_fixed<26,14> q_dense_158accum_t;
typedef ap_fixed<26,14> q_dense_158table_t;
typedef ap_fixed<17,7> layer28_t;
typedef ap_fixed<11,7> weight28_t;
typedef ap_fixed<11,7> bias28_t;
typedef ap_uint<1> layer28_index;
typedef ap_fixed<26,14> q_dense_158_quantized_reluaccum_t;
typedef ap_fixed<26,14> q_dense_158_quantized_relutable_t;
typedef ap_fixed<17,7> layer29_t;
typedef ap_fixed<26,14> q_dense_159accum_t;
typedef ap_fixed<26,14> q_dense_159table_t;
typedef ap_fixed<17,7> layer30_t;
typedef ap_fixed<11,7> weight30_t;
typedef ap_fixed<11,7> bias30_t;
typedef ap_uint<1> layer30_index;
typedef ap_fixed<26,14> q_dense_159_quantized_softmaxaccum_t;
typedef ap_fixed<26,14> q_dense_159_quantized_softmaxtable_t;
typedef ap_fixed<17,7> result_t;

#endif
