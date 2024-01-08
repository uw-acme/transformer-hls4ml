#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 100
#define N_INPUT_2_1 2
#define N_LAYER_1_2 100
#define N_LAYER_2_2 4
#define seq_out_5 100
#define feature_out_5 4
#define N_LAYER_1_8 100
#define N_LAYER_2_8 4
#define N_LAYER_1_10 100
#define N_LAYER_2_10 4
#define seq_out_14 100
#define feature_out_14 4
#define N_LAYER_1_17 100
#define N_LAYER_2_17 4
#define N_LAYER_1_19 100
#define N_LAYER_2_19 4
#define N_LAYER_1_23 100
#define N_LAYER_2_23 1
#define N_SIZE_1_25 100
#define N_LAYER_26 8
#define N_LAYER_28 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,7> input_1_default_t;
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<16,7> q_dense_default_t;
typedef ap_fixed<16,7> layer2_t;
typedef ap_fixed<6,4> weight2_t;
typedef ap_fixed<6,4> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,7> q_layer_normalization_default_t;
typedef ap_fixed<20,11> q_layer_normalizationtable_t;
typedef ap_fixed<16,7> layer4_t;
typedef ap_fixed<20,11> q_multi_head_attentionaccum_t;
typedef ap_fixed<20,11> q_multi_head_attentiontable_t;
typedef ap_fixed<16,7> q_multi_head_attention_default_t;
typedef ap_fixed<16,7> layer5_t;
typedef ap_fixed<16,7> add_default_t;
typedef ap_fixed<16,7> layer6_t;
typedef ap_fixed<16,7> q_layer_normalization_1_default_t;
typedef ap_fixed<20,11> q_layer_normalization_1table_t;
typedef ap_fixed<16,7> layer7_t;
typedef ap_fixed<16,7> q_dense_1_default_t;
typedef ap_fixed<16,7> layer8_t;
typedef ap_fixed<6,4> weight8_t;
typedef ap_fixed<6,4> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,7> q_dense_1_quantized_relu_default_t;
typedef ap_fixed<16,7> layer9_t;
typedef ap_fixed<32,8> q_dense_1_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_2_default_t;
typedef ap_fixed<16,7> layer10_t;
typedef ap_fixed<6,4> weight10_t;
typedef ap_fixed<6,4> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,7> add_1_default_t;
typedef ap_fixed<16,7> layer12_t;
typedef ap_fixed<16,7> q_layer_normalization_2_default_t;
typedef ap_fixed<20,11> q_layer_normalization_2table_t;
typedef ap_fixed<16,7> layer13_t;
typedef ap_fixed<20,11> q_multi_head_attention_1accum_t;
typedef ap_fixed<20,11> q_multi_head_attention_1table_t;
typedef ap_fixed<16,7> q_multi_head_attention_1_default_t;
typedef ap_fixed<16,7> layer14_t;
typedef ap_fixed<16,7> add_2_default_t;
typedef ap_fixed<16,7> layer15_t;
typedef ap_fixed<16,7> q_layer_normalization_3_default_t;
typedef ap_fixed<20,11> q_layer_normalization_3table_t;
typedef ap_fixed<16,7> layer16_t;
typedef ap_fixed<16,7> q_dense_3_default_t;
typedef ap_fixed<16,7> layer17_t;
typedef ap_fixed<6,4> weight17_t;
typedef ap_fixed<6,4> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<16,7> q_dense_3_quantized_relu_default_t;
typedef ap_fixed<16,7> layer18_t;
typedef ap_fixed<32,8> q_dense_3_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_4_default_t;
typedef ap_fixed<16,7> layer19_t;
typedef ap_fixed<6,4> weight19_t;
typedef ap_fixed<6,4> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_fixed<16,7> add_3_default_t;
typedef ap_fixed<16,7> layer21_t;
typedef ap_fixed<16,7> q_layer_normalization_4_default_t;
typedef ap_fixed<20,11> q_layer_normalization_4table_t;
typedef ap_fixed<16,7> layer22_t;
typedef ap_fixed<16,7> q_dense_5_default_t;
typedef ap_fixed<16,7> layer23_t;
typedef ap_fixed<6,4> weight23_t;
typedef ap_fixed<6,4> bias23_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<16,7> q_dense_5_quantized_relu_default_t;
typedef ap_fixed<16,7> layer24_t;
typedef ap_fixed<32,8> q_dense_5_quantized_relu_table_t;
typedef ap_fixed<16,7> model_default_t;
typedef ap_fixed<16,7> q_dense_6_default_t;
typedef ap_fixed<16,7> layer26_t;
typedef ap_fixed<6,4> weight26_t;
typedef ap_fixed<6,4> bias26_t;
typedef ap_uint<1> layer26_index;
typedef ap_fixed<16,7> q_dense_6_quantized_relu_default_t;
typedef ap_fixed<16,7> layer27_t;
typedef ap_fixed<32,8> q_dense_6_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_7_default_t;
typedef ap_fixed<16,7> layer28_t;
typedef ap_fixed<6,4> weight28_t;
typedef ap_fixed<6,4> bias28_t;
typedef ap_uint<1> layer28_index;
typedef ap_fixed<16,7> activation_default_t;
typedef ap_fixed<18,8> activationtable_t;
typedef ap_fixed<16,7> result_t;

#endif
