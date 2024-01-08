#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 50
#define N_INPUT_2_1 1
#define seq_out_2 50
#define feature_out_2 1
#define N_LAYER_1_3 50
#define N_LAYER_2_3 4
#define N_LAYER_1_5 50
#define N_LAYER_2_5 1
#define seq_out_9 50
#define feature_out_9 1
#define N_LAYER_1_10 50
#define N_LAYER_2_10 4
#define N_LAYER_1_12 50
#define N_LAYER_2_12 1
#define seq_out_16 50
#define feature_out_16 1
#define N_LAYER_1_17 50
#define N_LAYER_2_17 4
#define N_LAYER_1_19 50
#define N_LAYER_2_19 1
#define N_SIZE_1_23 50
#define N_LAYER_24 32
#define N_LAYER_26 16
#define N_LAYER_28 2

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,7> input_6_default_t;
typedef ap_fixed<16,7> input_t;
typedef ap_fixed<22,10> q_multi_head_attention_15accum_t;
typedef ap_fixed<20,11> q_multi_head_attention_15table_t;
typedef ap_fixed<16,7> q_multi_head_attention_15_default_t;
typedef ap_fixed<16,7> layer2_t;
typedef ap_fixed<16,7> q_dense_45_default_t;
typedef ap_fixed<16,7> layer3_t;
typedef ap_fixed<9,7> weight3_t;
typedef ap_fixed<9,7> bias3_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<16,7> q_dense_45_quantized_relu_default_t;
typedef ap_fixed<16,7> layer4_t;
typedef ap_fixed<32,8> q_dense_45_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_46_default_t;
typedef ap_fixed<16,7> layer5_t;
typedef ap_fixed<9,7> weight5_t;
typedef ap_fixed<9,7> bias5_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<16,7> add_30_default_t;
typedef ap_fixed<16,7> layer7_t;
typedef ap_fixed<16,7> add_31_default_t;
typedef ap_fixed<16,7> layer8_t;
typedef ap_fixed<22,10> q_multi_head_attention_16accum_t;
typedef ap_fixed<20,11> q_multi_head_attention_16table_t;
typedef ap_fixed<16,7> q_multi_head_attention_16_default_t;
typedef ap_fixed<16,7> layer9_t;
typedef ap_fixed<16,7> q_dense_47_default_t;
typedef ap_fixed<16,7> layer10_t;
typedef ap_fixed<9,7> weight10_t;
typedef ap_fixed<9,7> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,7> q_dense_47_quantized_relu_default_t;
typedef ap_fixed<16,7> layer11_t;
typedef ap_fixed<32,8> q_dense_47_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_48_default_t;
typedef ap_fixed<16,7> layer12_t;
typedef ap_fixed<9,7> weight12_t;
typedef ap_fixed<9,7> bias12_t;
typedef ap_uint<1> layer12_index;
typedef ap_fixed<16,7> add_32_default_t;
typedef ap_fixed<16,7> layer14_t;
typedef ap_fixed<16,7> add_33_default_t;
typedef ap_fixed<16,7> layer15_t;
typedef ap_fixed<22,10> q_multi_head_attention_17accum_t;
typedef ap_fixed<20,11> q_multi_head_attention_17table_t;
typedef ap_fixed<16,7> q_multi_head_attention_17_default_t;
typedef ap_fixed<16,7> layer16_t;
typedef ap_fixed<16,7> q_dense_49_default_t;
typedef ap_fixed<16,7> layer17_t;
typedef ap_fixed<9,7> weight17_t;
typedef ap_fixed<9,7> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<16,7> q_dense_49_quantized_relu_default_t;
typedef ap_fixed<16,7> layer18_t;
typedef ap_fixed<32,8> q_dense_49_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_50_default_t;
typedef ap_fixed<16,7> layer19_t;
typedef ap_fixed<9,7> weight19_t;
typedef ap_fixed<9,7> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_fixed<16,7> add_34_default_t;
typedef ap_fixed<16,7> layer21_t;
typedef ap_fixed<16,7> add_35_default_t;
typedef ap_fixed<16,7> layer22_t;
typedef ap_fixed<16,7> model_default_t;
typedef ap_fixed<16,7> q_dense_51_default_t;
typedef ap_fixed<16,7> layer24_t;
typedef ap_fixed<9,7> weight24_t;
typedef ap_fixed<9,7> bias24_t;
typedef ap_uint<1> layer24_index;
typedef ap_fixed<16,7> q_dense_51_quantized_relu_default_t;
typedef ap_fixed<16,7> layer25_t;
typedef ap_fixed<32,8> q_dense_51_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_52_default_t;
typedef ap_fixed<16,7> layer26_t;
typedef ap_fixed<9,7> weight26_t;
typedef ap_fixed<9,7> bias26_t;
typedef ap_uint<1> layer26_index;
typedef ap_fixed<16,7> q_dense_52_quantized_relu_default_t;
typedef ap_fixed<16,7> layer27_t;
typedef ap_fixed<32,8> q_dense_52_quantized_relu_table_t;
typedef ap_fixed<16,7> q_dense_53_default_t;
typedef ap_fixed<16,7> layer28_t;
typedef ap_fixed<9,7> weight28_t;
typedef ap_fixed<9,7> bias28_t;
typedef ap_uint<1> layer28_index;
typedef ap_fixed<16,7> q_dense_53_quantized_softmax_default_t;
typedef ap_fixed<16,7> result_t;
typedef ap_fixed<32,8> q_dense_53_quantized_softmax_table_t;

#endif
