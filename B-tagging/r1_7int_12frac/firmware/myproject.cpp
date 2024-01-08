//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_16[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer31_out[N_LAYER_30]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_16 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_16,layer31_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 384>(attention_output_weight2, "attention_output_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 6>(attention_output_bias2, "attention_output_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 384>(key_weight2, "key_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 64>(key_bias2, "key_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 384>(query_weight2, "query_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 64>(query_bias2, "query_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 384>(value_weight2, "value_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_45_default_t, 64>(value_bias2, "value_bias2.txt");
        nnet::load_weights_from_txt<weight4_t, 48>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 8>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 48>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 6>(b6, "b6.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 384>(attention_output_weight9, "attention_output_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 6>(attention_output_bias9, "attention_output_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 384>(key_weight9, "key_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 64>(key_bias9, "key_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 384>(query_weight9, "query_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 64>(query_bias9, "query_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 384>(value_weight9, "value_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_46_default_t, 64>(value_bias9, "value_bias9.txt");
        nnet::load_weights_from_txt<weight11_t, 48>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 8>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight13_t, 48>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 6>(b13, "b13.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 384>(attention_output_weight16, "attention_output_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 6>(attention_output_bias16, "attention_output_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 384>(key_weight16, "key_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 64>(key_bias16, "key_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 384>(query_weight16, "query_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 64>(query_bias16, "query_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 384>(value_weight16, "value_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_47_default_t, 64>(value_bias16, "value_bias16.txt");
        nnet::load_weights_from_txt<weight18_t, 48>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 8>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight20_t, 48>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 6>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight24_t, 2880>(w24, "w24.txt");
        nnet::load_weights_from_txt<bias24_t, 32>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight26_t, 512>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 16>(b26, "b26.txt");
        nnet::load_weights_from_txt<weight28_t, 128>(w28, "w28.txt");
        nnet::load_weights_from_txt<bias28_t, 8>(b28, "b28.txt");
        nnet::load_weights_from_txt<weight30_t, 24>(w30, "w30.txt");
        nnet::load_weights_from_txt<bias30_t, 3>(b30, "b30.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[seq_out_2*feature_out_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::multiheadattention<input_t, layer2_t, config2>(input_16, input_16, layer2_out, attention_output_weight2, attention_output_bias2, key_weight2, key_bias2, query_weight2, query_bias2, value_weight2, value_bias2); // q_multi_head_attention_45

    layer3_t layer3_out[seq_out_2*feature_out_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::add<layer2_t, input_t, layer3_t, config3>(layer2_out, input_16, layer3_out); // add_90

    layer4_t layer4_out[N_LAYER_1_4*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // q_dense_150

    layer5_t layer5_out[N_LAYER_1_4*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_dense_150_quantized_relu

    layer6_t layer6_out[N_LAYER_1_6*N_LAYER_2_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // q_dense_151

    layer8_t layer8_out[N_LAYER_1_6*N_LAYER_2_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::add<layer6_t, layer3_t, layer8_t, config8>(layer6_out, layer3_out, layer8_out); // add_91

    layer9_t layer9_out[seq_out_9*feature_out_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::multiheadattention<layer8_t, layer9_t, config9>(layer8_out, layer8_out, layer9_out, attention_output_weight9, attention_output_bias9, key_weight9, key_bias9, query_weight9, query_bias9, value_weight9, value_bias9); // q_multi_head_attention_46

    layer10_t layer10_out[seq_out_9*feature_out_9];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::add<layer9_t, layer8_t, layer10_t, config10>(layer9_out, layer8_out, layer10_out); // add_92

    layer11_t layer11_out[N_LAYER_1_11*N_LAYER_2_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // q_dense_152

    layer12_t layer12_out[N_LAYER_1_11*N_LAYER_2_11];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::relu<layer11_t, layer12_t, relu_config12>(layer11_out, layer12_out); // q_dense_152_quantized_relu

    layer13_t layer13_out[N_LAYER_1_13*N_LAYER_2_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer12_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // q_dense_153

    layer15_t layer15_out[N_LAYER_1_13*N_LAYER_2_13];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::add<layer13_t, layer10_t, layer15_t, config15>(layer13_out, layer10_out, layer15_out); // add_93

    layer16_t layer16_out[seq_out_16*feature_out_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::multiheadattention<layer15_t, layer16_t, config16>(layer15_out, layer15_out, layer16_out, attention_output_weight16, attention_output_bias16, key_weight16, key_bias16, query_weight16, query_bias16, value_weight16, value_bias16); // q_multi_head_attention_47

    layer17_t layer17_out[seq_out_16*feature_out_16];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::add<layer16_t, layer15_t, layer17_t, config17>(layer16_out, layer15_out, layer17_out); // add_94

    layer18_t layer18_out[N_LAYER_1_18*N_LAYER_2_18];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::dense<layer17_t, layer18_t, config18>(layer17_out, layer18_out, w18, b18); // q_dense_154

    layer19_t layer19_out[N_LAYER_1_18*N_LAYER_2_18];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<layer18_t, layer19_t, relu_config19>(layer18_out, layer19_out); // q_dense_154_quantized_relu

    layer20_t layer20_out[N_LAYER_1_20*N_LAYER_2_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, layer20_t, config20>(layer19_out, layer20_out, w20, b20); // q_dense_155

    layer22_t layer22_out[N_LAYER_1_20*N_LAYER_2_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::add<layer20_t, layer17_t, layer22_t, config22>(layer20_out, layer17_out, layer22_out); // add_95

    layer24_t layer24_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::dense<layer22_t, layer24_t, config24>(layer22_out, layer24_out, w24, b24); // q_dense_156

    layer25_t layer25_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::relu<layer24_t, layer25_t, relu_config25>(layer24_out, layer25_out); // q_dense_156_quantized_relu

    layer26_t layer26_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::dense<layer25_t, layer26_t, config26>(layer25_out, layer26_out, w26, b26); // q_dense_157

    layer27_t layer27_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::relu<layer26_t, layer27_t, relu_config27>(layer26_out, layer27_out); // q_dense_157_quantized_relu

    layer28_t layer28_out[N_LAYER_28];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0
    nnet::dense<layer27_t, layer28_t, config28>(layer27_out, layer28_out, w28, b28); // q_dense_158

    layer29_t layer29_out[N_LAYER_28];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    nnet::relu<layer28_t, layer29_t, relu_config29>(layer28_out, layer29_out); // q_dense_158_quantized_relu

    layer30_t layer30_out[N_LAYER_30];
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0
    nnet::dense<layer29_t, layer30_t, config30>(layer29_out, layer30_out, w30, b30); // q_dense_159

    nnet::softmax<layer30_t, result_t, softmax_config31>(layer30_out, layer31_out); // q_dense_159_quantized_softmax

}
