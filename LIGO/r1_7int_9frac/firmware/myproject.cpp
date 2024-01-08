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
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer30_out[N_LAYER_28]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer30_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 8>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<q_layer_normalization_default_t, 4>(s4, "s4.txt");
        nnet::load_weights_from_txt<q_layer_normalization_default_t, 4>(b4, "b4.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 256>(attention_output_weight5, "attention_output_weight5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 4>(attention_output_bias5, "attention_output_bias5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 256>(key_weight5, "key_weight5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 64>(key_bias5, "key_bias5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 256>(query_weight5, "query_weight5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 64>(query_bias5, "query_bias5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 256>(value_weight5, "value_weight5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_default_t, 64>(value_bias5, "value_bias5.txt");
        nnet::load_weights_from_txt<q_layer_normalization_1_default_t, 4>(s7, "s7.txt");
        nnet::load_weights_from_txt<q_layer_normalization_1_default_t, 4>(b7, "b7.txt");
        nnet::load_weights_from_txt<weight8_t, 16>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 4>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight10_t, 16>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 4>(b10, "b10.txt");
        nnet::load_weights_from_txt<q_layer_normalization_2_default_t, 4>(s13, "s13.txt");
        nnet::load_weights_from_txt<q_layer_normalization_2_default_t, 4>(b13, "b13.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 256>(attention_output_weight14, "attention_output_weight14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 4>(attention_output_bias14, "attention_output_bias14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 256>(key_weight14, "key_weight14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 64>(key_bias14, "key_bias14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 256>(query_weight14, "query_weight14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 64>(query_bias14, "query_bias14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 256>(value_weight14, "value_weight14.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_1_default_t, 64>(value_bias14, "value_bias14.txt");
        nnet::load_weights_from_txt<q_layer_normalization_3_default_t, 4>(s16, "s16.txt");
        nnet::load_weights_from_txt<q_layer_normalization_3_default_t, 4>(b16, "b16.txt");
        nnet::load_weights_from_txt<weight17_t, 16>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 4>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight19_t, 16>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 4>(b19, "b19.txt");
        nnet::load_weights_from_txt<q_layer_normalization_4_default_t, 4>(s22, "s22.txt");
        nnet::load_weights_from_txt<q_layer_normalization_4_default_t, 4>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight23_t, 4>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        nnet::load_weights_from_txt<weight26_t, 800>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 8>(b26, "b26.txt");
        nnet::load_weights_from_txt<weight28_t, 8>(w28, "w28.txt");
        nnet::load_weights_from_txt<bias28_t, 1>(b28, "b28.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_1_2*N_LAYER_2_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(input_1, layer2_out, w2, b2); // q_dense

    layer4_t layer4_out[N_LAYER_1_2*N_LAYER_2_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::layernormalize<layer2_t, layer4_t, config4>(layer2_out, layer4_out, s4, b4); // q_layer_normalization

    layer5_t layer5_out[seq_out_5*feature_out_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::multiheadattention<layer4_t, layer5_t, config5>(layer4_out, layer4_out, layer5_out, attention_output_weight5, attention_output_bias5, key_weight5, key_bias5, query_weight5, query_bias5, value_weight5, value_bias5); // q_multi_head_attention

    layer6_t layer6_out[seq_out_5*feature_out_5];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::add<layer5_t, layer4_t, layer6_t, config6>(layer5_out, layer4_out, layer6_out); // add

    layer7_t layer7_out[seq_out_5*feature_out_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::layernormalize<layer6_t, layer7_t, config7>(layer6_out, layer7_out, s7, b7); // q_layer_normalization_1

    layer8_t layer8_out[N_LAYER_1_8*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // q_dense_1

    layer9_t layer9_out[N_LAYER_1_8*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // q_dense_1_quantized_relu

    layer10_t layer10_out[N_LAYER_1_10*N_LAYER_2_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense_2

    layer12_t layer12_out[N_LAYER_1_10*N_LAYER_2_10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::add<layer10_t, layer7_t, layer12_t, config12>(layer10_out, layer7_out, layer12_out); // add_1

    layer13_t layer13_out[N_LAYER_1_10*N_LAYER_2_10];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::layernormalize<layer12_t, layer13_t, config13>(layer12_out, layer13_out, s13, b13); // q_layer_normalization_2

    layer14_t layer14_out[seq_out_14*feature_out_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::multiheadattention<layer13_t, layer14_t, config14>(layer13_out, layer13_out, layer14_out, attention_output_weight14, attention_output_bias14, key_weight14, key_bias14, query_weight14, query_bias14, value_weight14, value_bias14); // q_multi_head_attention_1

    layer15_t layer15_out[seq_out_14*feature_out_14];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::add<layer14_t, layer13_t, layer15_t, config15>(layer14_out, layer13_out, layer15_out); // add_2

    layer16_t layer16_out[seq_out_14*feature_out_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::layernormalize<layer15_t, layer16_t, config16>(layer15_out, layer16_out, s16, b16); // q_layer_normalization_3

    layer17_t layer17_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // q_dense_3

    layer18_t layer18_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer17_t, layer18_t, relu_config18>(layer17_out, layer18_out); // q_dense_3_quantized_relu

    layer19_t layer19_out[N_LAYER_1_19*N_LAYER_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer18_t, layer19_t, config19>(layer18_out, layer19_out, w19, b19); // q_dense_4

    layer21_t layer21_out[N_LAYER_1_19*N_LAYER_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::add<layer19_t, layer16_t, layer21_t, config21>(layer19_out, layer16_out, layer21_out); // add_3

    layer22_t layer22_out[N_LAYER_1_19*N_LAYER_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::layernormalize<layer21_t, layer22_t, config22>(layer21_out, layer22_out, s22, b22); // q_layer_normalization_4

    layer23_t layer23_out[N_LAYER_1_23*N_LAYER_2_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // q_dense_5

    layer24_t layer24_out[N_LAYER_1_23*N_LAYER_2_23];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::relu<layer23_t, layer24_t, relu_config24>(layer23_out, layer24_out); // q_dense_5_quantized_relu

    layer26_t layer26_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::dense<layer24_t, layer26_t, config26>(layer24_out, layer26_out, w26, b26); // q_dense_6

    layer27_t layer27_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::relu<layer26_t, layer27_t, relu_config27>(layer26_out, layer27_out); // q_dense_6_quantized_relu

    layer28_t layer28_out[N_LAYER_28];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0
    nnet::dense<layer27_t, layer28_t, config28>(layer27_out, layer28_out, w28, b28); // q_dense_7

    nnet::sigmoid<layer28_t, result_t, sigmoid_config30>(layer28_out, layer30_out); // activation

}
