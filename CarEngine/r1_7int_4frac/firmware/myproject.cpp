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
    input_t input_6[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer29_out[N_LAYER_28]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_6 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_6,layer29_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(attention_output_weight2, "attention_output_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 1>(attention_output_bias2, "attention_output_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(key_weight2, "key_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(key_bias2, "key_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(query_weight2, "query_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(query_bias2, "query_bias2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(value_weight2, "value_weight2.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_15_default_t, 48>(value_bias2, "value_bias2.txt");
        nnet::load_weights_from_txt<weight3_t, 4>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 4>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight5_t, 4>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 1>(b5, "b5.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(attention_output_weight9, "attention_output_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 1>(attention_output_bias9, "attention_output_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(key_weight9, "key_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(key_bias9, "key_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(query_weight9, "query_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(query_bias9, "query_bias9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(value_weight9, "value_weight9.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_16_default_t, 48>(value_bias9, "value_bias9.txt");
        nnet::load_weights_from_txt<weight10_t, 4>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 4>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight12_t, 4>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 1>(b12, "b12.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(attention_output_weight16, "attention_output_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 1>(attention_output_bias16, "attention_output_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(key_weight16, "key_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(key_bias16, "key_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(query_weight16, "query_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(query_bias16, "query_bias16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(value_weight16, "value_weight16.txt");
        nnet::load_weights_from_txt<q_multi_head_attention_17_default_t, 48>(value_bias16, "value_bias16.txt");
        nnet::load_weights_from_txt<weight17_t, 4>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 4>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight19_t, 4>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 1>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight24_t, 1600>(w24, "w24.txt");
        nnet::load_weights_from_txt<bias24_t, 32>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight26_t, 512>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 16>(b26, "b26.txt");
        nnet::load_weights_from_txt<weight28_t, 32>(w28, "w28.txt");
        nnet::load_weights_from_txt<bias28_t, 2>(b28, "b28.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[seq_out_2*feature_out_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::multiheadattention<input_t, layer2_t, config2>(input_6, input_6, layer2_out, attention_output_weight2, attention_output_bias2, key_weight2, key_bias2, query_weight2, query_bias2, value_weight2, value_bias2); // q_multi_head_attention_15

    layer3_t layer3_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // q_dense_45

    layer4_t layer4_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out); // q_dense_45_quantized_relu

    layer5_t layer5_out[N_LAYER_1_5*N_LAYER_2_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // q_dense_46

    layer7_t layer7_out[seq_out_2*feature_out_2];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::add<layer2_t, input_t, layer7_t, config7>(layer2_out, input_6, layer7_out); // add_30

    layer8_t layer8_out[N_LAYER_1_5*N_LAYER_2_5];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::add<layer5_t, layer7_t, layer8_t, config8>(layer5_out, layer7_out, layer8_out); // add_31

    layer9_t layer9_out[seq_out_9*feature_out_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::multiheadattention<layer8_t, layer9_t, config9>(layer8_out, layer8_out, layer9_out, attention_output_weight9, attention_output_bias9, key_weight9, key_bias9, query_weight9, query_bias9, value_weight9, value_bias9); // q_multi_head_attention_16

    layer10_t layer10_out[N_LAYER_1_10*N_LAYER_2_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense_47

    layer11_t layer11_out[N_LAYER_1_10*N_LAYER_2_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // q_dense_47_quantized_relu

    layer12_t layer12_out[N_LAYER_1_12*N_LAYER_2_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // q_dense_48

    layer14_t layer14_out[seq_out_9*feature_out_9];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::add<layer9_t, layer8_t, layer14_t, config14>(layer9_out, layer8_out, layer14_out); // add_32

    layer15_t layer15_out[N_LAYER_1_12*N_LAYER_2_12];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::add<layer12_t, layer14_t, layer15_t, config15>(layer12_out, layer14_out, layer15_out); // add_33

    layer16_t layer16_out[seq_out_16*feature_out_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::multiheadattention<layer15_t, layer16_t, config16>(layer15_out, layer15_out, layer16_out, attention_output_weight16, attention_output_bias16, key_weight16, key_bias16, query_weight16, query_bias16, value_weight16, value_bias16); // q_multi_head_attention_17

    layer17_t layer17_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // q_dense_49

    layer18_t layer18_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer17_t, layer18_t, relu_config18>(layer17_out, layer18_out); // q_dense_49_quantized_relu

    layer19_t layer19_out[N_LAYER_1_19*N_LAYER_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer18_t, layer19_t, config19>(layer18_out, layer19_out, w19, b19); // q_dense_50

    layer21_t layer21_out[seq_out_16*feature_out_16];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::add<layer16_t, layer15_t, layer21_t, config21>(layer16_out, layer15_out, layer21_out); // add_34

    layer22_t layer22_out[N_LAYER_1_19*N_LAYER_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::add<layer19_t, layer21_t, layer22_t, config22>(layer19_out, layer21_out, layer22_out); // add_35

    layer24_t layer24_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::dense<layer22_t, layer24_t, config24>(layer22_out, layer24_out, w24, b24); // q_dense_51

    layer25_t layer25_out[N_LAYER_24];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::relu<layer24_t, layer25_t, relu_config25>(layer24_out, layer25_out); // q_dense_51_quantized_relu

    layer26_t layer26_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::dense<layer25_t, layer26_t, config26>(layer25_out, layer26_out, w26, b26); // q_dense_52

    layer27_t layer27_out[N_LAYER_26];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::relu<layer26_t, layer27_t, relu_config27>(layer26_out, layer27_out); // q_dense_52_quantized_relu

    layer28_t layer28_out[N_LAYER_28];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0
    nnet::dense<layer27_t, layer28_t, config28>(layer27_out, layer28_out, w28, b28); // q_dense_53

    nnet::softmax<layer28_t, result_t, softmax_config29>(layer28_out, layer29_out); // q_dense_53_quantized_softmax

}
