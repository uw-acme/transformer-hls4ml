#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_code_gen.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_multiheadattention.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/attention_output_weight2.h"
#include "weights/attention_output_bias2.h"
#include "weights/key_weight2.h"
#include "weights/key_bias2.h"
#include "weights/query_weight2.h"
#include "weights/query_bias2.h"
#include "weights/value_weight2.h"
#include "weights/value_bias2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/attention_output_weight9.h"
#include "weights/attention_output_bias9.h"
#include "weights/key_weight9.h"
#include "weights/key_bias9.h"
#include "weights/query_weight9.h"
#include "weights/query_bias9.h"
#include "weights/value_weight9.h"
#include "weights/value_bias9.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/attention_output_weight16.h"
#include "weights/attention_output_bias16.h"
#include "weights/key_weight16.h"
#include "weights/key_bias16.h"
#include "weights/query_weight16.h"
#include "weights/query_bias16.h"
#include "weights/value_weight16.h"
#include "weights/value_bias16.h"
#include "weights/w17.h"
#include "weights/b17.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w24.h"
#include "weights/b24.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w28.h"
#include "weights/b28.h"

//hls-fpga-machine-learning insert layer-config
// q_multi_head_attention_15
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_15accum_t accum_t;
    typedef q_multi_head_attention_15_default_t bias_t;
    typedef q_multi_head_attention_15_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_15accum_t accum_t;
    typedef q_multi_head_attention_15_default_t bias_t;
    typedef q_multi_head_attention_15_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config2 : nnet::activ_config {
    static const unsigned n_in = 50;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_15table_t exp_table_t;
    typedef q_multi_head_attention_15table_t inv_table_t;
    typedef q_multi_head_attention_15accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config2 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_15accum_t accum_t;
    typedef q_multi_head_attention_15_default_t bias_t;
    typedef q_multi_head_attention_15_default_t weight_t;
    typedef config2_1 config_mult1;
    typedef config2_2 config_mult2;
    typedef softmax_config2 softmax_config1;

    static const unsigned num_heads = 3;
    static const unsigned head_dim_key = 16;
    static const unsigned head_dim_value = 16;
    static const unsigned feature_dim = 1;
    static const unsigned seq_len = 50;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
};

// q_dense_45
struct config3 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef q_dense_45_default_t accum_t;
    typedef bias3_t bias_t;
    typedef weight3_t weight_t;
    typedef layer3_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_45_quantized_relu
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 200;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_45_quantized_relu_table_t table_t;
};

// q_dense_46
struct config5 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 3;
    static const bool store_weights_in_bram = false;
    typedef q_dense_46_default_t accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef layer5_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_30
struct config7 : nnet::merge_config {
    static const unsigned n_elem = seq_out_2*feature_out_2;
};

// add_31
struct config8 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_5*N_LAYER_2_5;
};

// q_multi_head_attention_16
struct config9_1 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_16accum_t accum_t;
    typedef q_multi_head_attention_16_default_t bias_t;
    typedef q_multi_head_attention_16_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9_2 : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_16accum_t accum_t;
    typedef q_multi_head_attention_16_default_t bias_t;
    typedef q_multi_head_attention_16_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config9 : nnet::activ_config {
    static const unsigned n_in = 50;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_16table_t exp_table_t;
    typedef q_multi_head_attention_16table_t inv_table_t;
    typedef q_multi_head_attention_16accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config9 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_16accum_t accum_t;
    typedef q_multi_head_attention_16_default_t bias_t;
    typedef q_multi_head_attention_16_default_t weight_t;
    typedef config9_1 config_mult1;
    typedef config9_2 config_mult2;
    typedef softmax_config9 softmax_config1;

    static const unsigned num_heads = 3;
    static const unsigned head_dim_key = 16;
    static const unsigned head_dim_value = 16;
    static const unsigned feature_dim = 1;
    static const unsigned seq_len = 50;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
};

// q_dense_47
struct config10 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef q_dense_47_default_t accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef layer10_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_47_quantized_relu
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = 200;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_47_quantized_relu_table_t table_t;
};

// q_dense_48
struct config12 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef q_dense_48_default_t accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef layer12_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_32
struct config14 : nnet::merge_config {
    static const unsigned n_elem = seq_out_9*feature_out_9;
};

// add_33
struct config15 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_12*N_LAYER_2_12;
};

// q_multi_head_attention_17
struct config16_1 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_17accum_t accum_t;
    typedef q_multi_head_attention_17_default_t bias_t;
    typedef q_multi_head_attention_17_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16_2 : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_17accum_t accum_t;
    typedef q_multi_head_attention_17_default_t bias_t;
    typedef q_multi_head_attention_17_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config16 : nnet::activ_config {
    static const unsigned n_in = 50;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_17table_t exp_table_t;
    typedef q_multi_head_attention_17table_t inv_table_t;
    typedef q_multi_head_attention_17accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config16 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_17accum_t accum_t;
    typedef q_multi_head_attention_17_default_t bias_t;
    typedef q_multi_head_attention_17_default_t weight_t;
    typedef config16_1 config_mult1;
    typedef config16_2 config_mult2;
    typedef softmax_config16 softmax_config1;

    static const unsigned num_heads = 3;
    static const unsigned head_dim_key = 16;
    static const unsigned head_dim_value = 16;
    static const unsigned feature_dim = 1;
    static const unsigned seq_len = 50;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
};

// q_dense_49
struct config17 : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef q_dense_49_default_t accum_t;
    typedef bias17_t bias_t;
    typedef weight17_t weight_t;
    typedef layer17_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_49_quantized_relu
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = 200;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_49_quantized_relu_table_t table_t;
};

// q_dense_50
struct config19 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 2;
    static const bool store_weights_in_bram = false;
    typedef q_dense_50_default_t accum_t;
    typedef bias19_t bias_t;
    typedef weight19_t weight_t;
    typedef layer19_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_34
struct config21 : nnet::merge_config {
    static const unsigned n_elem = seq_out_16*feature_out_16;
};

// add_35
struct config22 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_19*N_LAYER_2_19;
};

// q_dense_51
struct config24 : nnet::dense_config {
    static const unsigned n_in = 50;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 702;
    static const unsigned n_nonzeros = 898;
    static const bool store_weights_in_bram = false;
    typedef q_dense_51_default_t accum_t;
    typedef bias24_t bias_t;
    typedef weight24_t weight_t;
    typedef layer24_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_51_quantized_relu
struct relu_config25 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_51_quantized_relu_table_t table_t;
};

// q_dense_52
struct config26 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 172;
    static const unsigned n_nonzeros = 340;
    static const bool store_weights_in_bram = false;
    typedef q_dense_52_default_t accum_t;
    typedef bias26_t bias_t;
    typedef weight26_t weight_t;
    typedef layer26_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_52_quantized_relu
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_52_quantized_relu_table_t table_t;
};

// q_dense_53
struct config28 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 2;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 11;
    static const unsigned n_nonzeros = 21;
    static const bool store_weights_in_bram = false;
    typedef q_dense_53_default_t accum_t;
    typedef bias28_t bias_t;
    typedef weight28_t weight_t;
    typedef layer28_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_53_quantized_softmax
struct softmax_config29 : nnet::activ_config {
    static const unsigned n_in = 2;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_dense_53_quantized_softmax_table_t exp_table_t;
    typedef q_dense_53_quantized_softmax_table_t inv_table_t;
    typedef q_dense_53_quantized_softmax_default_t accum_t;
    static const unsigned inv_range = 128;
    static const unsigned exp_range = 8;
};


#endif
