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
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/attention_output_weight9.h"
#include "weights/attention_output_bias9.h"
#include "weights/key_weight9.h"
#include "weights/key_bias9.h"
#include "weights/query_weight9.h"
#include "weights/query_bias9.h"
#include "weights/value_weight9.h"
#include "weights/value_bias9.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/attention_output_weight16.h"
#include "weights/attention_output_bias16.h"
#include "weights/key_weight16.h"
#include "weights/key_bias16.h"
#include "weights/query_weight16.h"
#include "weights/query_bias16.h"
#include "weights/value_weight16.h"
#include "weights/value_bias16.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/w24.h"
#include "weights/b24.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w28.h"
#include "weights/b28.h"
#include "weights/w30.h"
#include "weights/b30.h"

//hls-fpga-machine-learning insert layer-config
// multi_head_attention_2
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_2accum_t accum_t;
    typedef multi_head_attention_2_default_t bias_t;
    typedef multi_head_attention_2_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_2accum_t accum_t;
    typedef multi_head_attention_2_default_t bias_t;
    typedef multi_head_attention_2_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config2 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef multi_head_attention_2table_t exp_table_t;
    typedef multi_head_attention_2table_t inv_table_t;
    typedef multi_head_attention_2accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config2 : nnet::multiheadattention_config { 
    typedef multi_head_attention_2accum_t accum_t;
    typedef multi_head_attention_2_default_t bias_t;
    typedef multi_head_attention_2_default_t weight_t;
    typedef config2_1 config_mult1;
    typedef config2_2 config_mult2;
    typedef softmax_config2 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// add_4
struct config3 : nnet::merge_config {
    static const unsigned n_elem = seq_out_2*feature_out_2;
};

// dense_8
struct config4 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_8accum_t accum_t;
    typedef dense_8_default_t bias_t;
    typedef dense_8_default_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_8_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_8_relutable_t table_t;
};

// dense_9
struct config6 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_9accum_t accum_t;
    typedef dense_9_default_t bias_t;
    typedef dense_9_default_t weight_t;
    typedef layer6_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_5
struct config8 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_6*N_LAYER_2_6;
};

// multi_head_attention_3
struct config9_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_3accum_t accum_t;
    typedef multi_head_attention_3_default_t bias_t;
    typedef multi_head_attention_3_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_3accum_t accum_t;
    typedef multi_head_attention_3_default_t bias_t;
    typedef multi_head_attention_3_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config9 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef multi_head_attention_3table_t exp_table_t;
    typedef multi_head_attention_3table_t inv_table_t;
    typedef multi_head_attention_3accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config9 : nnet::multiheadattention_config { 
    typedef multi_head_attention_3accum_t accum_t;
    typedef multi_head_attention_3_default_t bias_t;
    typedef multi_head_attention_3_default_t weight_t;
    typedef config9_1 config_mult1;
    typedef config9_2 config_mult2;
    typedef softmax_config9 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// add_6
struct config10 : nnet::merge_config {
    static const unsigned n_elem = seq_out_9*feature_out_9;
};

// dense_10
struct config11 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_10accum_t accum_t;
    typedef dense_10_default_t bias_t;
    typedef dense_10_default_t weight_t;
    typedef layer11_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_10_relu
struct relu_config12 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_10_relutable_t table_t;
};

// dense_11
struct config13 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_11accum_t accum_t;
    typedef dense_11_default_t bias_t;
    typedef dense_11_default_t weight_t;
    typedef layer13_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_7
struct config15 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_13*N_LAYER_2_13;
};

// multi_head_attention_4
struct config16_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_4accum_t accum_t;
    typedef multi_head_attention_4_default_t bias_t;
    typedef multi_head_attention_4_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef multi_head_attention_4accum_t accum_t;
    typedef multi_head_attention_4_default_t bias_t;
    typedef multi_head_attention_4_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config16 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef multi_head_attention_4table_t exp_table_t;
    typedef multi_head_attention_4table_t inv_table_t;
    typedef multi_head_attention_4accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config16 : nnet::multiheadattention_config { 
    typedef multi_head_attention_4accum_t accum_t;
    typedef multi_head_attention_4_default_t bias_t;
    typedef multi_head_attention_4_default_t weight_t;
    typedef config16_1 config_mult1;
    typedef config16_2 config_mult2;
    typedef softmax_config16 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// add_8
struct config17 : nnet::merge_config {
    static const unsigned n_elem = seq_out_16*feature_out_16;
};

// dense_12
struct config18 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_12accum_t accum_t;
    typedef dense_12_default_t bias_t;
    typedef dense_12_default_t weight_t;
    typedef layer18_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_12_relu
struct relu_config19 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_12_relutable_t table_t;
};

// dense_13
struct config20 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef dense_13accum_t accum_t;
    typedef dense_13_default_t bias_t;
    typedef dense_13_default_t weight_t;
    typedef layer20_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_9
struct config22 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_20*N_LAYER_2_20;
};

// dense_14
struct config24 : nnet::dense_config {
    static const unsigned n_in = 90;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 2880;
    static const bool store_weights_in_bram = false;
    typedef dense_14accum_t accum_t;
    typedef dense_14_default_t bias_t;
    typedef dense_14_default_t weight_t;
    typedef layer24_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_14_relu
struct relu_config25 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_14_relutable_t table_t;
};

// dense_15
struct config26 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 512;
    static const bool store_weights_in_bram = false;
    typedef dense_15accum_t accum_t;
    typedef dense_15_default_t bias_t;
    typedef dense_15_default_t weight_t;
    typedef layer26_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_15_relu
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_15_relutable_t table_t;
};

// dense_16
struct config28 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 128;
    static const bool store_weights_in_bram = false;
    typedef dense_16accum_t accum_t;
    typedef dense_16_default_t bias_t;
    typedef dense_16_default_t weight_t;
    typedef layer28_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_16_relu
struct relu_config29 : nnet::activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef dense_16_relutable_t table_t;
};

// dense_17
struct config30 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 3;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 24;
    static const bool store_weights_in_bram = false;
    typedef dense_17accum_t accum_t;
    typedef dense_17_default_t bias_t;
    typedef dense_17_default_t weight_t;
    typedef layer30_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_17_softmax
struct softmax_config31 : nnet::activ_config {
    static const unsigned n_in = 3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef dense_17_softmaxtable_t exp_table_t;
    typedef dense_17_softmaxtable_t inv_table_t;
    typedef dense_17_softmaxaccum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};


#endif
