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
// q_multi_head_attention_45
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_45accum_t accum_t;
    typedef q_multi_head_attention_45_default_t bias_t;
    typedef q_multi_head_attention_45_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_45accum_t accum_t;
    typedef q_multi_head_attention_45_default_t bias_t;
    typedef q_multi_head_attention_45_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config2 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_45table_t exp_table_t;
    typedef q_multi_head_attention_45table_t inv_table_t;
    typedef q_multi_head_attention_45accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config2 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_45accum_t accum_t;
    typedef q_multi_head_attention_45_default_t bias_t;
    typedef q_multi_head_attention_45_default_t weight_t;
    typedef config2_1 config_mult1;
    typedef config2_2 config_mult2;
    typedef softmax_config2 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const bool store_weights_in_bram = false;
};

// add_90
struct config3 : nnet::merge_config {
    static const unsigned n_elem = seq_out_2*feature_out_2;
};

// q_dense_150
struct config4 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 12;
    static const unsigned n_nonzeros = 36;
    static const bool store_weights_in_bram = false;
    typedef q_dense_150accum_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_150_quantized_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_150_quantized_relutable_t table_t;
};

// q_dense_151
struct config6 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 47;
    static const bool store_weights_in_bram = false;
    typedef q_dense_151accum_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef layer6_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_91
struct config8 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_6*N_LAYER_2_6;
};

// q_multi_head_attention_46
struct config9_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_46accum_t accum_t;
    typedef q_multi_head_attention_46_default_t bias_t;
    typedef q_multi_head_attention_46_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_46accum_t accum_t;
    typedef q_multi_head_attention_46_default_t bias_t;
    typedef q_multi_head_attention_46_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config9 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_46table_t exp_table_t;
    typedef q_multi_head_attention_46table_t inv_table_t;
    typedef q_multi_head_attention_46accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config9 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_46accum_t accum_t;
    typedef q_multi_head_attention_46_default_t bias_t;
    typedef q_multi_head_attention_46_default_t weight_t;
    typedef config9_1 config_mult1;
    typedef config9_2 config_mult2;
    typedef softmax_config9 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const bool store_weights_in_bram = false;
};

// add_92
struct config10 : nnet::merge_config {
    static const unsigned n_elem = seq_out_9*feature_out_9;
};

// q_dense_152
struct config11 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 5;
    static const unsigned n_nonzeros = 43;
    static const bool store_weights_in_bram = false;
    typedef q_dense_152accum_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef layer11_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_152_quantized_relu
struct relu_config12 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_152_quantized_relutable_t table_t;
};

// q_dense_153
struct config13 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const bool store_weights_in_bram = false;
    typedef q_dense_153accum_t accum_t;
    typedef bias13_t bias_t;
    typedef weight13_t weight_t;
    typedef layer13_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_93
struct config15 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_13*N_LAYER_2_13;
};

// q_multi_head_attention_47
struct config16_1 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_47accum_t accum_t;
    typedef q_multi_head_attention_47_default_t bias_t;
    typedef q_multi_head_attention_47_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 384;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_47accum_t accum_t;
    typedef q_multi_head_attention_47_default_t bias_t;
    typedef q_multi_head_attention_47_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config16 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_47table_t exp_table_t;
    typedef q_multi_head_attention_47table_t inv_table_t;
    typedef q_multi_head_attention_47accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config16 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_47accum_t accum_t;
    typedef q_multi_head_attention_47_default_t bias_t;
    typedef q_multi_head_attention_47_default_t weight_t;
    typedef config16_1 config_mult1;
    typedef config16_2 config_mult2;
    typedef softmax_config16 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 6;
    static const unsigned seq_len = 15;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const bool store_weights_in_bram = false;
};

// add_94
struct config17 : nnet::merge_config {
    static const unsigned n_elem = seq_out_16*feature_out_16;
};

// q_dense_154
struct config18 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 3;
    static const unsigned n_nonzeros = 45;
    static const bool store_weights_in_bram = false;
    typedef q_dense_154accum_t accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    typedef layer18_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_154_quantized_relu
struct relu_config19 : nnet::activ_config {
    static const unsigned n_in = 120;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_154_quantized_relutable_t table_t;
};

// q_dense_155
struct config20 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;
    static const unsigned seq_len = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 6;
    static const unsigned n_nonzeros = 42;
    static const bool store_weights_in_bram = false;
    typedef q_dense_155accum_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    typedef layer20_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_95
struct config22 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_20*N_LAYER_2_20;
};

// q_dense_156
struct config24 : nnet::dense_config {
    static const unsigned n_in = 90;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 464;
    static const unsigned n_nonzeros = 2416;
    static const bool store_weights_in_bram = false;
    typedef q_dense_156accum_t accum_t;
    typedef bias24_t bias_t;
    typedef weight24_t weight_t;
    typedef layer24_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_156_quantized_relu
struct relu_config25 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_156_quantized_relutable_t table_t;
};

// q_dense_157
struct config26 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 59;
    static const unsigned n_nonzeros = 453;
    static const bool store_weights_in_bram = false;
    typedef q_dense_157accum_t accum_t;
    typedef bias26_t bias_t;
    typedef weight26_t weight_t;
    typedef layer26_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_157_quantized_relu
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_157_quantized_relutable_t table_t;
};

// q_dense_158
struct config28 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 12;
    static const unsigned n_nonzeros = 116;
    static const bool store_weights_in_bram = false;
    typedef q_dense_158accum_t accum_t;
    typedef bias28_t bias_t;
    typedef weight28_t weight_t;
    typedef layer28_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_158_quantized_relu
struct relu_config29 : nnet::activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    typedef q_dense_158_quantized_relutable_t table_t;
};

// q_dense_159
struct config30 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 3;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 24;
    static const bool store_weights_in_bram = false;
    typedef q_dense_159accum_t accum_t;
    typedef bias30_t bias_t;
    typedef weight30_t weight_t;
    typedef layer30_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_159_quantized_softmax
struct softmax_config31 : nnet::activ_config {
    static const unsigned n_in = 3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 4;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_dense_159_quantized_softmaxtable_t exp_table_t;
    typedef q_dense_159_quantized_softmaxtable_t inv_table_t;
    typedef q_dense_159_quantized_softmaxaccum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};


#endif
