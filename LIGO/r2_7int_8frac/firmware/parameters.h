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
#include "nnet_utils/nnet_layernorm.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_multiheadattention.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/s4.h"
#include "weights/b4.h"
#include "weights/attention_output_weight5.h"
#include "weights/attention_output_bias5.h"
#include "weights/key_weight5.h"
#include "weights/key_bias5.h"
#include "weights/query_weight5.h"
#include "weights/query_bias5.h"
#include "weights/value_weight5.h"
#include "weights/value_bias5.h"
#include "weights/s7.h"
#include "weights/b7.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/s13.h"
#include "weights/b13.h"
#include "weights/attention_output_weight14.h"
#include "weights/attention_output_bias14.h"
#include "weights/key_weight14.h"
#include "weights/key_bias14.h"
#include "weights/query_weight14.h"
#include "weights/query_bias14.h"
#include "weights/value_weight14.h"
#include "weights/value_bias14.h"
#include "weights/s16.h"
#include "weights/b16.h"
#include "weights/w17.h"
#include "weights/b17.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/s22.h"
#include "weights/b22.h"
#include "weights/w23.h"
#include "weights/b23.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w28.h"
#include "weights/b28.h"

//hls-fpga-machine-learning insert layer-config
// q_dense
struct config2 : nnet::dense_config {
    static const unsigned n_in = 2;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 7;
    static const bool store_weights_in_bram = false;
    typedef q_dense_default_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef layer2_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_layer_normalization
struct config4 : nnet::layernorm_config {
    static const unsigned n_in = N_LAYER_1_2*N_LAYER_2_2;
    static const unsigned seq_len = 100;
    static const unsigned table_size = 128;
    static constexpr double table_range = 8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef q_layer_normalization_default_t bias_t;
    typedef q_layer_normalization_default_t scale_t;
    typedef q_layer_normalizationtable_t table_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_multi_head_attention
struct config5_1 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attentionaccum_t accum_t;
    typedef q_multi_head_attention_default_t bias_t;
    typedef q_multi_head_attention_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attentionaccum_t accum_t;
    typedef q_multi_head_attention_default_t bias_t;
    typedef q_multi_head_attention_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config5 : nnet::activ_config {
    static const unsigned n_in = 100;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attentiontable_t exp_table_t;
    typedef q_multi_head_attentiontable_t inv_table_t;
    typedef q_multi_head_attentionaccum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config5 : nnet::multiheadattention_config { 
    typedef q_multi_head_attentionaccum_t accum_t;
    typedef q_multi_head_attention_default_t bias_t;
    typedef q_multi_head_attention_default_t weight_t;
    typedef config5_1 config_mult1;
    typedef config5_2 config_mult2;
    typedef softmax_config5 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 4;
    static const unsigned seq_len = 100;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
};

// add
struct config6 : nnet::merge_config {
    static const unsigned n_elem = seq_out_5*feature_out_5;
};

// q_layer_normalization_1
struct config7 : nnet::layernorm_config {
    static const unsigned n_in = seq_out_5*feature_out_5;
    static const unsigned seq_len = 100;
    static const unsigned table_size = 256;
    static constexpr double table_range = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef q_layer_normalization_1_default_t bias_t;
    typedef q_layer_normalization_1_default_t scale_t;
    typedef q_layer_normalization_1table_t table_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_1
struct config8 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 15;
    static const bool store_weights_in_bram = false;
    typedef q_dense_1_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef layer8_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_1_quantized_relu
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 400;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_1_quantized_relu_table_t table_t;
};

// q_dense_2
struct config10 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 16;
    static const bool store_weights_in_bram = false;
    typedef q_dense_2_default_t accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef layer10_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_1
struct config12 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_10*N_LAYER_2_10;
};

// q_layer_normalization_2
struct config13 : nnet::layernorm_config {
    static const unsigned n_in = N_LAYER_1_10*N_LAYER_2_10;
    static const unsigned seq_len = 100;
    static const unsigned table_size = 256;
    static constexpr double table_range = 8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef q_layer_normalization_2_default_t bias_t;
    typedef q_layer_normalization_2_default_t scale_t;
    typedef q_layer_normalization_2table_t table_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_multi_head_attention_1
struct config14_1 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 32;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_1accum_t accum_t;
    typedef q_multi_head_attention_1_default_t bias_t;
    typedef q_multi_head_attention_1_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14_2 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef q_multi_head_attention_1accum_t accum_t;
    typedef q_multi_head_attention_1_default_t bias_t;
    typedef q_multi_head_attention_1_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config14 : nnet::activ_config {
    static const unsigned n_in = 100;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::legacy;
    typedef q_multi_head_attention_1table_t exp_table_t;
    typedef q_multi_head_attention_1table_t inv_table_t;
    typedef q_multi_head_attention_1accum_t accum_t;
    static const unsigned inv_range = 256;
    static const unsigned exp_range = 4;
};

struct config14 : nnet::multiheadattention_config { 
    typedef q_multi_head_attention_1accum_t accum_t;
    typedef q_multi_head_attention_1_default_t bias_t;
    typedef q_multi_head_attention_1_default_t weight_t;
    typedef config14_1 config_mult1;
    typedef config14_2 config_mult2;
    typedef softmax_config14 softmax_config1;

    static const unsigned num_heads = 2;
    static const unsigned head_dim_key = 32;
    static const unsigned head_dim_value = 32;
    static const unsigned feature_dim = 4;
    static const unsigned seq_len = 100;

    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
};

// add_2
struct config15 : nnet::merge_config {
    static const unsigned n_elem = seq_out_14*feature_out_14;
};

// q_layer_normalization_3
struct config16 : nnet::layernorm_config {
    static const unsigned n_in = seq_out_14*feature_out_14;
    static const unsigned seq_len = 100;
    static const unsigned table_size = 512;
    static constexpr double table_range = 8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef q_layer_normalization_3_default_t bias_t;
    typedef q_layer_normalization_3_default_t scale_t;
    typedef q_layer_normalization_3table_t table_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_3
struct config17 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 16;
    static const bool store_weights_in_bram = false;
    typedef q_dense_3_default_t accum_t;
    typedef bias17_t bias_t;
    typedef weight17_t weight_t;
    typedef layer17_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_3_quantized_relu
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = 400;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_3_quantized_relu_table_t table_t;
};

// q_dense_4
struct config19 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 4;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 15;
    static const bool store_weights_in_bram = false;
    typedef q_dense_4_default_t accum_t;
    typedef bias19_t bias_t;
    typedef weight19_t weight_t;
    typedef layer19_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// add_3
struct config21 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_1_19*N_LAYER_2_19;
};

// q_layer_normalization_4
struct config22 : nnet::layernorm_config {
    static const unsigned n_in = N_LAYER_1_19*N_LAYER_2_19;
    static const unsigned seq_len = 100;
    static const unsigned table_size = 128;
    static constexpr double table_range = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef q_layer_normalization_4_default_t bias_t;
    typedef q_layer_normalization_4_default_t scale_t;
    typedef q_layer_normalization_4table_t table_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_5
struct config23 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 100;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 2;
    static const bool store_weights_in_bram = false;
    typedef q_dense_5_default_t accum_t;
    typedef bias23_t bias_t;
    typedef weight23_t weight_t;
    typedef layer23_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_5_quantized_relu
struct relu_config24 : nnet::activ_config {
    static const unsigned n_in = 100;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_5_quantized_relu_table_t table_t;
};

// q_dense_6
struct config26 : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 8;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 405;
    static const unsigned n_nonzeros = 395;
    static const bool store_weights_in_bram = false;
    typedef q_dense_6_default_t accum_t;
    typedef bias26_t bias_t;
    typedef weight26_t weight_t;
    typedef layer26_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_6_quantized_relu
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 2048;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef q_dense_6_quantized_relu_table_t table_t;
};

// q_dense_7
struct config28 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 1;
    static const unsigned seq_len = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 8;
    static const bool store_weights_in_bram = false;
    typedef q_dense_7_default_t accum_t;
    typedef bias28_t bias_t;
    typedef weight28_t weight_t;
    typedef layer28_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation
struct sigmoid_config30 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    typedef activationtable_t table_t;
};


#endif
