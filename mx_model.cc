/**
 * @file mx_model.cc
 * @brief 
 * @author Haoxiang Li
 * @version 
 * @date 2017-10-25
 */
#include "mx_model.h"
#include <iostream>
#include <fstream>
#include <numeric>

using namespace std;

void load_binary(const std::string& file_p, std::vector<uint8_t>& data) {

    std::ifstream fs(file_p, std::ios::binary|std::ios::in);
    fs.seekg(0, fs.end);
    data.resize(static_cast<std::size_t>(fs.tellg()));

    fs.seekg(0, fs.beg);
    fs.read(reinterpret_cast<char*>(&data.front()), data.size());
    fs.close();
}

MXModel::MXModel(const std::string& symbol_p, const std::string& params_p, 
        unsigned int imageH, unsigned int imageW) : _imageH(imageH), _imageW(imageW) {

    load_binary(symbol_p, _model_symbol);
    load_binary(params_p, _model_params);

    vector<string> input_keys = {"data"};
    const char *input_key_ptr = input_keys[0].data();
    vector<mx_uint> input_shape_ind = {0, 4};
    vector<mx_uint> input_shape_data = {1, 3, imageH, imageW};
    MXPredCreate(reinterpret_cast<char*>(_model_symbol.data()), 
            reinterpret_cast<char*>(_model_params.data()), _model_params.size(), 1, 0, 1, 
            &input_key_ptr, input_shape_ind.data(), input_shape_data.data(), &_predictor);

    _input_buffer.resize(3*imageH*imageW, 0.0f);
}

void MXModel::run_with_input_BGR_8UC3(const uint8_t *p_image_data) {
    // make it RGB, subtract mean, normalize
    float *ptrs[3] = { _input_buffer.data(),
        _input_buffer.data() + _imageH*_imageW, 
        _input_buffer.data() + _imageH*_imageW*2};
    for (size_t i = 0, out_idx = 0, in_idx = 0; i < _imageH; i++) {
        for (size_t j = 0; j < _imageW; j++) {
            ptrs[2][out_idx] = (p_image_data[in_idx++] - 128.0f)/128.0f;
            ptrs[1][out_idx] = (p_image_data[in_idx++] - 128.0f)/128.0f;
            ptrs[0][out_idx] = (p_image_data[in_idx++] - 128.0f)/128.0f;
            out_idx++;
        }
    }
    MXPredSetInput(_predictor, "data", _input_buffer.data(), _input_buffer.size());
    MXPredForward(_predictor);
}

std::vector<int> MXModel::get_output_shape(int output_idx) {
    mx_uint *shape = nullptr;
    mx_uint shape_len = 0;
    MXPredGetOutputShape(_predictor, output_idx, &shape, &shape_len);
    return vector<int>(shape, shape + shape_len);
}

std::vector<float> MXModel::get_output(int output_idx) {
    vector<int> shape = get_output_shape(output_idx);
    int tt_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    vector<float> out(tt_size);
    MXPredGetOutput(_predictor, output_idx, out.data(), out.size());
    return out;
}
