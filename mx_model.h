/**
 * @file mx_model.h
 * @brief 
 * @author Haoxiang Li
 * @version 
 * @date 2017-10-25
 */
extern "C" {
#include "c_predict_api.h"
}

#include <string>
#include <vector>

class MXModel {

public:

    MXModel(const std::string& symbol_p, const std::string& params_p, unsigned int imageH, unsigned int imageW);
    void run_with_input_BGR_8UC3(const uint8_t *p_image_data);

    std::vector<int> get_output_shape(int output_idx);
    std::vector<float> get_output(int output_idx);

private:

    PredictorHandle _predictor;
    std::vector<uint8_t> _model_symbol;
    std::vector<uint8_t> _model_params;
    std::vector<float> _input_buffer;
    unsigned int _imageW;
    unsigned int _imageH;
};
