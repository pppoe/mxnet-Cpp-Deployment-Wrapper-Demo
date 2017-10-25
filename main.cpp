/**
 * @file main.cpp
 * @brief 
 * @author Haoxiang Li
 * @version 
 * @date 2017-10-25
 */
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mx_model.h"

using namespace std;

int main(int argc, char **argv) {

    if (argc <= 4) {
        cout << "Usage:- model.json model.params synset.txt image" << endl;
        return -1;
    }

    ifstream fs(argv[3]);
    vector<string> class_names;
    string tmp_str;
    getline(fs, tmp_str);
    while (getline(fs, tmp_str)) {
        class_names.push_back(tmp_str);
    }
    fs.close();

    cout << class_names.size() << endl;

    MXModel model(argv[1], argv[2], 299, 299);

    cout << "Input: " << argv[4] << endl;

    cv::Mat image = cv::imread(argv[4], CV_LOAD_IMAGE_COLOR);
    assert(image.rows == 299 && image.cols == 299);
    model.run_with_input_BGR_8UC3(image.data);
    vector<float> outs = model.get_output(0);
    int pred_label = std::max_element(outs.begin(), outs.end()) - outs.begin();
    cout << pred_label << " " << outs[pred_label] << " " << class_names[pred_label] << endl;

    return 0;
}

