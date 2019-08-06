#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__
#include <string>

const std::string fddbRootPath = "/kyle/workspace/project/tools/evaluation/";
const std::string fddbImgDir = "/kyle/workspace/dataset/public/95_FDDB/";
const std::string fddbListFile = fddbRootPath + "model_inference/fddb_img_list.txt";
const std::string fddbDetFile = fddbRootPath + "model_inference/fddb_det.lst";
const std::string fddbAnnoFile = fddbRootPath + "model_inference/FDDB_annotation_ellipseList.txt";
const std::string fddbRocFilePrefix = fddbRootPath + "model_inference/tmp";
#endif