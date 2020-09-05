#ifndef BASE_OPTIMIZE_H_
#define BASE_OPTIMIZE_H_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "file_io.h"
#include "util.h"
#include "house.h"
#include "cad.h"

using std::cerr;
using std::endl;

// Optimization settings (change to find the optimal):
/*
1. Angle: we set it to 0 since garage/dormer/chimney may change positions; 
  We could change it to non-zero if garage optm is involved (we will implement); and dormer/chimney position adjustment is implemented (we may not implement)

2. TODO Garage: We will add it into optimization; (only exhaustive search)
  Note that snapping could also be included but we may not do it as well.

3. Chimney: To be checked. (We may not include position changes) Before checking, we use fixed params.

4. Dormer: To be checked. Before checking, we use fixed params.

5. TODO Ceres optimization: We don't optimize footprint in Ceres, since it's troublesome to adjust code with different number of garages

*/

namespace {

auto regu_lower_positive = [](const double ele) {return static_cast<double>(std::max(ele, 0.0));};
auto regu_lower_ratio_s = [](const double ratio_s) {return static_cast<double>(std::max(ratio_s, 0.0));};
auto regu_upper_ratio_s = [](const double ratio_s) {return static_cast<double>(std::min(ratio_s, 0.4999));};
auto regu_lower_ratio_e = [](const double ratio_e) {return static_cast<double>(std::max(ratio_e, 0.5001));};
auto regu_upper_ratio_e = [](const double ratio_e) {return static_cast<double>(std::min(ratio_e, 1.0));};

void ReguBound(const int ind, const int ind_inner, const std::vector<int> rooftypes, double& lower, double& upper){
  if(ind == 1 || ind == 2 || ind == 5 || ind == 6 || ind == 7 || ind == 8 )
    lower = regu_lower_positive(lower);
  if(ind == 3) {
    if(rooftypes[ind_inner] == 1) { lower = 0.0; upper = 0.0; }
    else {lower = regu_lower_ratio_s(lower); upper = regu_upper_ratio_s(upper);}
  }
  if(ind == 4) {
    if(rooftypes[ind_inner] == 1) { lower = 1.0; upper = 1.0;}
    else { lower = regu_upper_ratio_e(lower); upper = regu_upper_ratio_e(upper);}
  }
}

struct ParasModule {
  // std::vector<double> angles;
  // std::vector<double> tls_y; std::vector<double> tls_x;
  // std::vector<double> ratios_s; std::vector<double> ratios_e;
  // std::vector<double> heis; std::vector<double> wids;
  // std::vector<double> hs_ridge; std::vector<double> hs_eave;
  // std::vector<double> angles_delta;
  std::vector<std::vector<double> > params; // TODO(Henry: we need to init it such that the index makes sense)
  // std::vector<double> offset_ = {3, 3, 3, 0.2, 0.2, 3, 3, 2, 2, 0};
  std::vector<double> offset_ = {0, 3, 3, 0.2, 0.2, 3, 3, 2, 2, 0};
  std::vector<int> rooftypes_;
  std::vector<DPM::IFlat> iflats_;
  std::vector<DPM::LFlat> lflats_;
  DPM::ModuleType moduletype;
  inline void init() {
    for (int ind_params = 0; ind_params < 10; ++ind_params) {
      std::vector<double> temp_param;
      params.push_back(temp_param);
    }
  }

  friend std::ostream& operator<<(std::ostream& ostr, const ParasModule& lhs) {
    ostr << "ParasModule" << std::endl;
    for(auto&& params_one_type : lhs.params) {
      for(auto&& ele : params_one_type) {
        ostr << ele << ", ";
      }
      ostr << " || ";
    }
    ostr << endl;

    return ostr;
  };

  void DoubleVectorToParas(const std::vector<int>& type_paras, const std::vector<double>& v_paras) {
    for(auto&& params_one_type : params)
      params_one_type.clear();
    for(unsigned i = 0; i < type_paras.size(); ++i) {
      params[type_paras[i]].push_back(v_paras[i]);
    }
  }
  void DoubleVectorToParas(const std::vector<int>& type_paras, const double* v_paras) {
    for(auto&& params_one_type : params)
      params_one_type.clear();
    // sizeof(type_paras)/sizeof(type_paras)
    // for(unsigned i = 0; i < sizeof(type_paras)/sizeof(type_paras); ++i)
    for(unsigned i = 0; i < type_paras.size(); ++i)
      params[type_paras[i]].push_back(v_paras[i]);
  }

  void ParasToDoubleVector(std::vector<int>& type_paras, std::vector<double>& v_paras, std::vector<int>& constant_inds,
    std::vector<int>& upper_bound_inds, std::vector<double>& upper_bound_v,
    std::vector<int>& lower_bound_inds, std::vector<double>& lower_bound_v) {
      for(auto&& param_angle : params[0]) {
          type_paras.push_back(0); v_paras.push_back(param_angle);
          // todo: let house not rotate
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(param_angle + 0.0);
          upper_bound_inds.push_back(v_paras.size()-1); upper_bound_v.push_back(param_angle + 0.0001);
      }
      for(auto&& param_tls_y : params[1]) {
          type_paras.push_back(1); v_paras.push_back(param_tls_y); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(auto&& param_tls_x : params[2]) {
          type_paras.push_back(2); v_paras.push_back(param_tls_x); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(unsigned ind_inner = 0; ind_inner < params[3].size(); ++ind_inner) {
          type_paras.push_back(3); v_paras.push_back(params[3][ind_inner]); 
          if(rooftypes_[ind_inner] == 1){
            constant_inds.push_back(v_paras.size()-1);
          }
          else{
            lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0.0);
            upper_bound_inds.push_back(v_paras.size()-1); upper_bound_v.push_back(0.4999);
          }  
      }
      for(unsigned ind_inner = 0; ind_inner < params[4].size(); ++ind_inner) {
          type_paras.push_back(4); v_paras.push_back(params[4][ind_inner]); 
          if(rooftypes_[ind_inner] == 1){
            constant_inds.push_back(v_paras.size()-1);
          }
          else{
            lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0.5001);
            upper_bound_inds.push_back(v_paras.size()-1); upper_bound_v.push_back(1.0);
          }  
      }
      for(auto&& param_heis : params[5]) {
          type_paras.push_back(5); v_paras.push_back(param_heis); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(auto&& param_wids : params[6]) {
          type_paras.push_back(6); v_paras.push_back(param_wids); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(auto&& param_hs_ridge : params[7]) {
          type_paras.push_back(7); v_paras.push_back(param_hs_ridge); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(auto&& param_hs_eave : params[8]) {
          type_paras.push_back(8); v_paras.push_back(param_hs_eave); 
          lower_bound_inds.push_back(v_paras.size()-1); lower_bound_v.push_back(0);
      }
      for(auto&& param_angles_delta : params[9]) {
          type_paras.push_back(9); v_paras.push_back(param_angles_delta); 
          constant_inds.push_back(v_paras.size()-1);
      }
  }
  
  void cartesian(const int k_samples, const int k_subset, std::vector<ParasModule>& samples_para_module) {
    int n_valid_params = 0;
    const int k_cartesian = 10;
    for (int ind_params = 0; ind_params < k_cartesian; ++ind_params)
      n_valid_params += params[ind_params].size();

    std::vector<std::vector<std::vector<double> > > samples_all;
    for (int ind_params = 0; ind_params < k_cartesian; ++ind_params) {
      std::vector<std::vector<double> > samples_one_type;
      // for(auto&& param : params[ind_params]) {
      for(unsigned ind_inner = 0; ind_inner < params[ind_params].size(); ++ind_inner) {
        double lower = params[ind_params][ind_inner] - offset_[ind_params];
        double upper = params[ind_params][ind_inner] + offset_[ind_params];
        ReguBound(ind_params, ind_inner, rooftypes_, lower, upper);
        double step = (upper - lower) / static_cast<double>(k_samples);
        std::vector<double> samples;
        for (int i = 0; i < k_samples; ++i)
          samples.push_back(i * step + lower);
        samples_one_type.push_back(samples);
      }      
      samples_all.push_back(samples_one_type);
    }
    // getting the cartesian product 
    long long int num_all_sample = 1;
    for (unsigned i = 0; i < k_subset; ++i)
      num_all_sample *= k_samples;

    std::vector<int> inds_shuffled;
    for (int i = 0; i < n_valid_params; ++i)
      inds_shuffled.push_back(i);
    std::random_shuffle (inds_shuffled.begin(), inds_shuffled.end() );
    inds_shuffled.resize(k_subset);

    for (unsigned ind_sample = 0; ind_sample < num_all_sample; ++ind_sample) {
      ParasModule temp_para_module;
      std::vector<std::vector<double> > sample = params;
      int jj = k_subset - 1;
      int step_changing = 0;
      for (int ii = 0; ii < 10; ++ii) {
        if (std::find(inds_shuffled.begin(), inds_shuffled.end(), ii) == inds_shuffled.end())
          continue;
        for(int kk = 0; kk < params[ii].size(); ++kk) {
          if (jj == k_subset - 1)
            step_changing = ind_sample / std::pow(k_samples, jj);
          if (jj == 0)
            step_changing = ind_sample % k_samples;
          else
            step_changing = static_cast<int>(ind_sample / std::pow(k_samples, jj)) % k_samples;
          sample[ii][kk] = samples_all[ii][kk][step_changing];
          jj --;
        }
      }
      temp_para_module.params = sample;
      temp_para_module.rooftypes_ = rooftypes_;
      temp_para_module.iflats_ = iflats_;
      temp_para_module.lflats_ = lflats_;
      temp_para_module.moduletype = moduletype;
      samples_para_module.push_back(temp_para_module);
    }
  }

}; // end of struct

struct options {
  std::string filename_pts;
  std::string filename_paras;
  std::string filename_mesh_o;
};

}; // end of empty namespce
enum struct DormerParam{
  center_x_offset = 0,
  center_y_offset,
  width,
  height,
  triangular_ratio
};
struct DormerParamSearching{
  int dormer_no;
  DormerParam param;
};

namespace DPM {
  bool RunOptHouses(HouseGroup & housegroup);
  bool RunOptDormers(HouseGroup & housegroup, FileIO &file_io);
  bool RunOptChimneys(HouseGroup & housegroup, FileIO &file_io);
} // DPM


#endif  // BASE_OPTIMIZE_NEW_H_
