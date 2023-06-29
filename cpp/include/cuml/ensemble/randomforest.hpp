/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>
#include <cuml/tree/decisiontree.hpp>

#include <map>
#include <memory>

namespace raft {
class handle_t;  // forward decl
}

namespace ML {

enum RF_type {
  CLASSIFICATION,
  REGRESSION,
};

enum task_category { REGRESSION_MODEL = 1, CLASSIFICATION_MODEL = 2 };

struct RF_metrics {
  RF_type rf_type;

  // Classification metrics
  float accuracy;

  // Regression metrics
  double mean_abs_error;
  double mean_squared_error;
  double median_abs_error;
};

void lower_bound_test(
    const int* search_array,
    const int* input_vals,
    const int num_search_vals,
    const int num_inputs,
    int* output_array,
    const cudaStream_t stream);

RF_metrics set_all_rf_metrics(RF_type rf_type,
                              float accuracy,
                              double mean_abs_error,
                              double mean_squared_error,
                              double median_abs_error);
RF_metrics set_rf_metrics_classification(float accuracy);
RF_metrics set_rf_metrics_regression(double mean_abs_error,
                                     double mean_squared_error,
                                     double median_abs_error);
void print(const RF_metrics rf_metrics);

struct RF_params {
  /**
   * Number of decision trees in the random forest.
   */
  int n_trees;
  /**
   * Control bootstrapping.
   * If bootstrapping is set to true, bootstrapped samples are used for building
   * each tree. Bootstrapped sampling is done by randomly drawing
   * round(max_samples * n_samples) number of samples with replacement. More on
   * bootstrapping:
   *     https://en.wikipedia.org/wiki/Bootstrap_aggregating
   * If bootstrapping is set to false, whole dataset is used to build each
   * tree.
   */
  bool bootstrap;

  /** 
   * Control whether to use honesty features to allow causal inferencing
   * 
   * This indicates that the values used for averaging in the leaf node predictions
   * should be a disjoint set with the labels used for splits during training. 
   * See this issue for more detail: https://github.com/rapidsai/cuml/issues/5253
  */
  bool oob_honesty;

  /** 
   * Honesty double bootstrapping
   * 
   * With double bootstrapping, the set of samples that was not sampled for training
   * is again sampled with replacement. This leaves some samples that could be used 
   * for double OOB prediction. TODO: how can we make the user aware of which 
   * samples could be used for double OOB prediction?
  */
  bool double_bootstrap;

  /**
   * Ratio of dataset rows used while fitting each tree.
   */
  float max_samples;

  /**
   * Comment from rforestry:
   * The number of trees which we make sure have been created leaving
   * out each fold (each fold is a set of randomly selected groups).
   *  This is 0 by default, so we will not give any special treatment to
   * the groups when sampling observations, however if this is set to a positive integer, we
   * modify the bootstrap sampling scheme to ensure that exactly that many trees
   * have each group left out. We do this by, for each fold, creating minTreesPerGroupFold
   * trees which are built on observations sampled from the set of training observations
   * which are not in a group in the current fold. The folds form a random partition of
   * all of the possible groups, each of size foldGroupSize. This means we create at
   * least # folds * minTreesPerGroupFold trees for the forest.
   * If ntree > # folds * minTreesPerGroupFold, we create
   * max(# folds * minTreesPerGroupFold, ntree) total trees, in which at least minTreesPerGroupFold
   * are created leaving out each fold.
  */
  int minTreesPerGroupFold;

  /**
   * Comment from rforestry:
   * The number of groups that are selected randomly for each fold to be
   * left out when using minTreesPerGroupFold. When minTreesPerGroupFold is set and foldGroupSize is
   * set, all possible groups will be partitioned into folds, each containing foldGroupSize unique groups
   * (if foldGroupSize doesn't evenly divide the number of groups, a single fold will be smaller,
   * as it will contain the remaining groups). Then minTreesPerGroupFold are grown with each
   * entire fold of groups left out.
   */
  int foldGroupSize;

  /**
   * group_col_idx
   * The numeric index of the column to be used for group processing
   */
  int group_col_idx;
  
  /**
   * Decision tree training hyper parameter struct.
   */
  /**
   * random seed
   */
  uint64_t seed;
  /**
   * Number of concurrent GPU streams for parallel tree building.
   * Each stream is independently managed by CPU thread.
   * N streams need N times RF workspace.
   */
  int n_streams;
  DT::DecisionTreeParams tree_params;
};

/* Update labels so they are unique from 0 to n_unique_vals.
   Create an old_label to new_label map per random forest.
*/
void preprocess_labels(int n_rows,
                       std::vector<int>& labels,
                       std::map<int, int>& labels_map,
                       int verbosity = CUML_LEVEL_INFO);

/* Revert preprocessing effect, if needed. */
void postprocess_labels(int n_rows,
                        std::vector<int>& labels,
                        std::map<int, int>& labels_map,
                        int verbosity = CUML_LEVEL_INFO);

template <class T, class L>
struct RandomForestMetaData {
  std::vector<std::shared_ptr<DT::TreeMetaDataNode<T, L>>> trees;
  RF_params rf_params;
};

template <class T, class L>
void delete_rf_metadata(RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_summary_text(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_detailed_text(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_json(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
void build_treelite_forest(ModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features);

ModelHandle concatenate_trees(std::vector<ModelHandle> treelite_handles);

void compare_concat_forest_to_subforests(ModelHandle concat_tree_handle,
                                         std::vector<ModelHandle> treelite_handles);
// ----------------------------- Classification ----------------------------------- //

typedef RandomForestMetaData<float, int> RandomForestClassifierF;
typedef RandomForestMetaData<double, int> RandomForestClassifierD;

void fit(const raft::handle_t& user_handle,
         RandomForestClassifierF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         int verbosity = CUML_LEVEL_INFO);
void fit(const raft::handle_t& user_handle,
         RandomForestClassifierD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         int verbosity = CUML_LEVEL_INFO);

void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             int verbosity = CUML_LEVEL_INFO);
void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             int verbosity = CUML_LEVEL_INFO);

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierF* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 int verbosity = CUML_LEVEL_INFO);
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierD* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 int verbosity = CUML_LEVEL_INFO);

RF_params set_rf_params(int max_depth,
                        int max_leaves,
                        float max_features,
                        int max_n_bins,
                        int min_samples_leaf_splitting,
                        int min_samples_leaf_averaging,
                        int min_samples_split_splitting,
                        int min_samples_split_averaging,
                        float min_impurity_decrease,
                        bool bootstrap,
                        bool oob_honesty,
                        bool double_bootstrap,
                        int n_trees,
                        float max_samples,
                        uint64_t seed,
                        CRITERION split_criterion,
                        int cfg_n_streams,
                        int max_batch_size,
                        int minTreesPerGroupFold,
                        int foldGroupSize,
                        int group_col_idx);

// ----------------------------- Regression ----------------------------------- //

typedef RandomForestMetaData<float, float> RandomForestRegressorF;
typedef RandomForestMetaData<double, double> RandomForestRegressorD;

void fit(const raft::handle_t& user_handle,
         RandomForestRegressorF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         RF_params rf_params,
         int verbosity = CUML_LEVEL_INFO);
void fit(const raft::handle_t& user_handle,
         RandomForestRegressorD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         RF_params rf_params,
         int verbosity = CUML_LEVEL_INFO);

void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             float* predictions,
             int verbosity = CUML_LEVEL_INFO);
void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             double* predictions,
             int verbosity = CUML_LEVEL_INFO);

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorF* forest,
                 const float* ref_labels,
                 int n_rows,
                 const float* predictions,
                 int verbosity = CUML_LEVEL_INFO);
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorD* forest,
                 const double* ref_labels,
                 int n_rows,
                 const double* predictions,
                 int verbosity = CUML_LEVEL_INFO);
};  // namespace ML
