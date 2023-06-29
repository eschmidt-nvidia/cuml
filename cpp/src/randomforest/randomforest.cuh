/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <decisiontree/decisiontree.cuh>
#include <decisiontree/treelite_util.h>

#include <raft/random/permute.cuh>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/accuracy.cuh>
#include <raft/stats/regression_metrics.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#endif

#include <map>

struct set_mask_functor {
    const int n_rows;
    set_mask_functor(const int n_rows) 
      : n_rows(n_rows)
    {}
    
    __host__ __device__
    void operator()(const int& index, bool& output)
    {
      output = index < n_rows;
    }
};

namespace {

__global__ void log10(int* array) {
  for (int ix = 0; ix < 10; ++ix) {
    printf("array %d = %d\n", ix, array[ix]);
  }
}

__global__ void log10groups(const int* row_ids, const int* group_ids) {
  for (int ix = 0; ix < 10; ++ix) {
    printf("group ix %d, row %d = %d\n", ix, row_ids[ix], group_ids[row_ids[ix]]);
  }
}

void assign_groups_to_folds(
  int n_groups,
  int n_folds,
  int fold_size,
  std::vector<std::vector<int>> & fold_memberships,
  std::mt19937& rng) 
{
    std::vector<int> group_indices(n_groups);
    std::iota(group_indices.begin(), group_indices.end(), 0);

    std::shuffle(group_indices.begin(), group_indices.end(), rng);
          
    for (int ix_fold = 0; ix_fold < n_folds - 1; ++ix_fold) {
      std::copy(group_indices.begin() + ix_fold*fold_size,
                group_indices.begin() + (ix_fold+1)*fold_size,
                fold_memberships[ix_fold].begin());
      // std::sort(fold_memberships[ix_fold].begin(), fold_memberships[ix_fold].end());
    }

    // Last fold could be smaller
    const int last_fold_start = (n_folds - 1) * fold_size;
    const int last_fold_size = n_groups - last_fold_start;
    fold_memberships[n_folds - 1].resize(last_fold_size);
    for (int ix = 0; ix < last_fold_size; ++ix) {
      fold_memberships[n_folds - 1][ix] = group_indices[last_fold_start + ix];
    }
}

template<typename T, typename U>
__device__ int lower_bound(const T search_val, const U* array, int count) {
  int it, step;
  int first = 0;
  while (count > 0) {
    step = count / 2;
    it = first + step;
    if (array[it] < search_val) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template<typename T>
struct UniqueTransformFunctor {
  const T* unique_groups;
  const int num_groups;
  UniqueTransformFunctor(const T* unique_groups, const int num_groups)
    : unique_groups(unique_groups),
      num_groups(num_groups)
  {}

  __device__ int operator()(T group_val) {
    int res = lower_bound(group_val, unique_groups, num_groups);
    return res;
  }
};

struct LeaveOutSamplesCopyIfFunctor {
  
  int* remaining_groups;
  const int* sample_group_ids;
  const int num_rem_groups;
  LeaveOutSamplesCopyIfFunctor(
      rmm::device_uvector<int>* remaining_groups,
      const int* sample_group_ids)
    : remaining_groups(remaining_groups->data()),
      sample_group_ids(sample_group_ids),
      num_rem_groups(remaining_groups->size())
  {}

  __device__ bool operator()(const int ix_sample) {
    // Do a quick lower_bound search
    const int group_id = sample_group_ids[ix_sample];
    int it = lower_bound(group_id, remaining_groups, num_rem_groups);
    return remaining_groups[it] == group_id;
  }
};

void generate_row_indices_from_remaining_groups(
    rmm::device_uvector<int>* remaining_groups,
    rmm::device_uvector<int>* remaining_samples,
    const int* sample_group_ids,
    const size_t num_samples,
    const cudaStream_t stream,
    raft::random::Rng& rng)
{
  // From the remaining groups, we need to generate the remaining samples
  // We're going to copy_if the indices to remaining_samples, only if the sample group id is part of the remaining groups

  // We want to sort remaining groups so that we can do the search in logN time.
  thrust::sort(thrust::cuda::par.on(stream), remaining_groups->begin(), remaining_groups->end());

  LeaveOutSamplesCopyIfFunctor predicate{remaining_groups, sample_group_ids};

  auto output_it = thrust::copy_if(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_samples),
      remaining_samples->begin(),
      predicate);
  auto dist = thrust::distance(remaining_samples->begin(), output_it);
  remaining_samples->resize(dist, stream);
}

void sample_rows_from_remaining_rows(
    rmm::device_uvector<int>* remaining_samples,
    rmm::device_uvector<int>* sample_output,
    rmm::device_uvector<int>* workspace,
    const size_t num_samples,
    const size_t sample_output_offset,
    const cudaStream_t stream,
    raft::random::Rng& rng)
{
  // This will do the actual permutation / shuffle operation
  rng.uniformInt<int>(workspace->data(), num_samples, 0, remaining_samples->size(), stream);

  auto index_iter = thrust::make_permutation_iterator(remaining_samples->begin(), workspace->begin());

  thrust::copy(thrust::cuda::par.on(stream), index_iter, index_iter + num_samples, sample_output->begin() + sample_output_offset);
}

void leave_groups_out_sample(
    rmm::device_uvector<int>* remaining_groups,
    rmm::device_uvector<int>* remaining_samples,
    rmm::device_uvector<int>* sample_output,
    rmm::device_uvector<int>* workspace,
    const int* sample_group_ids,
    std::vector<int>& remaining_groups_host,
    const size_t num_samples,
    const size_t sample_output_offset,
    const cudaStream_t stream,
    raft::random::Rng& rng)
{
  raft::update_device(remaining_groups->data(), remaining_groups_host.data(), 
      remaining_groups_host.size(), stream);
  remaining_groups->resize(remaining_groups_host.size(), stream);
  
  generate_row_indices_from_remaining_groups(
    remaining_groups, remaining_samples, sample_group_ids, 
    num_samples, stream, rng);
  sample_rows_from_remaining_rows(
    remaining_samples,
    sample_output,
    workspace,
    num_samples,
    sample_output_offset,
    stream,
    rng);
}

void update_averaging_mask(
  rmm::device_uvector<bool>* split_row_mask,
  const size_t n_sampled_rows,
  const cudaStream_t stream)
{
    // First n_rows goes to training, num_avg_samples goes to averaging.
    auto begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_counting_iterator<int>(0), split_row_mask->begin()));
    auto end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_counting_iterator<int>(2 * n_sampled_rows), split_row_mask->end()));
    thrust::for_each(thrust::cuda::par.on(stream), begin_zip_iterator, end_zip_iterator, 
        thrust::make_zip_function(set_mask_functor(n_sampled_rows)));
}

} // end anon namespace

namespace ML {

void lower_bound_test(
    const int* search_array,
    const int* input_vals,
    const int num_search_vals,
    const int num_inputs,
    int* output_array,
    const cudaStream_t stream) 
{
  UniqueTransformFunctor transform_fn{search_array, num_search_vals};
  thrust::transform(
      thrust::cuda::par.on(stream),
      input_vals,
      input_vals + num_inputs,
      output_array,
      transform_fn);
}

template <class T, class L>
class RandomForest {
 protected:
  RF_params rf_params;  // structure containing RF hyperparameters
  int rf_type;          // 0 for classification 1 for regression

  size_t get_row_sample(const int tree_id,
                        const int n_rows,
                        int n_sampled_rows,
                        rmm::device_uvector<int>* selected_rows, // 2x n sampled rows
                        rmm::device_uvector<bool>* split_row_mask, // 2x n_sampled rows
                        rmm::device_uvector<int>* remaining_groups, // 1x n_groups
                        rmm::device_uvector<int>* remaining_samples, // 1x n_sampled_rows
                        rmm::device_uvector<int>* workspace, // 1x n sampled rows
                        int* groups,
                        const int n_groups,
                        const std::vector<std::vector<int>>& fold_memberships, // each group belongs to one fold
                        const cudaStream_t stream)
  {
    // Todo: split group_fold_rng across threads
    raft::common::nvtx::range fun_scope("bootstrapping row IDs @randomforest.cuh");

    // Hash these together so they are uncorrelated
    auto random_seed = DT::fnv1a32_basis;
    random_seed      = DT::fnv1a32(random_seed, rf_params.seed);
    random_seed      = DT::fnv1a32(random_seed, tree_id);
    raft::random::Rng rng(random_seed, raft::random::GenPhilox);

    // generate the random state needed for cpu-side sampling
    auto cpu_random_seed = DT::fnv1a32(random_seed, 1);
    std::random_device rd;
    std::mt19937 group_fold_rng(rd());
    group_fold_rng.seed(cpu_random_seed);
    
    std::vector<std::vector<int>> honest_group_assignments(2);
    auto& splitting_groups = honest_group_assignments[0];
    auto& averaging_groups = honest_group_assignments[1];
    if (n_groups > 0) {
      // Special handling for groups. We don't support split ratio honesty
      const std::vector<int>* current_fold_groups;
      std::vector<int> restricted_group_ixs;
      std::vector<int> restricted_group_ixs_diff;
      int restricted_ix_size = n_groups;
      if (rf_params.minTreesPerGroupFold > 0) {
        const int current_fold = tree_id / rf_params.minTreesPerGroupFold;
        current_fold_groups = &fold_memberships[current_fold];
        restricted_group_ixs.resize(n_groups);
        std::iota(restricted_group_ixs.begin(), restricted_group_ixs.end(), 0);
        
        restricted_ix_size = n_groups - current_fold_groups->size();
        restricted_group_ixs_diff.reserve(restricted_ix_size);
        std::set_difference(restricted_group_ixs.begin(),
                            restricted_group_ixs.end(),
                            current_fold_groups->begin(),
                            current_fold_groups->end(),
                            std::inserter(restricted_group_ixs_diff, restricted_group_ixs_diff.begin()));
      }

      if (rf_params.oob_honesty) {
        const float split_ratio = 0.632; 

        if (rf_params.minTreesPerGroupFold > 0) {
          // Doing group / fold "leave-out" logic
          int honest_split_size = split_ratio * (n_groups - current_fold_groups->size());

          splitting_groups.resize(honest_split_size);
          averaging_groups.resize(restricted_ix_size - honest_split_size);

          assign_groups_to_folds(
              restricted_ix_size,
              2,
              honest_split_size,
              honest_group_assignments,
              group_fold_rng);
          
          // Replace indices with the actual groups
          for (int ix_group = 0; ix_group < honest_split_size; ix_group++) {
            honest_group_assignments[0][ix_group] = restricted_group_ixs_diff[honest_group_assignments[0][ix_group]];
          }
          
          for (int ix_group = 0; ix_group < restricted_ix_size - honest_split_size; ix_group++) {
            honest_group_assignments[1][ix_group] = restricted_group_ixs_diff[honest_group_assignments[1][ix_group]];
          }
        } else {
          // Here we're not doing folds, we're partitioning the groups directly. We're also not leaving out groups?
          // Easy enough so I'll add it, to match rforestry functionality
          int honest_split_size = std::round(split_ratio * static_cast<double>(n_groups));

          // Avoid empty set
          if (honest_split_size == n_groups) {
            honest_split_size = n_groups - 1;
          } else if (honest_split_size == 0) {
            honest_split_size = 1;
          }

          splitting_groups.resize(honest_split_size);
          averaging_groups.resize(honest_split_size);
          assign_groups_to_folds(
              n_groups, 
              2,
              honest_split_size,
              honest_group_assignments,
              group_fold_rng);
        }

        leave_groups_out_sample(remaining_groups, remaining_samples, selected_rows, workspace,
            groups, splitting_groups, n_sampled_rows, 0, stream, rng);

        leave_groups_out_sample(remaining_groups, remaining_samples, selected_rows, workspace,
            groups, averaging_groups, n_sampled_rows, n_sampled_rows, stream, rng);

        update_averaging_mask(split_row_mask, n_sampled_rows, stream);

        return n_sampled_rows; // averaging sample count

      } else if (rf_params.minTreesPerGroupFold > 0) {
        // Just don't use samples from the current fold for splitting. No averaging.
        leave_groups_out_sample(remaining_groups, remaining_samples, selected_rows, workspace,
            groups, restricted_group_ixs_diff, n_sampled_rows, 0, stream, rng);
        return 0; // no averaging samples
      }
    }

    if (rf_params.bootstrap) {
      // Use bootstrapped sample set
      rng.uniformInt<int>(selected_rows->data(), n_sampled_rows, 0, n_rows, stream);
    } else {
      // Use all the samples from the dataset
      thrust::sequence(thrust::cuda::par.on(stream), selected_rows->begin(), selected_rows->end());
    }
    size_t num_avg_samples = 0;
    
    if (rf_params.oob_honesty and rf_params.bootstrap) {
      selected_rows->resize(n_sampled_rows, stream);
      // honesty doesn't make sense without bootstrapping -- all the obs were otherwise selected
      num_avg_samples = n_sampled_rows;
      assert(rf_params.bootstrap);
    
      // We'll have n_rows samples for splitting. 
      // Need to sort the selected rows to be able to use thrust set difference
      thrust::sort(thrust::cuda::par.on(stream), selected_rows->begin(), selected_rows->end());

      // Get the set of observations that are not used for split
      auto iter_end = thrust::set_difference(
          thrust::cuda::par.on(stream),
          thrust::make_counting_iterator<int>(0),
          thrust::make_counting_iterator<int>(n_sampled_rows),
          selected_rows->begin(),
          selected_rows->end(),
          remaining_samples->begin());
      
      // Now remaining_samples is the observations available for the averaging set
      size_t num_remaining_samples = iter_end - remaining_samples->begin();
      remaining_samples->resize(num_remaining_samples, stream);

      // Get the avg selected rows either as the remaining data, or bootstrapped again
      selected_rows->resize(n_sampled_rows * 2, stream);
      if (rf_params.double_bootstrap) {
        sample_rows_from_remaining_rows(remaining_samples, selected_rows, workspace, n_sampled_rows, n_sampled_rows, stream, rng);
      } else {
        thrust::copy(thrust::cuda::par.on(stream), remaining_samples->begin(), remaining_samples->end(), selected_rows->begin() + n_sampled_rows);
      }
      
      update_averaging_mask(split_row_mask, n_sampled_rows, stream);
    }

    return num_avg_samples;
  }

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols, bool predict) const
  {
    if (predict) {
      ASSERT(predictions != nullptr, "Error! User has not allocated memory for predictions.");
    }
    ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

    bool input_is_dev_ptr = DT::is_dev_ptr(input);
    bool preds_is_dev_ptr = DT::is_dev_ptr(predictions);

    if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
      ASSERT(false,
             "RF Error: Expected both input and labels/predictions to be GPU "
             "pointers");
    }
  }

 public:
  /**
   * @brief Construct RandomForest object.
   * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
   * @param[in] cfg_rf_type: Task type: 0 for classification, 1 for regression
   */
  RandomForest(RF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION)
    : rf_params(cfg_rf_params), rf_type(cfg_rf_type){};

  /**
   * @brief Build (i.e., fit, train) random forest for input data.
   * @param[in] user_handle: raft::handle_t
   * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
   *   excluding labels. Device pointer.
   * @param[in] n_rows: number of training data samples.
   * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
   * @param[in] labels: 1D array of target predictions/labels. Device Pointer.
            For classification task, only labels of type int are supported.
              Assumption: labels were preprocessed to map to ascending numbers from 0;
              needed for current gini impl in decision tree
            For regression task, the labels (predictions) can be float or double data type.
  * @param[in] n_unique_labels: (meaningful only for classification) #unique label values (known
  during preprocessing)
  * @param[in] forest: CPU point to RandomForestMetaData struct.
  */
  void fit(const raft::handle_t& user_handle,
           const T* input,
           int n_rows,
           int n_cols,
           L* labels,
           int n_unique_labels,
           RandomForestMetaData<T, L>*& forest)
  {
    raft::common::nvtx::range fun_scope("RandomForest::fit @randomforest.cuh");
    this->error_checking(input, labels, n_rows, n_cols, false);
    const raft::handle_t& handle = user_handle;
    int n_sampled_rows           = 0;
    if (this->rf_params.bootstrap) {
      n_sampled_rows = std::round(this->rf_params.max_samples * n_rows);
    } else {
      if (this->rf_params.max_samples != 1.0) {
        CUML_LOG_WARN(
          "If bootstrap sampling is disabled, max_samples value is ignored and "
          "whole dataset is used for building each tree");
        this->rf_params.max_samples = 1.0;
      }
      n_sampled_rows = n_rows;
    }
    int n_streams = this->rf_params.n_streams;
    ASSERT(static_cast<std::size_t>(n_streams) <= handle.get_stream_pool_size(),
           "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%lu)",
           n_streams,
           handle.get_stream_pool_size());

    int n_trees = this->rf_params.n_trees;
    // Compute the number of trees. This might change based on the "groups" logic
    std::vector<std::vector<int>> foldMemberships;
    const int foldGroupSize = this->rf_params.foldGroupSize;
    const int minTreesPerGroupFold = this->rf_params.minTreesPerGroupFold;
    std::unique_ptr<rmm::device_uvector<int>> groups;
    int n_groups = 0;

    if (this->rf_params.group_col_idx >= 0) {
      // Here we'll going to do a unique, then build a vector of indices into the unique vector
      cudaStream_t stream = handle.get_stream_from_stream_pool(0);
      rmm::device_uvector<T> input_groups(n_rows, stream);
      rmm::device_uvector<T> input_groups_unique(n_rows, stream);
      groups = std::make_unique<rmm::device_uvector<int>>(n_rows, stream);
      cudaMemcpyAsync(input_groups.data(), 
          input + n_rows * this->rf_params.group_col_idx, 
          n_rows * sizeof(T), cudaMemcpyDefault, stream);
      cudaMemcpyAsync(input_groups_unique.data(), 
          input + n_rows * this->rf_params.group_col_idx, 
          n_rows * sizeof(T), cudaMemcpyDefault, stream);
      // Sadly we have to sort the entire array for unique to work. Is there 
      // a way to just unique the unsorted array?
      thrust::sort(thrust::cuda::par.on(stream),
          input_groups_unique.data(),
          input_groups_unique.data() + n_rows);
      T* new_end = thrust::unique(thrust::cuda::par.on(stream),
          input_groups_unique.data(),
          input_groups_unique.data() + n_rows);
      // Now we'll have n_groups and can use some iterator to find the values for each group
      n_groups = new_end - input_groups_unique.data(); 

      UniqueTransformFunctor transform_fn{input_groups_unique.data(), n_groups};
      thrust::transform(
          thrust::cuda::par.on(stream),
          input_groups.data(),
          input_groups.data() + n_rows,
          groups->data(),
          transform_fn);
    }

    if (minTreesPerGroupFold > 0 and n_groups > 0) {
      // Use a separate RNG and the std functions for group membership. 
      std::random_device rd;
      std::mt19937 group_fold_rng(rd());
      auto random_seed = DT::fnv1a32_basis;
      random_seed      = DT::fnv1a32(random_seed, this->rf_params.seed);
      random_seed      = DT::fnv1a32(random_seed, std::numeric_limits<int>::max());
      group_fold_rng.seed(random_seed);

      int n_folds = (n_groups + foldGroupSize - 1) / foldGroupSize;
      n_trees = n_folds * minTreesPerGroupFold;
      forest->trees.resize(n_trees);
      // TODO: Why are there 2 separate rf_params structs? 
      // I think it would be best for this class to hold a pointer to the one that's passed in to fit.
      forest->rf_params.n_trees = n_trees;
      this->rf_params.n_trees = n_trees;
      
      foldMemberships = std::vector<std::vector<int>>(n_folds, std::vector<int>(minTreesPerGroupFold));

      //assign group to fold
      assign_groups_to_folds(n_groups, n_folds, foldGroupSize, foldMemberships, group_fold_rng);
    }

    // computing the quantiles: last two return values are shared pointers to device memory
    // encapsulated by quantiles struct
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, input, this->rf_params.tree_params.max_n_bins, n_rows, n_cols);

    // n_streams should not be less than n_trees
    if (n_trees < n_streams) n_streams = n_trees;

    // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
    // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device
    // ptr.
    // Use a deque instead of vector because it can be used on objects with a deleted copy
    // constructor
    std::deque<rmm::device_uvector<int>> selected_rows;
    std::deque<rmm::device_uvector<bool>> split_row_masks;
    std::deque<rmm::device_uvector<int>> workspaces;
    std::deque<rmm::device_uvector<int>> remaining_groups_vec;
    std::deque<rmm::device_uvector<int>> remaining_samples_vec;

    const bool use_extra_vecs = this->rf_params.oob_honesty or this->rf_params.minTreesPerGroupFold > 0;
    size_t max_sample_row_size = this->rf_params.oob_honesty ? n_sampled_rows * 2 : n_sampled_rows;
    for (int i = 0; i < n_streams; i++) {
      auto s = handle.get_stream_from_stream_pool(i);
      selected_rows.emplace_back(max_sample_row_size, s);
      if (use_extra_vecs) {
        split_row_masks.emplace_back(max_sample_row_size, s);
        workspaces.emplace_back(n_rows, s);
        remaining_samples_vec.emplace_back(n_rows, s);
      }
      if (n_groups > 0) {
        remaining_groups_vec.emplace_back(n_groups, s);
      }
    }

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < n_trees; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);
      
      rmm::device_uvector<int>* remaining_samples = use_extra_vecs ? &remaining_samples_vec[stream_id] : nullptr;
      rmm::device_uvector<int>* workspace = use_extra_vecs ? &workspaces[stream_id] : nullptr;
      rmm::device_uvector<int>* remaining_groups = n_groups > 0 ? &remaining_groups_vec[stream_id] : nullptr;
      rmm::device_uvector<bool>* split_row_mask = use_extra_vecs ? &split_row_masks[stream_id] : nullptr;
      int* this_groups = n_groups > 0 ? groups->data() : nullptr;
      auto n_avg_samples = this->get_row_sample(
          i, n_rows, n_sampled_rows, 
          &selected_rows[stream_id],
          split_row_mask,
          remaining_groups,
          remaining_samples,
          workspace,
          this_groups,
          n_groups,
          foldMemberships,
          s);

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled for tree's bootstrap sample.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */
      if (this->rf_params.oob_honesty) {
        forest->trees[i] = DT::DecisionTree::fit<true>(handle,
                                                       s,
                                                       input,
                                                       n_cols,
                                                       n_rows,
                                                       labels,
                                                       &selected_rows[stream_id],
                                                       split_row_masks[stream_id].data(),
                                                       n_avg_samples,
                                                       n_unique_labels,
                                                       this->rf_params.tree_params,
                                                       this->rf_params.seed,
                                                       quantiles,
                                                       i);
      } else {
        forest->trees[i] = DT::DecisionTree::fit<false>(handle,
                                                        s,
                                                        input,
                                                        n_cols,
                                                        n_rows,
                                                        labels,
                                                        &selected_rows[stream_id],
                                                        nullptr,
                                                        n_avg_samples,
                                                        n_unique_labels,
                                                        this->rf_params.tree_params,
                                                        this->rf_params.seed,
                                                        quantiles,
                                                        i);
      }
    }

    // Cleanup
    handle.sync_stream_pool();
    handle.sync_stream();
  }

  /**
   * @brief Predict target feature for input data
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   */
  void predict(const raft::handle_t& user_handle,
               const T* input,
               int n_rows,
               int n_cols,
               L* predictions,
               const RandomForestMetaData<T, L>* forest,
               int verbosity) const
  {
    ML::Logger::get().setLevel(verbosity);
    this->error_checking(input, predictions, n_rows, n_cols, true);
    std::vector<L> h_predictions(n_rows);
    cudaStream_t stream = user_handle.get_stream();

    std::vector<T> h_input(std::size_t(n_rows) * n_cols);
    raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, stream);
    user_handle.sync_stream(stream);

    int row_size = n_cols;

    ML::PatternSetter _("%v");
    for (int row_id = 0; row_id < n_rows; row_id++) {
      std::vector<T> row_prediction(forest->trees[0]->num_outputs);
      for (int i = 0; i < this->rf_params.n_trees; i++) {
        DT::DecisionTree::predict(user_handle,
                                  *forest->trees[i],
                                  &h_input[row_id * row_size],
                                  1,
                                  n_cols,
                                  row_prediction.data(),
                                  forest->trees[i]->num_outputs,
                                  verbosity);
      }
      for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
        row_prediction[k] /= this->rf_params.n_trees;
      }
      if (rf_type == RF_type::CLASSIFICATION) {  // classification task: use 'majority' prediction
        L best_class = 0;
        T best_prob  = 0.0;
        for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
          if (row_prediction[k] > best_prob) {
            best_class = k;
            best_prob  = row_prediction[k];
          }
        }

        h_predictions[row_id] = best_class;
      } else {
        h_predictions[row_id] = row_prediction[0];
      }
    }

    raft::update_device(predictions, h_predictions.data(), n_rows, stream);
    user_handle.sync_stream(stream);
  }

  /**
   * @brief Predict target feature for input data and score against ref_labels.
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
   * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   * @param[in] rf_type: task type: 0 for classification, 1 for regression
   */
  static RF_metrics score(const raft::handle_t& user_handle,
                          const L* ref_labels,
                          int n_rows,
                          const L* predictions,
                          int verbosity,
                          int rf_type = RF_type::CLASSIFICATION)
  {
    ML::Logger::get().setLevel(verbosity);
    cudaStream_t stream = user_handle.get_stream();
    RF_metrics stats;
    if (rf_type == RF_type::CLASSIFICATION) {  // task classifiation: get classification metrics
      float accuracy = raft::stats::accuracy(predictions, ref_labels, n_rows, stream);
      stats          = set_rf_metrics_classification(accuracy);
      if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);

      /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
        For non binary classification problems (i.e., one target and  > 2 labels), need avg.
        for each of these metrics */
    } else {  // regression task: get regression metrics
      double mean_abs_error, mean_squared_error, median_abs_error;
      raft::stats::regression_metrics(predictions,
                                      ref_labels,
                                      n_rows,
                                      stream,
                                      mean_abs_error,
                                      mean_squared_error,
                                      median_abs_error);
      stats = set_rf_metrics_regression(mean_abs_error, mean_squared_error, median_abs_error);
      if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);
    }

    return stats;
  }
};

// class specializations
template class RandomForest<float, int>;
template class RandomForest<float, float>;
template class RandomForest<double, int>;
template class RandomForest<double, double>;

}  // End namespace ML
