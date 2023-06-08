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

#include "dataset.h"
#include "split.cuh"
#include <cub/cub.cuh>
#include <limits>

namespace ML {
namespace DT {

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class GiniObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  static const bool oob_honesty = oob_honesty_;

 private:
  IdxT nclasses;
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  using BinT = std::conditional_t<oob_honesty, HonestCountBin, CountBin>;
  GiniObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : nclasses(nclasses), 
      min_samples_leaf_splitting(min_samples_leaf_splitting), 
      min_samples_leaf_averaging(min_samples_leaf_averaging)
  {}

  DI IdxT NumClasses() const { return nclasses; }

  /**
   * @brief compute the gini impurity reduction for each split
   */
  HDI DataT GainPerSplit(
      BinT* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT trainingLen, IdxT nRightAveraging)
  {
    constexpr DataT One  = DataT(1.0);
    auto invLen          = One / trainingLen;
    auto invLeft         = One / nLeftSplitting;
    auto nRightSplitting = trainingLen - nLeftSplitting;
    auto invRight        = One / nRightSplitting;
    auto gain            = DataT(0.0);

    // if there aren't enough samples in this split, don't bother!
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();

    for (IdxT j = 0; j < nclasses; ++j) {
      int val_i   = 0;
      auto lval_i = hist[n_bins * j + i].x;
      auto lval   = DataT(lval_i);
      gain += lval * invLeft * lval * invLen;

      val_i += lval_i;
      auto total_sum = hist[n_bins * j + n_bins - 1].x;
      auto rval_i    = total_sum - lval_i;
      auto rval      = DataT(rval_i);
      gain += rval * invRight * rval * invLen;

      val_i += rval_i;
      auto val = DataT(val_i) * invLen;
      gain -= val * val;
    }

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins)
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      IdxT nLeftSplitting = 0;
      IdxT nLeftAveraging = 0;
      IdxT nRightAveraging = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeftSplitting += shist[n_bins * j + i].x;
        if constexpr (oob_honesty) {
          nLeftAveraging += shist[n_bins * j + i].x_averaging;
        }
      }

      if constexpr (oob_honesty) {
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
        squantiles[i], col,
        GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging),
        nLeftAveraging + nLeftSplitting, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    int total = 0;
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        total += shist[i].x_averaging;
      } else {
        total += shist[i].x;
      }
    }
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = DataT(shist[i].x_averaging) / total;
      } else {
        out[i] = DataT(shist[i].x) / total;
      }
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class EntropyObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  static const bool oob_honesty = oob_honesty_;
  
 private:
  IdxT nclasses;
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  using BinT = std::conditional_t<oob_honesty, HonestCountBin, CountBin>;

  EntropyObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : nclasses(nclasses), 
      min_samples_leaf_splitting(min_samples_leaf_splitting), 
      min_samples_leaf_averaging(min_samples_leaf_averaging)
  {}

  DI IdxT NumClasses() const { return nclasses; }

  /**
   * @brief compute the Entropy (or information gain) for each split
   */
  HDI DataT GainPerSplit(
      BinT* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT lenTraining, IdxT nRightAveraging)
  {
    const auto nRightSplitting = lenTraining - nLeftSplitting;

    // if there aren't enough samples in this split, don't bother!
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();
        
    auto gain{DataT(0.0)};
    auto invLeft{DataT(1.0) / nLeftSplitting};
    auto invRight{DataT(1.0) / nRightSplitting};
    auto invLen{DataT(1.0) / lenTraining};
    for (IdxT c = 0; c < nclasses; ++c) {
      int val_i   = 0;
      auto lval_i = hist[n_bins * c + i].x;
      if (lval_i != 0) {
        auto lval = DataT(lval_i);
        gain += raft::myLog(lval * invLeft) / raft::myLog(DataT(2)) * lval * invLen;
      }

      val_i += lval_i;
      auto total_sum = hist[n_bins * c + n_bins - 1].x;
      auto rval_i    = total_sum - lval_i;
      if (rval_i != 0) {
        auto rval = DataT(rval_i);
        gain += raft::myLog(rval * invRight) / raft::myLog(DataT(2)) * rval * invLen;
      }

      val_i += rval_i;
      if (val_i != 0) {
        auto val = DataT(val_i) * invLen;
        gain -= val * raft::myLog(val) / raft::myLog(DataT(2));
      }
    }

    return gain;
  }

  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins)
  {    
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      IdxT nLeftSplitting = 0;
      IdxT nLeftAveraging = 0;
      IdxT nRightAveraging = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeftSplitting += shist[n_bins * j + i].x;
        if constexpr (oob_honesty) {
          nLeftAveraging += shist[n_bins * j + i].x_averaging;
        }
      }

      if constexpr (oob_honesty) {
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
        squantiles[i], col,
        GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging),
        nLeftAveraging + nLeftSplitting, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }
  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    // Output probability
    int total = 0;
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        total += shist[i].x_averaging;
      } else {
        total += shist[i].x;
      }
    }
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = DataT(shist[i].x_averaging) / total;
      } else {
        out[i] = DataT(shist[i].x) / total;

      }
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class MSEObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  static const bool oob_honesty = oob_honesty_;

  using BinT = std::conditional_t<oob_honesty, HonestAggregateBin, AggregateBin>;

 private:
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  HDI MSEObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : min_samples_leaf_splitting(min_samples_leaf_splitting),
      min_samples_leaf_averaging(min_samples_leaf_averaging)
  {}

  /**
   * @brief compute the Mean squared error impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy mean squared error reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       mean-squared errors.
   */
  HDI DataT GainPerSplit(
      BinT const* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT trainingLen, IdxT nRightAveraging) const
  {
    auto gain{DataT(0)};
    IdxT nRightSplitting{trainingLen - nLeftSplitting};
    auto invLen = DataT(1.0) / trainingLen;
    
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();
    
    auto label_sum        = hist[n_bins - 1].label_sum;
    DataT parent_obj      = -label_sum * label_sum * invLen;
    DataT left_obj        = -(hist[i].label_sum * hist[i].label_sum) / nLeftSplitting;
    DataT right_label_sum = hist[i].label_sum - label_sum;
    DataT right_obj       = -(right_label_sum * right_label_sum) / nRightSplitting;
    gain                  = parent_obj - (left_obj + right_obj);
    gain *= DataT(0.5) * invLen;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeftSplitting = shist[i].count;
      
      int nLeftAveraging = 0;
      int nRightAveraging = 0;
      if constexpr (oob_honesty) {
        nLeftAveraging = shist[i].count_averaging;
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
          squantiles[i], col, 
          GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging), 
          nLeftSplitting + nLeftAveraging, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = shist[i].label_sum / shist[i].count_averaging;
      } else {
        out[i] = shist[i].label_sum / shist[i].count;
      }
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class PoissonObjectiveFunction {
 public:
  using DataT  = DataT_;
  using LabelT = LabelT_;
  using IdxT   = IdxT_;
  static const bool oob_honesty = oob_honesty_;

  using BinT = std::conditional_t<oob_honesty, HonestAggregateBin, AggregateBin>;

 private:
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();

  HDI PoissonObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : min_samples_leaf_splitting(min_samples_leaf_splitting),
      min_samples_leaf_averaging(min_samples_leaf_averaging)
  {}

  /**
   * @brief compute the poisson impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy poisson half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       poisson half deviances.
   */
  HDI DataT GainPerSplit(
      BinT const* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT trainingLen, IdxT nRightAveraging) const
  {
    // get the lens'
    IdxT nRightSplitting  = trainingLen - nLeftSplitting;
    auto invLen = DataT(1) / trainingLen;

    // if there aren't enough samples in this split, don't bother!
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = -label_sum * raft::myLog(label_sum * invLen);
    DataT left_obj   = -left_label_sum * raft::myLog(left_label_sum / nLeftSplitting);
    DataT right_obj  = -right_label_sum * raft::myLog(right_label_sum / nRightSplitting);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeftSplitting = shist[i].count;
      
      int nLeftAveraging = 0;
      int nRightAveraging = 0;
      if constexpr (oob_honesty) {
        nLeftAveraging = shist[i].count_averaging;
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
          squantiles[i], col,
          GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging),
          nLeftSplitting + nLeftAveraging, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = shist[i].label_sum / shist[i].count_averaging;
      } else {
        out[i] = shist[i].label_sum / shist[i].count;
      }
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class GammaObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  static const bool oob_honesty = oob_honesty_;
  
  
  using BinT = std::conditional_t<oob_honesty, HonestAggregateBin, AggregateBin>;
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();
 private:
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  HDI GammaObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : min_samples_leaf_splitting{min_samples_leaf_splitting},
      min_samples_leaf_averaging{min_samples_leaf_averaging}
  {}

  /**
   * @brief compute the gamma impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy gamma half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       gamma half deviances.
   */
  HDI DataT GainPerSplit(
      BinT const* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT trainingLen, IdxT nRightAveraging) const
  {
    IdxT nRightSplitting = trainingLen - nLeftSplitting;

    // if there aren't enough samples in this split, don't bother!
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();

    auto invLen = DataT(1) / trainingLen;
    DataT label_sum       = hist[n_bins - 1].label_sum;
    DataT left_label_sum  = (hist[i].label_sum);
    DataT right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = trainingLen * raft::myLog(label_sum * invLen);
    DataT left_obj   = nLeftSplitting * raft::myLog(left_label_sum / nLeftSplitting);
    DataT right_obj  = nRightSplitting * raft::myLog(right_label_sum / nRightSplitting);
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain * invLen;

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeftSplitting = shist[i].count;
      
      int nLeftAveraging = 0;
      int nRightAveraging = 0;
      if constexpr (oob_honesty) {
        nLeftAveraging = shist[i].count_averaging;
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
          squantiles[i], col,
          GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging),
          nLeftSplitting + nLeftAveraging, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = shist[i].label_sum / shist[i].count_averaging;
      } else {
        out[i] = shist[i].label_sum / shist[i].count;
      }
    }
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_, bool oob_honesty_=false>
class InverseGaussianObjectiveFunction {
 public:
  using DataT                = DataT_;
  using LabelT               = LabelT_;
  using IdxT                 = IdxT_;
  
  static const bool oob_honesty = oob_honesty_;
  
  using BinT = std::conditional_t<oob_honesty, HonestAggregateBin, AggregateBin>;
  
  static constexpr auto eps_ = 10 * std::numeric_limits<DataT>::epsilon();
 private:
  IdxT min_samples_leaf_splitting;
  IdxT min_samples_leaf_averaging;

 public:
  HDI InverseGaussianObjectiveFunction(IdxT nclasses, IdxT min_samples_leaf_splitting, IdxT min_samples_leaf_averaging)
    : min_samples_leaf_splitting{min_samples_leaf_splitting},
      min_samples_leaf_averaging{min_samples_leaf_averaging}
  {}

  /**
   * @brief compute the inverse gaussian impurity reduction (or purity gain) for each split
   *
   * @note This method is used to speed up the search for the best split
   *       by calculating the gain using a proxy inverse gaussian half deviance reduction.
   *       It is a proxy quantity such that the split that maximizes this value
   *       also maximizes the impurity improvement. It neglects all constant terms
   *       of the impurity decrease for a given split.
   *       The Gain is the difference in the proxy impurities of the parent and the
   *       weighted sum of impurities of its children
   *       and is mathematically equivalent to the respective differences of
   *       inverse gaussian deviances.
   */
  HDI DataT GainPerSplit(
      const BinT* hist, IdxT i, IdxT n_bins, IdxT nLeftSplitting,
      IdxT nLeftAveraging, IdxT trainingLen, IdxT nRightAveraging) const
  {
    // get the lens'
    IdxT nRightSplitting  = trainingLen - nLeftSplitting;
    
    // if there aren't enough samples in this split, don't bother!
    if constexpr (oob_honesty) {
      if (nLeftAveraging < min_samples_leaf_averaging || nRightAveraging < min_samples_leaf_averaging)
        return -std::numeric_limits<DataT>::max();
    }

    if (nLeftSplitting < min_samples_leaf_splitting || nRightSplitting < min_samples_leaf_splitting)
      return -std::numeric_limits<DataT>::max();

    auto label_sum       = hist[n_bins - 1].label_sum;
    auto left_label_sum  = (hist[i].label_sum);
    auto right_label_sum = (hist[n_bins - 1].label_sum - hist[i].label_sum);

    // label sum cannot be non-positive
    if (label_sum < eps_ || left_label_sum < eps_ || right_label_sum < eps_)
      return -std::numeric_limits<DataT>::max();

    // compute the gain to be
    DataT parent_obj = -DataT(trainingLen) * DataT(trainingLen) / label_sum;
    DataT left_obj   = -DataT(nLeftSplitting) * DataT(nLeftSplitting) / left_label_sum;
    DataT right_obj  = -DataT(nRightSplitting) * DataT(nRightSplitting) / right_label_sum;
    DataT gain       = parent_obj - (left_obj + right_obj);
    gain             = gain / (2 * trainingLen);

    return gain;
  }

  DI Split<DataT, IdxT> Gain(
    BinT const* shist, DataT const* squantiles, IdxT col, IdxT len, IdxT avg_len, IdxT n_bins) const
  {
    Split<DataT, IdxT> sp;
    for (IdxT i = threadIdx.x; i < n_bins; i += blockDim.x) {
      auto nLeftSplitting = shist[i].count;
      
      int nLeftAveraging = 0;
      int nRightAveraging = 0;
      if constexpr (oob_honesty) {
        nLeftAveraging = shist[i].count_averaging;
        nRightAveraging = avg_len - nLeftAveraging;
      }

      sp.update({
          squantiles[i], col,
          GainPerSplit(shist, i, n_bins, nLeftSplitting, nLeftAveraging, len - avg_len, nRightAveraging),
          nLeftSplitting + nLeftAveraging, nLeftAveraging, nRightAveraging});
    }
    return sp;
  }

  DI IdxT NumClasses() const { return 1; }

  static DI void SetLeafVector(BinT const* shist, int nclasses, DataT* out)
  {
    for (int i = 0; i < nclasses; i++) {
      if constexpr (oob_honesty) {
        out[i] = shist[i].label_sum / shist[i].count_averaging;
      } else {
        out[i] = shist[i].label_sum / shist[i].count;
      }
    }
  }
};
}  // end namespace DT
}  // end namespace ML
