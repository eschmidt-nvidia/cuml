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
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace DT {

struct CountBin {
  int x;
  CountBin(CountBin const&) = default;
  HDI CountBin(int x_) : x(x_) {}
  HDI CountBin() : x(0) {}

  DI static void IncrementHistogram(CountBin* hist, int n_bins, int b, int label, bool /*is_split_row*/, bool /*split_op*/)
  {
    auto offset = label * n_bins + b;
    CountBin::AtomicAdd(hist + offset, {1});
  }
  DI static void AtomicAdd(CountBin* address, CountBin val) { atomicAdd(&address->x, val.x); }
  HDI CountBin& operator+=(const CountBin& b)
  {
    x += b.x;
    return *this;
  }
  HDI CountBin operator+(CountBin b) const
  {
    b += *this;
    return b;
  }
};

struct HonestCountBin : CountBin {
  int x_averaging;

  HonestCountBin(HonestCountBin const&) = default;
  HDI HonestCountBin(int x_train, int x_averaging) : CountBin(x_train), x_averaging(x_averaging) {}
  HDI HonestCountBin() : CountBin(), x_averaging(0) {}

  DI static void IncrementHistogram(HonestCountBin* hist, int n_bins, int b, int label, bool is_split_row, bool /*is_split_op*/)
  {
    auto offset = label * n_bins + b;    
    if (is_split_row) {
      atomicAdd(&(hist + offset)->x, {1});
    } else {
      atomicAdd(&(hist + offset)->x_averaging, {1});
    }
  }

  DI static void AtomicAdd(HonestCountBin* address, HonestCountBin val) 
  { 
    atomicAdd(&address->x, val.x); 
    atomicAdd(&address->x_averaging, val.x_averaging); 
  }
  HDI HonestCountBin& operator+=(const HonestCountBin& b)
  {
    CountBin::operator+=(b);
    x_averaging += b.x_averaging;
    return *this;
  }
  HDI HonestCountBin operator+(HonestCountBin b) const
  {    
    b += *this;
    return b;
  }
};

struct AggregateBin {
  double label_sum;
  int count;

  AggregateBin(AggregateBin const&) = default;
  HDI AggregateBin() : label_sum(0.0), count(0) {}
  HDI AggregateBin(double label_sum, int count) : label_sum(label_sum), count(count) {}

  DI static void IncrementHistogram(AggregateBin* hist, int n_bins, int b, double label, bool, bool)
  {
    AggregateBin::AtomicAdd(hist + b, {label, 1});
  }
  DI static void AtomicAdd(AggregateBin* address, AggregateBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->count, val.count);
  }
  HDI AggregateBin& operator+=(const AggregateBin& b)
  {
    label_sum += b.label_sum;
    count += b.count;
    return *this;
  }
  HDI AggregateBin operator+(AggregateBin b) const
  {
    b += *this;
    return b;
  }
};

struct HonestAggregateBin : AggregateBin {
  int count_averaging;

  HonestAggregateBin(HonestAggregateBin const&) = default;
  HDI HonestAggregateBin() : AggregateBin(), count_averaging(0) {}
  HDI HonestAggregateBin(double label_sum, int count, int count_averaging) 
    : AggregateBin(label_sum, count), count_averaging(count_averaging) 
  {}

  DI static void IncrementHistogram(
      HonestAggregateBin* hist, int n_bins, int b, double label, bool is_split_row, bool is_split_op)
  {
    HonestAggregateBin* address = hist + b;
    const int train_incr = static_cast<int>(is_split_row);
    const int avg_incr = 1 - train_incr;
    
    // Either split row and split op, or neither. Otherwise no increment of the label.
    const double label_incr = not (is_split_row xor is_split_op) ? label : 0.0; 
    HonestAggregateBin::AtomicAdd(address, {label_incr, train_incr, avg_incr});
  }
  DI static void AtomicAdd(HonestAggregateBin* address, HonestAggregateBin val)
  {
    atomicAdd(&address->label_sum, val.label_sum);
    atomicAdd(&address->count, val.count);
    atomicAdd(&address->count_averaging, val.count_averaging);
  }
  HDI HonestAggregateBin& operator+=(const HonestAggregateBin& b)
  {
    AggregateBin::operator+=(b);
    count_averaging += b.count_averaging;
    return *this;
  }
  HDI HonestAggregateBin operator+(HonestAggregateBin b) const
  {
    b += *this;
    return b;
  }
};

}  // namespace DT
}  // namespace ML