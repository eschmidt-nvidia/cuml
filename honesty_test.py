#! /usr/bin/env python

import pandas as pd
import numpy as np
import cuml

from cuml.ensemble import RandomForestClassifier as RFC
from cuml.ensemble import RandomForestRegressor as RFR

import time


input = pd.read_parquet("/home/scratch.eschmidt_sw/gotvBIG.parquet")

# input = input.iloc[:50000, :]

# cuML doesn't handle string inputs
input["vh_stratum"] = input["vh_stratum"].replace({"below": -1.0, "average":0.0, "above":1.0, "":np.nan}).astype(float)

states = input['state'].unique()
print(f"num unique states {len(states)}")
states_map = {}
for ix_state,state in enumerate(states):
  states_map[state] = float(ix_state)

input["state"] = input["state"].map(states_map)

# state fields are redundant with state value
# state_fields = ["d_st_AK","d_st_AR","d_st_AZ","d_st_CO","d_st_FL","d_st_GA","d_st_IA","d_st_KS","d_st_KY","d_st_LA","d_st_ME","d_st_MI","d_st_NC","d_st_NH","d_st_SD","d_st_TX","d_st_WI"]
# input = input.drop(labels=state_fields, axis=1)

input = input.astype('float32')
input = input.dropna()

# Choose how many index include for random selection
num_rows = input.shape[0]
ix_train = np.random.choice(num_rows, replace=False, size=int(num_rows*0.7))
ix_test = np.setdiff1d(np.arange(num_rows), ix_train)

x = input.drop(labels=["voted14"], axis=1)
y = input["voted14"]

x_train = x.iloc[ix_train, :]
y_train = y.iloc[ix_train]

x_test = x.iloc[ix_test, :]
y_test = y.iloc[ix_test]

n_trees = 100

# Start group call -- note we're not using groups to specify OOB predictions
# Note, here we specify a column index to use for groups. Then the fit() function
# will use the GPU to compute unique group ids for every sample. 
group_col_idx = x.columns.get_loc("state")
random_forest_regress = RFR(n_estimators=n_trees, oob_honesty=True, split_criterion=2, 
    random_state=42, minTreesPerGroupFold=5, foldGroupSize=1, group_col_idx=group_col_idx)
start = time.time()
trainedRFR = random_forest_regress.fit(x_train, y_train)
end = time.time()
pred_test_regress = trainedRFR.predict(x_test)
mse = cuml.metrics.mean_squared_error(y_test, pred_test_regress)
print(f"Group Honesty {mse} time {end-start}")

random_forest_regress = RFR(n_estimators=n_trees, split_criterion=2, random_state=42)  
start = time.time()
trainedRFR = random_forest_regress.fit(x_train, y_train)
end = time.time()
pred_test_regress = trainedRFR.predict(x_test)
mse = cuml.metrics.mean_squared_error(y_test, pred_test_regress)
print(f"No honesty {mse} time {end-start}")

random_forest_regress = RFR(n_estimators=n_trees, oob_honesty=True, split_criterion=2, random_state=42)
start = time.time()
trainedRFR = random_forest_regress.fit(x_train, y_train)
end = time.time()
pred_test_regress = trainedRFR.predict(x_test)
mse = cuml.metrics.mean_squared_error(y_test, pred_test_regress)
print(f"Honesty {mse} time {end-start}")

