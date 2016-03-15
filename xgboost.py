
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import math

# Load the data
train_path = "/Users/mac/Desktop/termtwo/DS/project/train.csv"
test_path = "/Users/mac/Desktop/termtwo/DS/project/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare the inputs for the model
train_X = train_df.ix[:, 1:-1].as_matrix()
test_X = test_df.ix[:, 1:].as_matrix()

# Log(X+1) scale if useful
# train_X = np.log(train_X + 1)

# Encode the classes
le = LabelEncoder()
train_y = le.fit_transform(train_df['target'])

# Make the right input format for data
dtrain = xgb.DMatrix(train_X, label = train_y)
dtest = xgb.DMatrix(test_X)

# Tuning parameters 
# Step size shrinkage used in update to prevents overfitting
etas = [0.01]

# Maximum depth of a tree
depths = [6, 8]

# Subsample ratio of the training instance
subs = [0.6, 0.7]

# Specify the learning task and the corresponding learning objective.
objective = "multi:softmax"

# Evaluation method
eval_metric = "mlogloss"

# Random state
random_seed = 123

# The number of rounds for boosting
def nround(eta, depth):
    return int(1000 / (eta * math.sqrt(depth)))

# List for appending the result of each loop
dfs = []
for eta in etas:
    for depth in depths:
        for sub in subs:
            num_round = nround(eta, depth)
            param = {'max_depth': depth, 'eta': eta, 'subsample': sub,
                     'objective': objective, 'eval_metric': eval_metric, 'num_class': 9, 'silent': 1}
            cv = xgb.cv(param, dtrain, num_boost_round = num_round, nfold = 5,
                        as_pandas = True, seed = random_seed, show_progress = True,
                        early_stopping_rounds = 100, maximize = False)
            optimal = (cv['test-mlogloss-mean'] + cv['test-mlogloss-std']).idxmin()
            best = cv.ix[optimal]
            # Result of one loop
            out = pd.DataFrame(data = {'test-mlogloss-mean': best['test-mlogloss-mean'] + best['test-mlogloss-std'],
                                       'train-mlogloss-mean': best['train-mlogloss-mean'] + best['train-mlogloss-std'],
                                       'eta': eta, 'max_depth': depth, 'subsample': sub, 'num_round': num_round,
                                       'best_round': optimal + 1}, index = [1])
            dfs.append(out)
            print out

# Save all tuning result
result = pd.concat(dfs)
result.to_csv("/Users/mac/Desktop/termtwo/DS/project/xgboost_raw2.csv")

# Do the prediction based on the model with lowest logloss in validation set, 
# output the probability of each classes.
param['objective'] = 'multi:softprob'
eval_metric = "mlogloss"
param = {'max_depth': 8, 'eta': 0.01, 'subsample': 0.6,
         'objective': objective, 'eval_metric': eval_metric, 'num_class': 9, 'silent': 0}

# Since the turned out number of rounds is based on 4/5 dataset
# the best number of rounds for the whole dataset should be larger.
# I would multiply the epochs with 5/4 to get the more accurate number of rounds.
bst = xgb.train(param, dtrain, 2626)

yprob = bst.predict(dtest)
pred = pd.DataFrame(yprob)
pred.to_csv("/Users/mac/Desktop/termtwo/DS/project/xgboost_pred.csv")

