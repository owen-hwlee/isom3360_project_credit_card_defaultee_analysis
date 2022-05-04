# This is a customer cost function to input into GridCV to search for the optimal hyperparameters
# Instead of using traditional metrics such as loss function or accuracies that assigns equal weighting to each misclassification,
# we have decided to employ a custom cost function that takes misclassification cost into account.

# Referenced documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring

from sklearn.metrics import make_scorer

# We estimate the following figures:
# - FP: Misclassified good customer (actually does not default)
#   - Cost: 3306
# - FN: Misclassified bad customer (actually defaults)
#   - Cost: 6500

# According to US statistics, the average credit card balance is approximately $6500 USD.
# This is essentially a one-time cost because credit card defaultees will likely not be able to apply for credit card services again.
# Cost of undetected bad customer is 6500.

# According to data from JP Morgan, banks earn a profit margin of around 4.17% - 4.21% of credit card balance.
# Annual profit per credit card user is at $273 USD.
# We can assume this profit will go on for 15 years if it is a good credit client.
# Taking annual discount rate at 2.8%, the present value of the profit is 3306.
# Cost of misclassified good customer is 3306.

cost_fp = 3306
cost_fn = 6500

# custom cost function, returns financial/economic misclassification cost
def cost(y_true, y_pred):
    total = 0
    for i in range(y_true.shape[0]):
        # FP, true=0, pred=1
        if y_pred[i] > y_true.iloc[i,0]:
            total += cost_fp
        # FN, true=1, pred=0
        if y_true.iloc[i,0] > y_pred[i]:
            total += cost_fn
    return total

# customer scorer object, to be imported for use during cross validation
custom_loss = make_scorer(cost, greater_is_better=False)
# !!!!!!!!!!!!!!!!!!!! THIS SHOULD ACTUALLY NOT BE USED FOR CROSS VALIDATION
# !!!!!!!!!!!!!!!!!!!! COST ANALYSIS SHOULD ONLY BE USED TO FIND DECISION THRESHOLD

# economic cost analysis function, returns economic cost
def economic_analysis(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    econ_cost = cost(y_true, y_pred)
    print("The estimated economic cost at decision threshold {0:.3f} = ${1}".format(threshold, econ_cost))
    return econ_cost



# module definition overhead
if __name__=="__main__":
    pass