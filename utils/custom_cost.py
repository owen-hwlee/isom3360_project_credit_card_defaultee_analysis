# This is a customer cost function to input into GridCV to search for the optimal hyperparameters
# Instead of using traditional metrics such as loss function or accuracies that assigns equal weighting to each misclassification,
# we have decided to employ a custom cost function that takes misclassification cost into account.

# Referenced documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring

from sklearn.metrics import make_scorer

# We estimate the following figures:
# - FP: Misclassified bad customer
#   - Cost: 6500
# - FN: Misclassified good customer
#   - Cost: 5700

cost_fp = 6500
cost_fn = 5700

# custom cost function, returns financial/economic misclassification cost
def cost(y_true, y_pred):
    total = 0
    for i in range(len(y_true)):
        # FP, true=0, pred=1
        if y_pred[i] > y_true[i]:
            total += cost_fn
        # FN, true=1, pred=0
        if y_true[i] > y_pred[i]:
            total += cost_fn
    return total

# customer scorer object, to be imported for use during cross validation
custom_loss = make_scorer(cost, greater_is_better=False)



# module definition overhead
if __name__=="__main__":
    pass