# import libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# helper function to analyze prediction results
def preds_analysis(truth, preds, pred_type):
    
    if (pred_type not in ('Train', 'Validation', 'Test')):
        raise ValueError(f"Attribute 'pred_type' must be either 'Train', 'Validation', or 'Test'.")
    
    print(f"---------- Evaluation ({pred_type}) ----------\n")

    # output all metrics scores
    print("Accuracy:", accuracy_score(truth, preds, normalize=True, sample_weight=None))
    print("Precision:", precision_score(truth, preds, sample_weight=None))
    print("Recall:", recall_score(truth, preds, sample_weight=None))
    print("F1 Score:", f1_score(truth, preds, sample_weight=None))

    # display confusion matrix
    print("Confusion matrix:\n", confusion_matrix(truth, preds))
    
    # print classification report
    print("Classification report:\n", classification_report(truth, preds))
    return None


# helper function to analyze model
def model_eval(model, X, y):
    # training
    print("---------- Evaluation ----------\n")
    print("Evaluation: Training")
    preds = model.predict(X_train)

    # output all metrics scores
    print("\tAccuracy:", accuracy_score(y_train, preds, normalize=True, sample_weight=None))
    # print("Precision:", precision_score(truth, preds, sample_weight=None))
    # print("Recall:", recall_score(truth, preds, sample_weight=None))

    # display confusion matrix
    print("\tConfusion matrix:\n", confusion_matrix(y_train, preds))
    
    # print classification report
    print("\tClassification report:\n", classification_report(y_train, preds))
    
    
    # validation
    print("Evaluation: Validation")
    preds = model.predict(X_val)

    # output all metrics scores
    print("\tAccuracy:", accuracy_score(y_val, preds, normalize=True, sample_weight=None))
    # print("Precision:", precision_score(truth, preds, sample_weight=None))
    # print("Recall:", recall_score(truth, preds, sample_weight=None))

    # display confusion matrix
    print("\tConfusion matrix:\n", confusion_matrix(y_val, preds))
    
    # print classification report
    print("\tClassification report:\n", classification_report(y_val, preds))
    
    return None




# module definition overhead
if __name__=="__init__":
    pass