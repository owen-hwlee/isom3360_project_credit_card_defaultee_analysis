# import libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


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
def model_eval(model, X, y, cv=5):
    # obtain model predicted probabilities
    # for class 1
    proba_preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # get false positive and true positives for each model
    fpr, tpr, thresholds = roc_curve(y, proba_preds)
    # calculate area under curve
    area_under_curve = auc(fpr, tpr)
    
    # plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, marker='o', color='darkorange', label="ROC curve (area = %0.3f)" % area_under_curve)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.03, 1.0])
    plt.ylim([0.0, 1.03])
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc='lower right')
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
    
    print(f"The AUC score for this model is {area_under_curve}.")
    
    return None




# module definition overhead
if __name__=="__init__":
    pass