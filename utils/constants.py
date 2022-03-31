# constants

random_state = 3360

# list of features
# this list is compiled after finishing data preprocessing
# hence this list can directly be imported and used in model evaluation

categorical_features = []

numerical_features = []

features = sorted(categorical_features + numerical_features)



# module definition overhead
if __name__=="__init__":
    pass