import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


def load_data_rf(csv_file):
    df = pd.read_csv(csv_file)
    # Assuming the target column is named 'winner'
    X = df.drop(columns=['winner'])

    # Apply standardization
    # scaler_standard = StandardScaler()
    # X_standardized = pd.DataFrame(scaler_standard.fit_transform(X), columns=X.columns)

    # # Apply normalization
    # scaler_minmax = MinMaxScaler()
    # X_normalized = pd.DataFrame(scaler_minmax.fit_transform(X), columns=X.columns)
    y = df['winner']
    return X, y

def desc_tree(mode = "Hardpoint", viz = False, ablation=False):
    name = "hardpoint"
    if mode == "Control":
        name = "control"
    elif mode == "SnD":
        name = "snd"
    # Load your CSV data into a DataFrame
    df = pd.read_csv(f"./lentz_final/CDL {mode}/{name}_clean.csv")

    # Specify the features (X) and the target variable (y)
    X = df.drop(columns=['winner', "matchGame.matchId"], axis=1)
    y = df['winner']

    # Split the data into training (70%), testing (15%), and validation (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)
    if not ablation:
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)
    else:
        X_test, y_test, = X_temp, y_temp
         # Load your CSV data into a DataFrame
        abl_df = pd.read_csv(f"./lentz_final/CDL {mode}/ablation.csv")

        # Specify the features (X) and the target variable (y)
        X_valid = abl_df.drop(columns=['winner'], axis=1)
        y_valid = abl_df['winner']
    
    # Assuming X_train and y_train are your training features and labels
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [10, 20, 30, None],
        'max_leaf_nodes': [10, 20, 30, None],
        'criterion': ["gini", "entropy"]
    }

    dt_model = DecisionTreeClassifier(random_state=69)
    dt_model.fit(X_train, y_train)

    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Extract relevant information from cv_results_
    results = pd.DataFrame(grid_search.cv_results_)

    if viz == True:
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results)), results['mean_test_score'], color='skyblue')
        plt.xlabel('Model Number')
        plt.ylabel('Mean Test Score (Accuracy)')
        plt.title('Cross-Validated Accuracy for Different Models')
        plt.xticks([])  # Remove x-axis labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'./lentz_final/CDL {mode}/grid_search_plot.png')

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    best_dt_model = DecisionTreeClassifier(random_state=69, **best_params)
    best_dt_model.fit(X_train, y_train)
    print(f"MODE: {mode}")
    print(f"Best Params: {best_params}")

    # Predict on the validation set
    y_pred_0 = best_dt_model.predict(X_train)
    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_train, y_pred_0)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    # Predict on the validation set

    # Predict on the validation set
    y_pred = best_dt_model.predict(X_valid)
    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    # Predict on the validation set

    y_pred_t = best_dt_model.predict(X_test)
    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_test, y_pred_t)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot the decision tree
    if viz == True:
        plt.figure(figsize=(30, 17))
        plot_tree(best_dt_model, filled=True, feature_names=list(X_train.columns), class_names=['Loss', 'Win'])
        plt.savefig(f'./lentz_final/CDL {mode}/decision_tree_plot_altsplit.png', dpi = 300)  # Save the figure as an image

def random_forest(mode = "Hardpoint", viz = False, ablation=False):
    # Load data
    name = "hardpoint"
    if mode == "Control":
        name = "control"
    elif mode == "SnD":
        name = "snd"
    # Load your CSV data into a DataFrame
    df = pd.read_csv(f"./lentz_final/CDL {mode}/{name}_clean.csv")

    # Specify the features (X) and the target variable (y)
    X = df.drop(columns=['winner', "matchGame.matchId"], axis=1)
    y = df['winner']

    # Split the data into training (70%), testing (15%), and validation (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)
    if not ablation:
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)
    else:
        X_test, y_test, = X_temp, y_temp
         # Load your CSV data into a DataFrame
        abl_df = pd.read_csv(f"./lentz_final/CDL {mode}/ablation.csv")

        # Specify the features (X) and the target variable (y)
        X_valid = abl_df.drop(columns=['winner'], axis=1)
        y_valid = abl_df['winner']
    
    param_grid = {
        'bootstrap': [True, False],
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_leaf_nodes': [10, 20, 30, None],
        'max_features': ['auto'],  
        'criterion' : ['gini', 'entropy']
    }
    # Create a RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=69)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Extract relevant information from cv_results_
    results = pd.DataFrame(grid_search.cv_results_)


    if viz == True:
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(results)), results['mean_test_score'], color='skyblue')
        plt.xlabel('Model Number')
        plt.ylabel('Mean Test Score (Accuracy)')
        plt.title('Cross-Validated Accuracy for Different Models')
        plt.xticks([])  # Remove x-axis     labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'./lentz_final/CDL {mode}/grid_search_plot_rf.png')

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train the model with the best parameters
    best_rf_model = RandomForestClassifier(random_state=42)
    best_rf_model.fit(X_train, y_train)

    feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig(f'./lentz_final/CDL {mode}/rf_features_{mode}.png')

    # Predict on the validation set
    y_pred_0 = best_rf_model.predict(X_train)
    # Calculate accuracy on the validation set
    accuracy_train = accuracy_score(y_train, y_pred_0)

    # Predict on the validation set
    y_pred = best_rf_model.predict(X_valid)
    # Calculate accuracy on the validation set
    accuracy_valid = accuracy_score(y_valid, y_pred)

    # Predict on the validation set
    y_pred_test = best_rf_model.predict(X_test)
    # Calculate accuracy on the validation set
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(mode)
    print(f"Best Features {len(best_rf_model.feature_names_in_)}:", best_rf_model.feature_names_in_)
    print("Best Train Accuracy:", accuracy_train)
    print("Best Validation Accuracy:", accuracy_valid)
    print("Best Test Accuracy:", accuracy_test)

def random_forest_tuned(mode="Hardpoint", viz=False, ablation=False):
    # Load data
    name = "hardpoint"
    if mode == "Control":
        name = "control"
    elif mode == "SnD":
        name = "snd"
    
    # Load your CSV data into a DataFrame
    df = pd.read_csv(f"./lentz_final/CDL {mode}/{name}_clean.csv")

    # Specify the features (X) and the target variable (y)
    X = df.drop(columns=['winner', "matchGame.matchId"], axis=1)
    y = df['winner']

    # Split the data into training (70%), testing (15%), and validation (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)
    
    if not ablation:
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)
    else:
        X_test, y_test, = X_temp, y_temp
        # Load your CSV data into a DataFrame
        abl_df = pd.read_csv(f"./lentz_final/CDL {mode}/ablation.csv")

        # Specify the features (X) and the target variable (y)
        X_valid = abl_df.drop(columns=['winner'], axis=1)
        y_valid = abl_df['winner']

    # Set specific hyperparameters for each mode
    if mode == "Hardpoint":
        best_params = {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto',
                       'max_leaf_nodes': 30, 'min_samples_split': 5, 'n_estimators': 100}
    elif mode == "Control":
        best_params = {'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto',
                       'max_leaf_nodes': None, 'min_samples_split': 2, 'n_estimators': 100}
    elif mode == "SnD":
        best_params = {'bootstrap': True, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'auto',
                       'max_leaf_nodes': None, 'min_samples_split': 10, 'n_estimators': 100}
    
    # Create a RandomForestClassifier with specific hyperparameters
    best_rf_model = RandomForestClassifier(random_state=69, **best_params)
    best_rf_model.fit(X_train, y_train)

    feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.subplots_adjust(bottom=0.35)  # Increase the bottom margin
    if ablation:
        plt.savefig(f'./lentz_final/CDL {mode}/rf_features_{mode}_abl.png')
    else:
        plt.savefig(f'./lentz_final/CDL {mode}/rf_features_{mode}.png')


    # Predict on the validation set
    y_pred_0 = best_rf_model.predict(X_train)
    # Calculate accuracy on the validation set
    accuracy_train = accuracy_score(y_train, y_pred_0)

    # Predict on the validation set
    y_pred = best_rf_model.predict(X_valid)
    # Calculate accuracy on the validation set
    accuracy_valid = accuracy_score(y_valid, y_pred)

    # Predict on the validation set
    y_pred_test = best_rf_model.predict(X_test)
    # Calculate accuracy on the validation set
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(mode)
    print(f"Best Features {len(best_rf_model.feature_names_in_)}:", best_rf_model.feature_names_in_)
    print("Importances: ", feature_importances)
    print("Best Train Accuracy:", accuracy_train)
    print("Best Validation Accuracy:", accuracy_valid)
    print("Best Test Accuracy:", accuracy_test)


def main():
    # Train Random Forest Classifier
    hp_best = random_forest_tuned(mode = "Hardpoint", ablation=True)
    print(" ")
    ctrl_best = random_forest_tuned(mode = "Control", ablation=True)
    print(" ")
    snd_best = random_forest_tuned(mode = "SnD", ablation=True)

if __name__ == "__main__":
    main()