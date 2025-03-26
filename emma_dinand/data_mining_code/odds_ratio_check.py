import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial


def odds_ratio_linear_check(X, y):

    # X contains the feature vectors, y output (0/1)
    # somehow check if the odds ratio of X to y is linear
    # return the linear odds ratio
    X = pd.DataFrame({
    'X1': (X[:,0]),  # First exposure variable
    'X2': (X[:,1]),  # Second exposure variable
    'X3': (X[:,2]),  # Third exposure variable
    'Y': y  # Binary outcome
    })
    
    
    X = sm.add_constant(X[['X1', 'X2', 'X3']])
    X['Y'] = y
    
    model = sm.Logit(y, X[['const', 'X1', 'X2', 'X3']])
    result = model.fit()
    

    # Get odds ratios
    odds_ratios = np.exp(result.params)
    p_values = result.pvalues
    conf_intervals = np.exp(result.conf_int())

    # Display results
    summary_table = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'p-value': p_values,
        'Lower CI': conf_intervals[0],
        'Upper CI': conf_intervals[1]
    })

    print(summary_table)
    claude_test(result, X, y)
    # def log_odds_plot(data, x_var, ax):
        
    #     data['bin'] = pd.qcut(data[x_var], q=20, duplicates='drop')  # Bin X into 10 quantiles
    #     grouped = data.groupby('bin').agg(mean_X=(x_var, 'mean'), mean_Y=('Y', 'mean'))
    #     grouped['log_odds'] = np.log(grouped['mean_Y'] / (1 - grouped['mean_Y']))  # Compute log-odds
    #     sns.regplot(x=grouped['mean_X'], y=grouped['log_odds'], ax=ax, ci=None, scatter_kws={"s": 50})
        
    #     ax.set_xlabel(x_var)
    #     ax.set_ylabel("Log-Odds of Y=1")
    #     ax.set_title(f"Log-Odds vs {x_var}")

    # # Plot for each predictor
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # for i, var in enumerate(['X1', 'X2', 'X3']):
    #     log_odds_plot(X.copy(), var, axes[i])

    # plt.tight_layout()
    # plt.show()

def claude_test(result, X, y):
    # Get predicted probabilities
    predicted_probs = result.predict()

    # Calculate logits
    logits = np.log(predicted_probs / (1 - predicted_probs))

    # Plot logits against each predictor
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(X['X1'], logits)
    axes[0].set_title('Logit vs X1')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('Logit')

    axes[1].scatter(X['X2'], logits)
    axes[1].set_title('Logit vs X2')
    axes[1].set_xlabel('X2')

    axes[2].scatter(X['X3'], logits)
    axes[2].set_title('Logit vs X3')
    axes[2].set_xlabel('X3')

    plt.tight_layout()
    plt.show()

    # Another approach: Use Box-Tidwell test for linearity
    # Create interaction terms between each predictor and its natural log
    # X_test = X.copy()
    # for var in ['X1', 'X2', 'X3']:
    #     # Add small constant to handle zero values if needed
    #     X_test[f'{var}_ln'] = X[var] * np.log(X[var] + 0.00001)

    # # Fit model with interaction terms
    # model_bt = sm.Logit(y, X_test[['const', 'X1', 'X2', 'X3', 'X1_ln', 'X2_ln', 'X3_ln']])
    # result_bt = model_bt.fit()

    # If any of the interaction terms are significant, linearity assumption is violated
    # print(result_bt.summary())