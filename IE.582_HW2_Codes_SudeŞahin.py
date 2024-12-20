#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Data Preprocessing**
# 
# Firstly, data was uploaded, examined and organized in a way that would be useful for homework requirements and tasks.

# In[5]:


file_path = "match_data.csv"  
data = pd.read_csv(file_path)


# In[6]:


print(data.head())


# In[7]:


print(data.columns)


# In[8]:


print(data.describe())


# In[9]:


clean_data = data[(data['suspended'] != True) & (data['stopped'] != True)]
print(f"Number of rows of cleaned data: {len(clean_data)}")


# In[10]:


missing_values = clean_data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])


# In[11]:


missing_ratio = (missing_values / len(clean_data)) * 100
print(missing_ratio[missing_ratio > 0].sort_values(ascending=False))


# In[12]:


missing_ratio_filtered = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

# Identify columns with more than 50% missing values and clean
columns_to_drop = missing_ratio_filtered[missing_ratio_filtered > 50].index
clean_data = clean_data.drop(columns=columns_to_drop, axis=1)
print(f"Number of columns remaining: {clean_data.shape[1]}")


# In[13]:


# Fill in missing values ​​by selecting only numeric columns
numeric_columns = clean_data.select_dtypes(include='number').columns
clean_data[numeric_columns] = clean_data[numeric_columns].fillna(clean_data[numeric_columns].mean())

# Rechecking missing values
print(f"Number of remaining missing values: {clean_data.isnull().sum().sum()}")


# In[14]:


# Find out which columns have missing values
remaining_missing = clean_data.isnull().sum()
print(remaining_missing[remaining_missing > 0])


# In[15]:


print(clean_data.head())


# In[16]:


print(clean_data.columns)


# In[17]:


# Basic statistics for numeric columns
print(clean_data.describe())


# **TASK 1**

# In[18]:


# Calculate probabilities (P(home win), P(draw), P(away win))
clean_data['P_home_win'] = 1 / clean_data['1']  # Assuming column '1' represents home win odds
clean_data['P_draw'] = 1 / clean_data['X']      # Assuming column 'X' represents draw odds
clean_data['P_away_win'] = 1 / clean_data['2']  # Assuming column '2' represents away win odds

# Normalize probabilities
normalization_factor = clean_data['P_home_win'] + clean_data['P_draw'] + clean_data['P_away_win']
clean_data['P_home_win_norm'] = clean_data['P_home_win'] / normalization_factor
clean_data['P_draw_norm'] = clean_data['P_draw'] / normalization_factor
clean_data['P_away_win_norm'] = clean_data['P_away_win'] / normalization_factor

# Calculate P(home win) - P(away win)
clean_data['Home_Away_Diff'] = clean_data['P_home_win'] - clean_data['P_away_win']

# Bin P(home win) - P(away win) values
bins = np.linspace(-1, 1, 11)  # Define bins from -1 to 1 with 10 equal intervals
clean_data['Bins'] = pd.cut(clean_data['Home_Away_Diff'], bins=bins)

# Calculate actual draw probabilities for each bin
bin_analysis = clean_data.groupby('Bins').apply(
    lambda x: pd.Series({
        'Bookmaker_Draw_Probability': x['P_draw'].mean(),
        'Actual_Draw_Rate': (x['result'] == 'X').mean()  # Assuming 'result' contains the actual outcome
    })
).reset_index()


# In[19]:


import matplotlib.pyplot as plt

# Scatter Plot: P(draw) vs P(home win) - P(away win)
plt.figure(figsize=(12, 6))
plt.scatter(clean_data['Home_Away_Diff'], clean_data['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Enhance Plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot: P(draw) vs P(home win) - P(away win)')
plt.legend()
plt.grid()
plt.show()


# In[20]:


# Calculate midpoints for bins if not already done
bin_analysis['Bin_Midpoint'] = bin_analysis['Bins'].apply(lambda x: (x.left + x.right) / 2)

# Scatter Plot: P(draw) vs P(home win) - P(away win) with Binned Outcomes
plt.figure(figsize=(12, 6))

# Original bookmaker probabilities
plt.scatter(clean_data['Home_Away_Diff'], clean_data['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Binned outcomes: Actual draw probabilities
plt.scatter(bin_analysis['Bin_Midpoint'], bin_analysis['Actual_Draw_Rate'], color='red', label='Binned Actual Draw Probabilities', s=100)

# Enhance Plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot: P(draw) vs P(home win) - P(away win) with Binned Outcomes')
plt.legend()
plt.grid()
plt.show()


# In[21]:


import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Step 1: Fit a polynomial trend line for bookmaker probabilities
bookmaker_trend = Polynomial.fit(clean_data['Home_Away_Diff'], clean_data['P_draw'], deg=2)

# Generate predictions for bookmaker trend line
x_vals = np.linspace(-1, 1, 100)  # X-axis range for smooth trend line
bookmaker_y_vals = bookmaker_trend(x_vals)  # Polynomial predictions for bookmaker

# Step 2: Fit a polynomial trend line for binned actual probabilities
actual_trend = Polynomial.fit(bin_analysis['Bin_Midpoint'], bin_analysis['Actual_Draw_Rate'], deg=2)

# Generate predictions for actual outcomes trend line
actual_y_vals = actual_trend(x_vals)

# Step 3: Plot scatter plot and trend lines
plt.figure(figsize=(12, 6))

# Scatter plot for bookmaker probabilities
plt.scatter(clean_data['Home_Away_Diff'], clean_data['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Binned outcomes scatter points
plt.scatter(bin_analysis['Bin_Midpoint'], bin_analysis['Actual_Draw_Rate'], color='red', label='Binned Actual Draw Probabilities', s=100)

# Bookmaker trend line
plt.plot(x_vals, bookmaker_y_vals, color='black', linestyle='-', linewidth=3, label='Bookmaker Trend Line')  # Adjusted to black and thicker

# Actual outcomes trend line
plt.plot(x_vals, actual_y_vals, color='red', linestyle='--', linewidth=2, label='Actual Outcomes Trend Line')

# Enhance plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot with Trend Lines: P(draw) vs P(home win) - P(away win)')
plt.legend()
plt.grid()
plt.show()


# **Task 1 Data Interpretation**
# 
# As we can observe from the last Scatter Plot with Trend Lines: P(draw) vs P(home win) - P(away win) graph,
# 
# The graph visualizes the relationship between the difference in probabilities for a home win and an away win (P(home win) - P(away win)) on the x-axis and the probability of a draw (P(draw)) on the y-axis.
# Blue dots represent bookmaker-provided draw probabilities (P(draw)) for individual matches. These dots scatter around the bookmaker trend line and show how the draw probabilities vary as the difference between P(home win) and P(away win) changes. When P(home win) - P(away win) is close to 0, the draw probability is higher because teams are perceived to be evenly matched.
# Red dots represent binned actual draw probabilities, calculated based on the actual outcomes in different bins. These red dots aggregate the outcomes into bins, showing the observed probability of a draw for matches in each bin. This allows us to compare actual outcomes with bookmaker probabilities.
# The black line reflects the bookmaker's trend in predicting draw probabilities across different game scenarios. The red dashed line represents actual outcomes and helps identify whether the bookmaker's predictions align with observed match results.
# 
# **Insights and Interpretation**
# 
# The bookmaker trend line and actual outcomes trend line are fairly close to each other, indicating that bookmaker predictions are relatively accurate in capturing the likelihood of draws.However, slight deviations suggest that there may be room for inefficiencies or discrepancies in certain match situations.
# When P(home win) - P(away win) is near 0 (balanced teams), both bookmaker predictions and actual outcomes show the highest probabilities of a draw. This aligns with intuition, as closely matched teams are more likely to draw. At the extremes of P(home win) - P(away win) (e.g., -1 or 1), the actual outcomes trend line sometimes deviates from the bookmaker line. This may suggest potential inefficiencies in how bookmakers account for one-sided matches and having bias.
# To sum up, the graph shows that bookmaker predictions for draws are generally aligned with observed results, especially for balanced matches. However, there are slight discrepancies at the extremes, indicating possible inefficiencies in the odds market. These inefficiencies could be explored further for betting strategies.
# 
# 

# **TASK 2**

# In[22]:


# Determine matches with goals after the 90th minute
# If the 'minute' column is specified as 45+ or 90+, goal checks can be performed

# Filter rows corresponding to 90+ minutes
late_goals = clean_data[(clean_data['halftime'] == '2nd-half') & (clean_data['minute'] > 45)]

# Maç ID'lerini belirleme
late_goal_matches = late_goals['fixture_id'].unique()

print(f"Number of matches with goals in 90+ minutes: {len(late_goal_matches)}")


# In[23]:


# Filtering matches with red cards in the first 15 minutes
early_red_cards = clean_data[(clean_data['minute'] <= 15) & 
                             ((clean_data['Redcards - away'] > 0) | (clean_data['Redcards - home'] > 0))]

# Determining match IDs
red_card_matches = early_red_cards['fixture_id'].unique()

print(f"Number of matches with red cards in the first 15 minutes: {len(red_card_matches)}")


# In[24]:


# Merging noisy match IDs
noise_matches = set(late_goal_matches).union(set(red_card_matches))

# Removing noisy matches from the dataset
clean_task2 = clean_data[~clean_data['fixture_id'].isin(noise_matches)].copy()

print(f"Matches containing noise were removed. Number of remaining matches: {clean_task2['fixture_id'].nunique()}")


# In[25]:


print(clean_task2.describe())


# In[26]:


# Calculate probabilities (P(home win), P(draw), P(away win))
clean_task2['P_home_win'] = 1 / clean_task2['1']  # Assuming column '1' represents home win odds
clean_task2['P_draw'] = 1 / clean_task2['X']      # Assuming column 'X' represents draw odds
clean_task2['P_away_win'] = 1 / clean_task2['2']  # Assuming column '2' represents away win odds

# Normalize probabilities
normalization_factor = clean_task2['P_home_win'] + clean_task2['P_draw'] + clean_task2['P_away_win']
clean_task2['P_home_win_norm'] = clean_task2['P_home_win'] / normalization_factor
clean_task2['P_draw_norm'] = clean_task2['P_draw'] / normalization_factor
clean_task2['P_away_win_norm'] = clean_task2['P_away_win'] / normalization_factor

# Calculate P(home win) - P(away win)
clean_task2['Home_Away_Diff'] = clean_task2['P_home_win'] - clean_task2['P_away_win']

# Bin P(home win) - P(away win) values
bins = np.linspace(-1, 1, 11)  # Define bins from -1 to 1 with 10 equal intervals
clean_task2['Bins'] = pd.cut(clean_task2['Home_Away_Diff'], bins=bins)

# Calculate actual draw probabilities for each bin
bin_analysis_task2 = clean_task2.groupby('Bins').apply(
    lambda x: pd.Series({
        'Bookmaker_Draw_Probability': x['P_draw'].mean(),
        'Actual_Draw_Rate': (x['result'] == 'X').mean()  # Assuming 'result' contains the actual outcome
    })
).reset_index()


# In[27]:


import matplotlib.pyplot as plt

# Scatter Plot: P(draw) vs P(home win) - P(away win)
plt.figure(figsize=(12, 6))
plt.scatter(clean_task2['Home_Away_Diff'], clean_task2['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Enhance Plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot: P(draw) vs P(home win) - P(away win)')
plt.legend()
plt.grid()
plt.show()


# In[28]:


# Calculate midpoints for bins if not already done
bin_analysis_task2['Bin_Midpoint'] = bin_analysis_task2['Bins'].apply(lambda x: (x.left + x.right) / 2)

# Scatter Plot: P(draw) vs P(home win) - P(away win) with Binned Outcomes
plt.figure(figsize=(12, 6))

# Original bookmaker probabilities
plt.scatter(clean_task2['Home_Away_Diff'], clean_task2['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Binned outcomes: Actual draw probabilities
plt.scatter(bin_analysis_task2['Bin_Midpoint'], bin_analysis_task2['Actual_Draw_Rate'], color='red', label='Binned Actual Draw Probabilities', s=100)

# Enhance Plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot: P(draw) vs P(home win) - P(away win) with Binned Outcomes (Task 2)')
plt.legend()
plt.grid()
plt.show()


# In[29]:


import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Step 1: Fit a polynomial trend line for bookmaker probabilities
bookmaker_trend_task2 = Polynomial.fit(clean_task2['Home_Away_Diff'], clean_task2['P_draw'], deg=2)

# Generate predictions for bookmaker trend line
x_vals_task2 = np.linspace(-1, 1, 100)  # X-axis range for smooth trend line
bookmaker_y_vals_task2 = bookmaker_trend_task2(x_vals_task2)  # Polynomial predictions for bookmaker

# Step 2: Fit a polynomial trend line for binned actual probabilities
actual_trend_task2 = Polynomial.fit(bin_analysis_task2['Bin_Midpoint'], bin_analysis_task2['Actual_Draw_Rate'], deg=2)

# Generate predictions for actual outcomes trend line
actual_y_vals_task2 = actual_trend_task2(x_vals_task2)

# Step 3: Plot scatter plot and trend lines
plt.figure(figsize=(12, 6))

# Scatter plot for bookmaker probabilities
plt.scatter(clean_task2['Home_Away_Diff'], clean_task2['P_draw'], color='blue', alpha=0.5, label='Bookmaker Draw Probabilities')

# Binned outcomes scatter points
plt.scatter(bin_analysis_task2['Bin_Midpoint'], bin_analysis_task2['Actual_Draw_Rate'], color='red', label='Binned Actual Draw Probabilities', s=100)

# Bookmaker trend line
plt.plot(x_vals_task2, bookmaker_y_vals_task2, color='black', linestyle='-', linewidth=3, label='Bookmaker Trend Line')  # Adjusted to black and thicker

# Actual outcomes trend line
plt.plot(x_vals_task2, actual_y_vals_task2, color='red', linestyle='--', linewidth=2, label='Actual Outcomes Trend Line')

# Enhance plot
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('P(home win) - P(away win)')
plt.ylabel('P(draw)')
plt.title('Scatter Plot with Trend Lines: P(draw) vs P(home win) - P(away win) (Task 2)')
plt.legend()
plt.grid()
plt.show()


# **Task 2 Data Interpretation**
# 
# Task 2  focuses on evaluating bookmaker draw probabilities after removing certain matches that introduce noise, such as:
# Matches with decisive events toward the end of the game (e.g., winning/drawing goals after the 90th minute) and, matches affected by early red cards (e.g., red cards in the first 15 minutes).  This ensures that external, uncharacteristic events do not skew the results.This ensures that external, uncharacteristic events do not skew the results.
# 
# The x-axis represents P(home win) - P(away win), and the y-axis represents P(draw).
# 
# 
# The black trend line remains similar to Task 1, indicating that bookmaker predictions are robust even when noisy matches are excluded. The red dashed trend line shows actual draw probabilities, which remain significantly lower than bookmaker predictions, consistent with Task 1.
# At the extremes (P(home win) - P(away win) close to -1 or 1), actual draw probabilities remain near zero, highlighting that one-sided matches rarely end in draws. The gap between bookmaker predictions and actual outcomes at these extremes is still evident. In the middle range P(home win) - P(away win) near 0, there is a slight decrease in actual draw probabilities (red dots), which suggests that noisy matches had a slight upward bias on draw outcomes.
# The overall trends between bookmaker probabilities and actual outcomes nearly remain consistent with Task 1, even after removing noisy matches. Additionally, bookmakers still overestimate draw probabilities across all ranges.
# 
# **Effect of Noise:**
# 
# Removing noisy matches reduces the actual draw probabilities slightly in central bins, indicating that certain uncharacteristic match outcomes inflated observed draws in Task This validates the robustness of the model and bookmaker predictions, as their overall trends remain stable.For bettors or analysts, focusing on matches without extreme events (like red cards or last-minute goals) may provide a more accurate evaluation of bookmaker efficiency. The continued discrepancy between bookmaker and actual probabilities suggests inefficiencies that could be exploited.
# 
# 
# 

# **TASK3**

# In[30]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# In[31]:


# Load the dataset
data_path = 'match_data.csv'  # Path to your dataset
data = pd.read_csv(data_path)

# Display the first few rows and column names to understand the structure
print(clean_data.head())  # Inspect the first 5 rows
print(clean_data.columns)  # Check all column names


# In[32]:


# Convert odds to probabilities
clean_data['P_home_win'] = 1 / clean_data['1']
clean_data['P_draw'] = 1 / clean_data['X']
clean_data['P_away_win'] = 1 / clean_data['2']

# Normalize probabilities
normalization_factor = clean_data['P_home_win'] + clean_data['P_draw'] + clean_data['P_away_win']
clean_data['P_home_win_norm'] = clean_data['P_home_win'] / normalization_factor
clean_data['P_draw_norm'] = clean_data['P_draw'] / normalization_factor
clean_data['P_away_win_norm'] = clean_data['P_away_win'] / normalization_factor


# In[5]:


# Map 'result' (target variable) to numeric values
result_mapping = {'1': 0, 'X': 1, '2': 2}  # 0: Home Win, 1: Draw, 2: Away Win
clean_data['result'] = clean_data['result'].map(result_mapping)


# In[33]:


# Remove rows with missing values
data = clean_data.dropna()


# In[34]:


# Define the features for the model
selected_features = [
    'P_home_win_norm',        # Normalized probability for home win
    'P_draw_norm',            # Normalized probability for draw
    'P_away_win_norm',        # Normalized probability for away win
    'Ball Possession % - home', # Ball possession percentage by home
    'Ball Possession % - away', # Ball possession percentage by away
    'Shots On Target - home', # Shots on target by home team
    'Shots On Target - away', # Shots on target by away team
    'Corners - home',         # Corners by home team
    'Corners - away'          # Corners by away team
]

# Define features (X) and target (y)
X = clean_data[selected_features]
y = clean_data['result']


# In[35]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


# Initialize the Decision Tree model
decision_tree = DecisionTreeClassifier(
    max_depth=5,              # Limit the depth of the tree to avoid overfitting
    criterion='gini',         # Use Gini impurity as the splitting criterion
    min_samples_split=10      # Minimum samples required to split an internal node
)

# Train the Decision Tree model
decision_tree.fit(X_train, y_train)


# In[37]:


# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[38]:


# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree, 
    feature_names=selected_features, 
    class_names=['Home Win', 'Draw', 'Away Win'], 
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()


# In[39]:


# Get predicted probabilities for each class
predicted_probabilities = decision_tree.predict_proba(X_test)

# Create a DataFrame to compare predicted and implied probabilities
comparison = pd.DataFrame({
    'Predicted_Draw_Probability': predicted_probabilities[:, 1],  # Model's predicted probability for draw
    'Implied_Draw_Probability': X_test['P_draw_norm']             # Bookmaker's implied probability for draw
})

# Calculate deviation between predicted and implied probabilities
comparison['Deviation'] = comparison['Predicted_Draw_Probability'] - comparison['Implied_Draw_Probability']

# Display the first few rows of the comparison
print(comparison.head())


# In[40]:


# Get predicted probabilities for each class
predicted_probabilities = decision_tree.predict_proba(X_test)

# Create a DataFrame for comparison
comparison = pd.DataFrame({
    'Predicted_Home_Prob': predicted_probabilities[:, 0],
    'Predicted_Draw_Prob': predicted_probabilities[:, 1],
    'Predicted_Away_Prob': predicted_probabilities[:, 2],
    'Implied_Home_Prob': X_test['P_home_win_norm'],
    'Implied_Draw_Prob': X_test['P_draw_norm'],
    'Implied_Away_Prob': X_test['P_away_win_norm']
})

# Calculate deviations
comparison['Home_Deviation'] = comparison['Predicted_Home_Prob'] - comparison['Implied_Home_Prob']
comparison['Draw_Deviation'] = comparison['Predicted_Draw_Prob'] - comparison['Implied_Draw_Prob']
comparison['Away_Deviation'] = comparison['Predicted_Away_Prob'] - comparison['Implied_Away_Prob']

# Display deviations
print(comparison[['Home_Deviation', 'Draw_Deviation', 'Away_Deviation']].describe())


# **Task 3 Data Interpretation**
# 
# The decision tree achieved an accuracy of 64.34%, highlighting its moderate ability to predict match outcomes based on bookmaker probabilities and match statistics. Key splits in the tree relied heavily on normalized probabilities (P_home_win_norm, P_draw_norm, P_away_win_norm), demonstrating their importance in predicting outcomes. Secondary splits included game dynamics such as ball possession and shots on target, which further refined predictions. However, the model struggled more with "Draw" and "Away Win" classifications, reflecting the inherent challenges in predicting these less frequent outcomes.
# 
# **Market Efficiency Insights**
# 
# Deviations between the model’s predicted probabilities and bookmaker-implied probabilities revealed notable inefficiencies:
# Draw Probabilities: Bookmakers often overestimate draw probabilities, as evidenced by larger deviations (mean = 0.031). This inefficiency could be exploited by bettors targeting specific match conditions.
# Home and Away Wins: Deviations for home and away wins were smaller (means = 0.006 and -0.037, respectively), indicating that bookmaker odds are generally more efficient for these outcomes.
# The decision tree suggests that match statistics complement bookmaker probabilities well, offering actionable insights into match outcomes. Moreover, the observed inefficiencies, particularly for draws, highlight potential opportunities for improved betting strategies or market adjustments.
# 
# 

# **Conclusions and Real-World Implications**
# 
# The analysis revealed that bookmaker odds are generally efficient for "Home Win" and "Away Win" predictions, but they consistently overestimate draw probabilities. This inefficiency provides potential opportunities for bettors to exploit patterns where bookmakers overvalue draws, particularly in balanced matches. Additionally, match statistics such as possession and shots on target proved crucial in enhancing predictive accuracy, suggesting that real-time game dynamics are significant factors in understanding outcomes.
# 
# **Challenges Faced and How They Were Overcome**
# 
# One major challenge was managing the large and complex dataset, which required significant preprocessing to remove noisy data, such as matches affected by early red cards or last-minute goals. This was addressed by cleaning the data and selecting features with the highest predictive power. Another challenge was balancing the model's predictions across imbalanced outcomes like "Draw," which was mitigated by optimizing decision tree parameters to ensure fair representation and prevent overfitting.
# 

# **References:**
# 
# [1] Jonas Mirza and Niklas Fejes,2016, “Statistical Football Modeling A Study of Football
# Betting and Implementation of Statistical Algorithms in Premier League”
# 
# [2] Štrumbelj, E., 2014. On determining probability forecasts from betting odds. International
# journal of forecasting, 30(4), pp.934-943.
# 
# [3] Shin, H.S., 1993. Measuring the incidence of insider trading in a market for state-contingent
# claims. The Economic Journal, 103(420), pp.1141-1153.
# 
# [4] OpenAI. (2024). ChatGPT (December 2024 version). Retrieved from https://openai.com/chatgpt
# 
