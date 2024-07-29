#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pulp


# In[2]:


import pandas as pd
import numpy as np
import pulp

# Initialize dataset
df = pd.DataFrame({
    "account_id": [f"account_{i+1}" for i in range(100)],
    "csm_name": np.random.choice([f"csm_{i+1}" for i in range(10)], size=100),
    "health_score": np.random.randint(0, 101, size=100),
    "propensity_score": np.random.randint(0, 101, size=100),
    "score_1": np.random.randint(0, 101, size=100),
    "score_2": np.random.randint(0, 101, size=100),
    "score_3": np.random.randint(0, 101, size=100),
})

# Initialize global problem variable
prob = None
x = None
total_csm_scores = None

def initialize_problem(df, num_csms, max_accounts_per_csm=30):
    global prob, x, total_csm_scores
    num_accounts = len(df)
    prob = pulp.LpProblem("CSM_Routing_Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(num_accounts) for j in range(num_csms)], cat='Binary')

    # Calculate total scores for each account
    total_scores = df['health_score'] + df['propensity_score'] + df['score_1'] + df['score_2'] + df['score_3']
    
    # Auxiliary variables for the total score assigned to each CSM
    total_csm_scores = [pulp.LpVariable(f"total_score_{j}", cat='Continuous') for j in range(num_csms)]
    
    for j in range(num_csms):
        prob += total_csm_scores[j] == pulp.lpSum(x[i, j] * total_scores.iloc[i] for i in range(num_accounts))
    
    # Calculate the average total score
    avg_total_score = pulp.lpSum(total_csm_scores[j] for j in range(num_csms)) / num_csms
    
    # Auxiliary variables for the differences
    diff = [pulp.LpVariable(f"diff_{j}", cat='Continuous') for j in range(num_csms)]
    for j in range(num_csms):
        prob += diff[j] == total_csm_scores[j] - avg_total_score
    
    # Auxiliary variables for squared differences using absolute value and linear constraints
    diff_squared = [pulp.LpVariable(f"diff_squared_{j}", lowBound=0, cat='Continuous') for j in range(num_csms)]
    abs_diff = [pulp.LpVariable(f"abs_diff_{j}", lowBound=0, cat='Continuous') for j in range(num_csms)]
    for j in range(num_csms):
        prob += abs_diff[j] >= diff[j]
        prob += abs_diff[j] >= -diff[j]
        prob += diff_squared[j] >= abs_diff[j]

    # Objective function: minimize the sum of squared differences
    prob += pulp.lpSum(diff_squared[j] for j in range(num_csms))

    # Constraint 1: Each account is assigned to exactly one CSM
    for i in range(num_accounts):
        prob += pulp.lpSum(x[i, j] for j in range(num_csms)) == 1

    # Constraint 2: Each CSM is assigned no more than max_accounts_per_csm accounts
    for j in range(num_csms - 1):
        prob += pulp.lpSum(x[i, j] for i in range(num_accounts)) <= max_accounts_per_csm

    # Calculate the mean number of assignments for existing CSMs
    mean_assignments = num_accounts / num_csms

    # Constraint for new CSM to get at least the mean number of assignments
    prob += pulp.lpSum(x[i, num_csms - 1] for i in range(num_accounts)) >= mean_assignments

    # Define current assignment (for reducing shuffling)
    current_assignment = df['csm_name'].apply(lambda csm: int(csm.split('_')[1]) - 1 if pd.notna(csm) else np.nan)

    # Constraint 3: Minimize shuffling by penalizing changes in assignment
    shuffle_penalty = pulp.lpSum(x[i, j] * (current_assignment.iloc[i] != j) for i in range(num_accounts) for j in range(num_csms) if pd.notna(current_assignment.iloc[i]))
    prob += shuffle_penalty

def solve_problem(df, num_csms, max_accounts_per_csm=30):
    initialize_problem(df, num_csms, max_accounts_per_csm)
    prob.solve()
    num_accounts = len(df)
    assignments = [(i, j) for i in range(num_accounts) for j in range(num_csms) if pulp.value(x[i, j]) == 1]
    optimized_df = df.copy()
    optimized_df['new_csm_name'] = [f'csm_{j+1}' for i, j in assignments]
    return optimized_df

def add_new_csm(df):
    num_csms = len(df['csm_name'].unique()) + 1
    return solve_problem(df, num_csms, max_accounts_per_csm=30)

def add_new_accounts(df, new_accounts):
    new_accounts['csm_name'] = np.nan
    new_df = pd.concat([df, new_accounts], ignore_index=True)
    num_csms = len(new_df['csm_name'].dropna().unique()) + 1  # Add one to account for the new accounts with NaN CSM
    return solve_problem(new_df, num_csms, max_accounts_per_csm=30)

def remove_csm(df, csm_to_remove):
    df = df[df['csm_name'] != csm_to_remove]
    num_csms = len(df['csm_name'].unique())
    return solve_problem(df, num_csms, max_accounts_per_csm=30)

def remove_accounts(df, accounts_to_remove):
    df = df[~df['account_id'].isin(accounts_to_remove)]
    num_csms = len(df['csm_name'].unique())
    return solve_problem(df, num_csms, max_accounts_per_csm=30)

def get_total_scores(optimized_df):
    total_scores = optimized_df.groupby('new_csm_name')[['health_score', 'propensity_score', 'score_1', 'score_2', 'score_3']].sum()
    total_scores['total_score'] = total_scores.sum(axis=1)
    return total_scores

# Example usage:
optimized_df = solve_problem(df, 10)
print("Initial Optimization:")
optimized_df


# In[3]:


df['csm_name'].value_counts().sort_values().sort_index()


# In[ ]:





# In[4]:


# Add new CSM
optimized_df = add_new_csm(df)
print("After adding a new CSM:")
print(optimized_df.head())


# In[5]:


# Get total scores for each CSM
total_scores = get_total_scores(optimized_df)
print("Total Scores for each CSM:")
total_scores


# In[6]:


# Verify new CSMs in the DataFrame
print("CSM Distribution after adding new CSM:")
optimized_df['new_csm_name'].value_counts().sort_values().sort_index()


# In[7]:


# Get total scores for each CSM
total_scores = get_total_scores(optimized_df)
print("Total Scores for each CSM:")
total_scores


# In[8]:


optimized_df['csm_name'] = optimized_df['new_csm_name']


# In[9]:


# Add new CSM
optimized_df = add_new_csm(optimized_df)
print("After adding a new CSM:")
optimized_df


# In[ ]:





# In[10]:


# Get total scores for each CSM
total_scores = get_total_scores(optimized_df)
print("Total Scores for each CSM:")
total_scores


# In[ ]:





# In[11]:


optimized_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




