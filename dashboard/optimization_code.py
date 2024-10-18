# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:17:48 2024

@author: AR13020
"""
# https://chatgpt.com/c/66f61921-5f44-8007-a1c4-c2a528223073
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from ortools.linear_solver import pywraplp
solver = pywraplp.Solver.CreateSolver('CBC')

# Sets
T = range(30)  # time periods, e.g., range(24) for 24 hours or days
S = range(19)        # skills
W = range(10)     # workforce    
P = range(20)      # Projects 

# Parameters
all_sheets = pd.read_excel('Input.xlsx', sheet_name=None, header=None)

sheet_names = list(all_sheets.keys())

Sets    = all_sheets[sheet_names[0]] 
W_skills_Avl   = all_sheets[sheet_names[1]] 
Duration_dict     = all_sheets[sheet_names[2]]
Req_skills   = all_sheets[sheet_names[3]]
Manager_skills_Avl    = all_sheets[sheet_names[4]]
Due_Starting_Date     = all_sheets[sheet_names[5]]
Due_Finishing_Date   = all_sheets[sheet_names[6]]
# WAD_sheet            = all_sheets[sheet_names[7]]
# Score_sheet          = all_sheets[sheet_names[8]]
Cost_sheet           = all_sheets[sheet_names[9]]    
Prec_sheet = all_sheets[sheet_names[10]]
calendar_df = all_sheets[sheet_names[11]]

# print(Calendar)
print(Cost_sheet)
 
 
 
 
calendar_df.reset_index(drop=True, inplace=True)


Cal = {}

for i in range(calendar_df.shape[0]):  
    for j in range(calendar_df.shape[1]): 
        value = calendar_df.iloc[i, j]  
        
        # Adjust index manually
        adjusted_i = i - 1
        adjusted_j = j - 1

        if isinstance(value, (int, float)) and not pd.isna(value):
            Cal[(adjusted_i, adjusted_j)] = int(value)

print('Calender',Cal)

Prec_sheet.reset_index(drop=True, inplace=True)

Prec = {}

for i in range(Prec_sheet.shape[0]):  
    for j in range(Prec_sheet.shape[1]): 
        value = Prec_sheet.iloc[i, j]  
        
        # Adjust index manually
        adjusted_i = i - 1
        adjusted_j = j - 1

        if isinstance(value, (int, float)) and not pd.isna(value):
            Prec[(adjusted_i, adjusted_j)] = int(value)

print('Precedence Matrix',Prec)



skills_W = W_skills_Avl.iloc[1:, 0]   
workers = W_skills_Avl.iloc[1:, 1]  
values_W = W_skills_Avl.iloc[1:, 2]  
 
unique_skills  = skills_W.unique()
unique_workers = workers.unique()

WSAvl  = {}

# Populate the dictionary with values from the DataFrame
for i in range(len(skills_W )):
    skill = skills_W.iloc[i]
    worker = workers.iloc[i]
    value = values_W.iloc[i]
    
    # Find the index of the current skill and project
    skill_index = list(unique_skills).index(skill)
    Worker_index = list(unique_workers).index(worker)
    
    # Add the value to the dictionary
    WSAvl[(skill_index , Worker_index )] = int(value)

# print(WSAvl )

# sys.exit()

skills_Req = Req_skills.iloc[1:, 0]   # Skills column
projects_Req = Req_skills.iloc[1:, 1]  # Projects column
values_Req = Req_skills.iloc[1:, 2]    # Values column

# Get unique skills and projects
unique_skills_req = skills_Req.unique()
unique_projects_req = projects_Req.unique()

# Create the dictionary for storing requirement values
RSkl = {}

# Populate the dictionary with values from the DataFrame
for i in range(len(skills_Req)):
    skill = skills_Req.iloc[i]
    project = projects_Req.iloc[i]
    value = values_Req.iloc[i]
    
    # Find the index of the current skill and project
    skill_index = list(unique_skills_req).index(skill)
    project_index = list(unique_projects_req).index(project)
    
    # Add the value to the dictionary
    RSkl[(skill_index ,project_index )] = int(value)

# print(RSkl)

skills_M = Manager_skills_Avl.iloc[1:, 0]   # Skills column
Manager = Manager_skills_Avl.iloc[1:, 1]    # Managers column
values = Manager_skills_Avl.iloc[1:, 2]     # Values column

# Get unique managers and skills
unique_skills = skills_M.unique()
unique_managers = Manager.unique()

# Create the dictionary for storing availability values
MSAvl  = {}

# Populate the dictionary with values from the DataFrame
for i in range(len(skills_M)):
    skill = skills_M.iloc[i]
    manager = Manager.iloc[i]
    value = values.iloc[i]
    
    # Find the index of the current skill and manager
    skill_index = list(unique_skills).index(skill)
    manager_index = list(unique_managers).index(manager)
    
    # Add the value to the dictionary
    MSAvl[(skill_index, manager_index )] = int(value)


projects_dur = Duration_dict.iloc[1:, 0]   
values_dur = Duration_dict.iloc[1:, 1]  
Dur = {i: values_dur.iloc[i] for i in range(len(projects_dur))}

 
projects_sd = Due_Starting_Date.iloc[1:, 0]  # Project names from Due_Starting_Date
values_sd = Due_Starting_Date.iloc[1:, 1]  # Start dates from Due_Starting_Date
SD = {i: values_sd.iloc[i] for i in range(len(projects_sd))}



projects_fd = Due_Finishing_Date.iloc[1:, 0]  # Project names from Due_Starting_Date
values_fd = Due_Finishing_Date.iloc[1:, 1]  # Start dates from Due_Starting_Date
FD = {i: values_fd.iloc[i] for i in range(len(projects_fd))}

 

 
# WAD= WAD_sheet .iloc[1:, 0]  # Project names from Due_Starting_Date
# values_WAD = WAD_sheet.iloc[1:, 1]  # Start dates from Due_Starting_Date
# WAD = {i: values_WAD.iloc[i] for i in range(len(WAD))} 
# print("WAD values", WAD)

 
 
calendar_df_dropped = calendar_df.iloc[1:, 1:]
WAD_values = calendar_df_dropped.iloc[:, T].sum(axis=1).to_list() 
WAD = {i: WAD_values[i] for i in range(len(WAD_values))}
print("WAD values  :", WAD)
 

 
Score = {}
for w in W:
    score_sum = 0
    for t in T:
        score_sum += (len(T) - t + 1 - Cal[w, t]) * (1 - Cal[w, t])
    Score[w] = score_sum / len(T)
print("Workers Score", Score)


Cost=  Cost_sheet .iloc[1:, 0]  # Project names from Due_Starting_Date
values_Cost = Cost_sheet.iloc[1:, 1]  # Start dates from Due_Starting_Date
Cost = {i: values_Cost.iloc[i] for i in range(len(Cost))} 
 
# print("Workers Available Days",WAD)
# print("Workers Availablity Score",Score)
# print("Workers Cost per day",Cost)


# sys.exit()

print("Start Solving")

X =  {(p, w): solver.BoolVar(f'X[{p}][{w}]') for p in P for w in W}
ST = {(p): solver.IntVar(0, 500, f'ST[{p}]') for p in P }
FT = {(p): solver.IntVar(0, 500, f'FT[{p}]') for p in P }

# Constraints

for p in P:
    solver.Add(sum(X[p, w] for w in W) == 1)
     
for p in P:
    for w in W:
        for s in S:
            solver.Add( RSkl[s,p] - WSAvl[s, w] <= 2*(1-X[p, w]) )   

for p in P:
    solver.Add(ST[p] +Dur[p] <= FT[p])
    
for p in P:
    solver.Add(ST[p] >= SD[p]) 
 
for p in P:
    solver.Add(FT[p] <= FD[p]) 

for w in W:
      solver.Add(sum(X[p, w]* Dur[p] for p in P) <= WAD[w])
      
for p1 in P:
    for p2 in P:
      solver.Add(FT[p1]*Prec[p2,p1] <= ST[p2])

                                                  
weight_ft  = 0.5         
weight_score = 0.3           
weight_cal = 0.5     
weight_cost = 0.1    

# Objective 1: Minimize Sum(FT[p] for p in P)
objective1 = sum(FT[p] for p in P)

# Objective 2: Minimize Sum(X[p, w] * Score[w] for p in P for w in W)
objective2 = sum(X[p, w] * Score[w] for p in P for w in W)

# Objective 3: Maximize Sum(X[p, w] * WAD[w] for p in P for w in W) (Maximization converted to minimization by negating)
objective3 = sum(X[p, w] *(1- Cal[w,t]) for p in P for w in W for t in T)

# Objective 4: Minimize Sum(X[p, w] * Cost[w] for p in P for w in W)
objective4 = sum(X[p, w] * Cost[w] for p in P for w in W)

# Define the combined objective with relative weights
combined_objective = (weight_ft  * objective1 +
                      weight_score * objective2 +
                      weight_cal * objective3 +
                      weight_cost * objective4)

# Minimize the combined objective
solver.Minimize(combined_objective)
status = solver.Solve()

# Check the status and extract the results
if status == pywraplp.Solver.OPTIMAL:
    print('Solution found!')
    # Extract the solution values
    for p in P:
        for w in W:
                if X[p, w].solution_value() == 1:    
                    print(f'Project {p+1}, Worker {w+1}')
    for p in P:
       print(f'Project {p+1}, ST: {ST[p].solution_value()}, FT: {FT[p].solution_value()}, Due date: {SD[p]}, Dur: {Dur[p]}')
        
    print('Objective value =', solver.Objective().Value())
else:
    print('No optimal solution found.')

print("Start Heuristics") 

# Initialize dictionaries for adjusted start and finish times
adjusted_ST = {}
adjusted_FT = {}

# Initialize a dictionary to track the latest finish time for each worker
latest_finish_time = {w: 0 for w in W}  # For each worker, store their latest finish time

for p in P:
    # Find the worker assigned to project p
    for w in W:
        # Ensure we use the value of x[p, w] correctly (as a Python boolean check)
        if X[p, w].solution_value() == 1:  # Get the assigned worker from the solver's solution value
            assigned_worker = w  # Get the assigned worker for this project
            break  # Stop once the worker is found

    # Get the required start time and duration from the project variables
    required_start_time = ST[p].solution_value()  # Extract solution values for start time
    duration = Dur[p]  # Duration of the project is a parameter, so it's directly accessible
    available_days = 0  # Counter for available workdays

    # Assign the earliest possible start time based on the finish time of the last project
    earliest_possible_start = max(latest_finish_time[assigned_worker], required_start_time)

    # Start checking the worker's calendar from the earliest possible start time
    t = earliest_possible_start

    # Find the first available day on or after the earliest possible start time
    while Cal[assigned_worker, t] == 0:  # If it's an off day, move to the next day
        t += 1

    # Set the adjusted start time to the first available day
    adjusted_ST[p] = t

    # Now, calculate the finish time based on worker availability and project duration
    while available_days < duration:  # Work for the required number of available days
        if Cal[assigned_worker, t] == 1:  # If the worker is available on day t
            available_days += 1  # Increment available workdays
        t += 1  # Move to the next time period

    # The finish time is the last day the worker is available for the full duration
    adjusted_FT[p] = t - 1  # t-1 because t was incremented after the last workday

    # Update the latest finish time for the assigned worker so the next project starts after this
    latest_finish_time[assigned_worker] = adjusted_FT[p]

# Output adjusted start and finish times
for p in P:
    print(f"Project {p+1}: Adjusted ST: {adjusted_ST[p]}, Adjusted FT: {adjusted_FT[p]} ")
 
 
import datetime

# Define current date as the base for ST[1]
current_date = datetime.datetime.now().date()

# Assuming adjusted_ST and adjusted_FT contain the start and finish times as integers (days offset from ST[1])
# Convert those to dates
adjusted_ST_dates = {p: current_date + datetime.timedelta(days=adjusted_ST[p]) for p in P}
adjusted_FT_dates = {p: current_date + datetime.timedelta(days=adjusted_FT[p]) for p in P}

# Create the DataFrame with the updated dates
data = {
    "Project": [f"Project {p+1}" for p in P],
    # Extract the worker assigned to each project by checking the solution value of X[p, w]
    "Worker Assigned": [
        next((w +1 for w in W if X[p, w].solution_value() == 1), "No Worker Assigned") for p in P
    ],
    "Adjusted ST": [adjusted_ST_dates.get(p, "No ST") for p in P],  # Start date as date
    "Adjusted FT": [adjusted_FT_dates.get(p, "No FT") for p in P] ,  # Finish date as date
    "Project Duration": [Dur[p] for p in P]
}

df = pd.DataFrame(data)

def get_skills_for_project(p):
    skills = [f"Skill {s+1}" for s in range(len(unique_skills)) if RSkl.get((s, p), 0) == 1]
    return ", ".join(skills) if skills else "No Skills"

df["Skills Required"] = [get_skills_for_project(p) for p in P]



# Save to Excel
file_path = 'project_schedule_with_dates.xlsx'
df.to_excel(file_path, index=False)

print(f"Excel file saved at: {file_path}")

# Graphs
                                # WAD vs Available Time
Sum_X_Dur_list = []
for w in W:
        Sum_X_Dur = 0
        for p in P:
            # Retrieve the value of the decision variable X[p, w] from the solution
            if X[(p, w)].solution_value() == 1:
                Sum_X_Dur += X[(p, w)].solution_value() * Dur[p]
                # print(f'X[{p},{w}] * Dur[{p}] = {X[(p, w)].solution_value()} * {Dur[p]} = {X[(p, w)].solution_value() * Dur[p]}')
        Sum_X_Dur_list.append(Sum_X_Dur)
print("Total duration assigned to each worker:", Sum_X_Dur_list)
                            
WAD_list = [WAD[w] for w in W if w in WAD]
print(WAD_list)

workers = np.arange(len(WAD_list))

# Set bar width
bar_width = 0.35

# Create a figure and axis with increased size
fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size

# Plot the available time (WAD) as the first bar for each worker
bar1 = ax.bar(workers, WAD_list, bar_width, label='Available Time (WAD)')

# Plot the utilized time (Sum_X_Dur) as the second bar for each worker, slightly offset to the right
bar2 = ax.bar(workers + bar_width, Sum_X_Dur_list, bar_width, label='Utilized Time')

# Set labels and title
ax.set_xlabel('Workers')
ax.set_ylabel('Time (in days)')
ax.set_title('Comparison of Available vs Utilized Time for Each Worker')

# Set x-axis ticks at the center of the bars and label them as workers
ax.set_xticks(workers + bar_width / 2)
ax.set_xticklabels([f'W {i+1}' for i in workers], rotation=45, ha="right")

# Adjust the layout to add more space between elements
plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
 
# Add values on top of each bar (for available time)
for i, v in enumerate(WAD_list):
    ax.text(i - 0.05, v + 0.5, str(v), color='blue', fontweight='bold')

# Add values on top of each bar (for utilized time)
for i, v in enumerate(Sum_X_Dur_list):
    ax.text(i + bar_width - 0.05, v + 0.5, str(v), color='orange', fontweight='bold')
    
# Display the plot
plt.tight_layout()
plt.show()

                              # Cost per Worker

Cost_W = []
for w in W:
        Cost_Project = 0
        for p in P:
            if X[(p, w)].solution_value() == 1:
                Cost_Project += X[(p, w)].solution_value() *Cost[w]*Dur[p]
        Cost_W.append(Cost_Project)
print("Total Cost to each worker:", Cost_W)

Cost_P = []
for p in P:
        Cost_Project = 0
        for w in W:
              if X[(p, w)].solution_value() == 1:
                Cost_Project += X[(p, w)].solution_value() *Cost[w]*Dur[p]
        Cost_P.append(Cost_Project)
print("Total Cost to each project:", Cost_P)


fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar graph
ax.bar(W, Cost_W, color='#76C7C0', edgecolor='blue')

# Set labels and title
ax.set_xlabel('Workers')
ax.set_ylabel('Total Cost')
ax.set_title('Total Cost per Worker')

# Set x-axis ticks and labels for workers
ax.set_xticks(np.arange(len(W)))
ax.set_xticklabels([f'Worker {i+1}' for i in W])

# Add total cost labels on top of each bar
for i, total_cost in enumerate(Cost_W):
    ax.text(i, total_cost + 2, f'{total_cost}', ha='center', fontweight='bold')

# Display the plot
plt.tight_layout()
plt.show()


                                # Cost per Project
                                
fig, ax = plt.subplots(figsize=(12, 6))   

# Create the bar graph
ax.bar(P, Cost_P, color='#FFBB28', edgecolor='blue')

# Set labels and title
ax.set_xlabel('Projects')
ax.set_ylabel('Cost')
ax.set_title('Cost per Project')

# Rotate x-axis labels for better spacing
ax.set_xticks(np.arange(len(P)))
ax.set_xticklabels([f'Project {i+1}' for i in P], rotation=45, ha="right")

# Add cost labels on top of each bar
for i, cost in enumerate(Cost_P):
    ax.text(i, cost + 1, f'{cost}', ha='center', fontweight='bold')

# Add padding to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
                                # Worker Assignment

for p in P:
    for w in W:
        if X[p, w].solution_value() == 1:  
            print(f"X[{p+1}, {w+1}] = 1")

x_values = []
y_values = []            
for p in P:
    for w in W:
        if X[p, w].solution_value() == 1:  # Only include assignments where X[p, w] is 1
            x_values.append(w)  # Worker on x-axis
            y_values.append(p)  # Project on y-axis

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', marker='o')

# Set labels and title
plt.xlabel('Workers')
plt.ylabel('Projects')
plt.title('Project Assignment to Worker')

# Set x-axis and y-axis ticks
plt.xticks(W, [f'Worker {w+1}' for w in W])
plt.yticks(P, [f'Project {p+1}' for p in P])

# Show the plot            
plt.tight_layout()
plt.show()
                                          # Gantt Chart

adjusted_ST_dates = {p: current_date + datetime.timedelta(days=adjusted_ST[p]) for p in P}
adjusted_FT_dates = {p: current_date + datetime.timedelta(days=adjusted_FT[p]) for p in P}

gantt_data = []
for p in P:
    gantt_data.append({
        'Project': p,
        'Start': adjusted_ST_dates[p],
        'Finish': adjusted_FT_dates[p]
    })

# Convert to DataFrame for plotting
df = pd.DataFrame(gantt_data)

# Create Gantt chart
fig, ax = plt.subplots(figsize=(10, 5))

# Plot each project as a horizontal bar
for index, row in df.iterrows():
    ax.barh(row['Project'], (row['Finish'] - row['Start']).days, left=row['Start'], align='center')

# Format the x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=45)

# Set y-axis as sequential numbers
ax.set_yticks(range(1, len(P)+1))  # Sequential numbering from 1 to the number of projects
ax.set_ylabel('Project No')

# Labels and title
ax.set_xlabel('Time')
ax.set_title('Optimized Gantt Chart')

plt.tight_layout()
plt.show()