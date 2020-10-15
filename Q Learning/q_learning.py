
# importing the libraries 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# selecting all the connected points

points = [(0,1), (1,6), (2,6), (3,4), (4,7),
          (5,8), (6,8), (6,17), (6,7), 
          (7,12), (7,17), (7,9), (8,11),
          (10,11), (11,15),(12,15), (12,16),
          (14,15),(12,13)]
 
# the target location

end_location = 17

# creating network graph according to the points 

G = nx.Graph()
G.add_edges_from(points)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()

# creating the reward matrix

MATRIX_SIZE = 18

r_matrix = np.matrix(np.ones((MATRIX_SIZE, MATRIX_SIZE)))
r_matrix = r_matrix*-1

# assigning the reward values for each action

for point in points:
    
    if point[1] == end_location:
        r_matrix[point] = 100
    
    else:
        r_matrix[point] = 0
        
    if point[0] == end_location:
        r_matrix[point[::-1]] = 100
        
    else:
        r_matrix[point[::-1]] = 0
        
r_matrix[end_location, end_location] =100

# creating the q_table ( Zero matrix)

q_table = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# gamma value

gamma = 0.85

# function for finding the available actions in a given state

def available_actions(state):
    actions_row = r_matrix[state,]
    actions = np.where(actions_row >= 0)[1]
    return actions

# function for selecting the next action
    
def next_action(available_act):
    next_act = int(np.random.choice(available_act,1))
    return next_act

# function for updating q_matrix
    
def update_q(current_state, action, gamma):
    max_index = np.where(q_table[action,] == np.max(q_table[action,]))[1]
    
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, 1))
        
    else:
        max_index = int(max_index)
        
    max_value = q_table[action, max_index]
    
    q_table[current_state, action] = r_matrix[current_state, action] + gamma*max_value
    
    
    
# training 
    
for i in range(100000):
    
    current_state = int(np.random.randint(0,18))
    available_act = available_actions(current_state)
    action = next_action(available_act)
    update_q(current_state, action, gamma)
reduced_q = (q_table/np.max(q_table) * 100)

print(reduced_q)


# testing

current_state = 14
path = [current_state]


while current_state != 17:
    next_act = np.where(q_table[current_state,] == np.max(q_table[current_state,]))[1]
    
    if next_act.shape[0] > 1:
        
        next_act = int(np.random.choice(next_act, 1))
    
    else:
        next_act = int(next_act)
        
    path.append(next_act)
    
    current_state = next_act
    

# printing the path

print(path)  # [14, 15, 12, 7, 17]
    

