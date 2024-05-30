#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Room:
    def __init__(self, name, length, width, height, purpose):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.purpose = purpose
        self.utilities = []
    
    def add_utility(self, utility):
        self.utilities.append(utility)
    
    def area(self):
        return self.length * self.width
    
    def volume(self):
        return self.length * self.width * self.height

class Floor:
    def __init__(self, level):
        self.level = level
        self.rooms = []
    
    def add_room(self, room):
        self.rooms.append(room)
    
    def total_area(self):
        return sum(room.area() for room in self.rooms)
    
    def total_volume(self):
        return sum(room.volume() for room in self.rooms)

class Building:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.floors = []
    
    def add_floor(self, floor):
        self.floors.append(floor)
    
    def total_area(self):
        return sum(floor.total_area() for floor in self.floors)
    
    def total_volume(self):
        return sum(floor.total_volume() for floor in self.floors)
    
    def building_summary(self):
        summary = f"Building Name: {self.name}\nAddress: {self.address}\n"
        summary += f"Total Floors: {len(self.floors)}\n"
        summary += f"Total Area: {self.total_area()} sq units\n"
        summary += f"Total Volume: {self.total_volume()} cubic units\n"
        return summary
class Utility:
    def __init__(self, name, type):
        self.name = name
        self.type = type

# Examples of utilities
electricity = Utility("Electricity", "Electrical")
water = Utility("Water", "Plumbing")
internet = Utility("Internet", "Communication")
# Create rooms
living_room = Room("Living Room", 5, 7, 3, "Living")
kitchen = Room("Kitchen", 4, 4, 3, "Cooking")
bedroom = Room("Bedroom", 4, 5, 3, "Sleeping")
bathroom = Room("Bathroom", 2, 3, 3, "Bathing")

# Add utilities to rooms
living_room.add_utility(electricity)
living_room.add_utility(internet)
kitchen.add_utility(electricity)
kitchen.add_utility(water)
bedroom.add_utility(electricity)
bathroom.add_utility(electricity)
bathroom.add_utility(water)

# Create floors and add rooms to floors
ground_floor = Floor(0)
ground_floor.add_room(living_room)
ground_floor.add_room(kitchen)

first_floor = Floor(1)
first_floor.add_room(bedroom)
first_floor.add_room(bathroom)

# Create building and add floors to building
building = Building("Complex Building", "1234 Python Lane")
building.add_floor(ground_floor)
building.add_floor(first_floor)

# Print building summary
print(building.building_summary())


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_building(building):
    fig, axes = plt.subplots(len(building.floors), 1, figsize=(10, 5 * len(building.floors)))
    
    if len(building.floors) == 1:
        axes = [axes]
    
    for ax, floor in zip(axes, building.floors):
        ax.set_title(f'Floor {floor.level}')
        ax.set_xlim(0, 20)  # Adjust limits as necessary
        ax.set_ylim(0, 20)
        ax.set_aspect('equal')
        ax.grid(True)
        
        current_x = 0
        current_y = 0
        max_y = 0
        
        for room in floor.rooms:
            rect = patches.Rectangle((current_x, current_y), room.length, room.width, linewidth=1, edgecolor='black', facecolor='lightgray')
            ax.add_patch(rect)
            ax.text(current_x + room.length / 2, current_y + room.width / 2, room.name, ha='center', va='center')
            
            current_x += room.length
            max_y = max(max_y, current_y + room.width)
        
        ax.set_xlim(0, current_x)
        ax.set_ylim(0, max_y)
    
    plt.tight_layout()
    plt.show()

# Create rooms
living_room = Room("Living Room", 7, 5, 3, "Living")
kitchen = Room("Kitchen", 4, 4, 3, "Cooking")
bedroom = Room("Bedroom", 5, 4, 3, "Sleeping")
bathroom = Room("Bathroom", 3, 2, 3, "Bathing")

# Add utilities to rooms
living_room.add_utility(electricity)
living_room.add_utility(internet)
kitchen.add_utility(electricity)
kitchen.add_utility(water)
bedroom.add_utility(electricity)
bathroom.add_utility(electricity)
bathroom.add_utility(water)

# Create floors and add rooms to floors
ground_floor = Floor(0)
ground_floor.add_room(living_room)
ground_floor.add_room(kitchen)

first_floor = Floor(1)
first_floor.add_room(bedroom)
first_floor.add_room(bathroom)

# Create building and add floors to building
building = Building("Complex Building", "1234 Python Lane")
building.add_floor(ground_floor)
building.add_floor(first_floor)

# Visualize the building
plot_building(building)


# In[ ]:




