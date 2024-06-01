#!/usr/bin/env python
# coding: utf-8

#menu order
import tkinter as tk  # Importing tkinter
from tkinter import messagebox  # For showing message boxes

# Define the menu with food items and prices
menu = {
    "Burger": 8.99,
    "Pizza": 12.99,
    "Salad": 7.49,
    "Sandwich": 6.99,
    "Soda": 1.99,
}

# Function to calculate the total cost based on selections
def calculate_total(selections):
    total = sum(menu[item] * quantity for item, quantity in selections.items())
    return total

# Callback function for placing an order
def place_order():
    # Get the selected items and their quantities
    selections = {item: int(quantity_var[item].get()) for item in menu if int(quantity_var[item].get()) > 0}

    if not selections:
        messagebox.showinfo("No Order", "No items selected for ordering!")
        return

    # Calculate the total
    total = calculate_total(selections)

    # Create the order summary
    order_summary = "\n".join([f"{item}: {quantity} x ${menu[item]:.2f}" for item, quantity in selections.items()])
    order_summary += f"\n\nTotal: ${total:.2f}"

    # Confirm the order
    response = messagebox.askyesno("Confirm Order", f"Your order:\n\n{order_summary}\n\nDo you want to confirm this order?")
    
    if response:
        messagebox.showinfo("Order Confirmed", "Thank you for your order!")
    else:
        messagebox.showinfo("Order Cancelled", "Your order has been cancelled.")

# Create the main window
root = tk.Tk()
root.title("Food Ordering App")

# Create a frame for the menu
menu_frame = tk.Frame(root)
menu_frame.pack(pady=10)

# Add labels for the menu items and entry fields for quantities
quantity_var = {}  # Dictionary to hold the quantity variables
for item in menu:
    frame = tk.Frame(menu_frame)
    frame.pack(fill="x")

    label = tk.Label(frame, text=f"{item}: ${menu[item]:.2f}")
    label.pack(side="left", padx=10)

    quantity_var[item] = tk.StringVar(value="0")  # Default to 0 quantity
    entry = tk.Entry(frame, textvariable=quantity_var[item], width=5)
    entry.pack(side="left")

# Create a button to place the order
order_button = tk.Button(root, text="Place Order", command=place_order)
order_button.pack(pady=10)

# Start the tkinter main loop
root.mainloop()


