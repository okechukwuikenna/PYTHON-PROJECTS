#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize the main window
root = tk.Tk()
root.title("Personal Budget Tracker")

# Create DataFrame to store the data
columns = ["Date", "Category", "Amount", "Type"]
try:
    df = pd.read_csv("budget_data.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)

# Function to add income or expense
def add_transaction():
    date = entry_date.get()
    category = entry_category.get()
    amount = entry_amount.get()
    ttype = var_type.get()

    if date and category and amount and ttype:
        try:
            amount = float(amount)
            new_row = pd.DataFrame([[date, category, amount, ttype]], columns=columns)
            global df
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("budget_data.csv", index=False)
            messagebox.showinfo("Success", "Transaction added successfully!")
            entry_date.delete(0, tk.END)
            entry_category.delete(0, tk.END)
            entry_amount.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid amount.")
    else:
        messagebox.showerror("Error", "Please fill in all fields.")

# Function to show summary
def show_summary():
    global df
    if df.empty:
        messagebox.showinfo("Info", "No transactions to display.")
        return

    income = df[df['Type'] == 'Income']['Amount'].sum()
    expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    balance = income - expenses

    summary_text = f"Total Income: ${income}\nTotal Expenses: ${expenses}\nBalance: ${balance}"
    messagebox.showinfo("Summary", summary_text)

# Function to show visualization
def show_visualization():
    global df
    if df.empty:
        messagebox.showinfo("Info", "No transactions to display.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    monthly_data = df.groupby([df['Date'].dt.to_period('M'), 'Type'])['Amount'].sum().unstack().fillna(0)
    monthly_data.plot(kind='bar', stacked=True)
    plt.title("Monthly Income and Expenses")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.show()

# Create GUI elements
label_date = tk.Label(root, text="Date (YYYY-MM-DD):")
label_date.grid(row=0, column=0, padx=10, pady=10)
entry_date = tk.Entry(root)
entry_date.grid(row=0, column=1, padx=10, pady=10)

label_category = tk.Label(root, text="Category:")
label_category.grid(row=1, column=0, padx=10, pady=10)
entry_category = tk.Entry(root)
entry_category.grid(row=1, column=1, padx=10, pady=10)

label_amount = tk.Label(root, text="Amount:")
label_amount.grid(row=2, column=0, padx=10, pady=10)
entry_amount = tk.Entry(root)
entry_amount.grid(row=2, column=1, padx=10, pady=10)

var_type = tk.StringVar()
var_type.set("Income")
radio_income = tk.Radiobutton(root, text="Income", variable=var_type, value="Income")
radio_income.grid(row=3, column=0, padx=10, pady=10)
radio_expense = tk.Radiobutton(root, text="Expense", variable=var_type, value="Expense")
radio_expense.grid(row=3, column=1, padx=10, pady=10)

button_add = tk.Button(root, text="Add Transaction", command=add_transaction)
button_add.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

button_summary = tk.Button(root, text="Show Summary", command=show_summary)
button_summary.grid(row=5, column=0, padx=10, pady=10)

button_visualize = tk.Button(root, text="Show Visualization", command=show_visualization)
button_visualize.grid(row=5, column=1, padx=10, pady=10)

# Run the main loop
root.mainloop()

