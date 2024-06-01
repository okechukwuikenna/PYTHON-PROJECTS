#!/usr/bin/env python
# coding: utf-8

import json

# File to store contact information
CONTACTS_FILE = "contacts.json"

# Load contacts from file
def load_contacts():
    try:
        with open(CONTACTS_FILE, "r") as file:
            contacts = json.load(file)
    except FileNotFoundError:
        contacts = {}
    return contacts

# Save contacts to file
def save_contacts(contacts):
    with open(CONTACTS_FILE, "w") as file:
        json.dump(contacts, file)

# Add a new contact
def add_contact(contacts, name, phone):
    contacts[name] = phone
    save_contacts(contacts)
    print(f"Contact '{name}' added successfully.")

# Edit an existing contact
def edit_contact(contacts, name, new_phone):
    if name in contacts:
        contacts[name] = new_phone
        save_contacts(contacts)
        print(f"Contact '{name}' edited successfully.")
    else:
        print(f"Contact '{name}' not found.")

# Delete a contact
def delete_contact(contacts, name):
    if name in contacts:
        del contacts[name]
        save_contacts(contacts)
        print(f"Contact '{name}' deleted successfully.")
    else:
        print(f"Contact '{name}' not found.")

# Search for a contact
def search_contact(contacts, name):
    if name in contacts:
        print(f"Name: {name}, Phone: {contacts[name]}")
    else:
        print(f"Contact '{name}' not found.")

# Main function to interact with the contact book
def main():
    contacts = load_contacts()

    while True:
        print("\nContact Book")
        print("1. Add Contact")
        print("2. Edit Contact")
        print("3. Delete Contact")
        print("4. Search Contact")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter contact name: ")
            phone = input("Enter contact phone number: ")
            add_contact(contacts, name, phone)
        elif choice == "2":
            name = input("Enter contact name to edit: ")
            new_phone = input("Enter new phone number: ")
            edit_contact(contacts, name, new_phone)
        elif choice == "3":
            name = input("Enter contact name to delete: ")
            delete_contact(contacts, name)
        elif choice == "4":
            name = input("Enter contact name to search: ")
            search_contact(contacts, name)
        elif choice == "5":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

