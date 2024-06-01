#!/usr/bin/env python
# coding: utf-8

#Creating a simple Point of Sale (POS) machine payment system using Python involves several key components#
class Product:
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

class Inventory:
    def __init__(self):
        self.products = {}

    def add_product(self, product):
        self.products[product.name] = product

    def get_product(self, name):
        return self.products.get(name)

    def update_quantity(self, name, quantity):
        if name in self.products:
            self.products[name].quantity -= quantity

# Example usage
if __name__ == "__main__":
    inventory = Inventory()
    inventory.add_product(Product('Apple', 0.5, 100))
    inventory.add_product(Product('Banana', 0.3, 150))

    product = inventory.get_product('Apple')
    if product:
        print(f"Product: {product.name}, Price: {product.price}, Quantity: {product.quantity}")


class Cart:
    def __init__(self):
        self.items = []

    def add_item(self, product, quantity):
        self.items.append((product, quantity))

    def calculate_total(self):
        return sum(product.price * quantity for product, quantity in self.items)

    def generate_receipt(self):
        receipt = "\nReceipt:\n"
        receipt += "----------------------------\n"
        for product, quantity in self.items:
            receipt += f"{product.name} x {quantity} = ${product.price * quantity:.2f}\n"
        receipt += "----------------------------\n"
        receipt += f"Total: ${self.calculate_total():.2f}\n"
        return receipt

# Example usage
if __name__ == "__main__":
    inventory = Inventory()
    inventory.add_product(Product('Apple', 0.5, 100))
    inventory.add_product(Product('Banana', 0.3, 150))

    cart = Cart()
    cart.add_item(inventory.get_product('Apple'), 3)
    cart.add_item(inventory.get_product('Banana'), 2)

    print(cart.generate_receipt())


class Payment:
    @staticmethod
    def process_cash_payment(total, cash_given):
        if cash_given >= total:
            change = cash_given - total
            return True, change
        else:
            return False, 0

    @staticmethod
    def process_card_payment(total):
        # For simplicity, assume card payments are always successful
        return True

# Example usage
if __name__ == "__main__":
    inventory = Inventory()
    inventory.add_product(Product('Apple', 0.5, 100))
    inventory.add_product(Product('Banana', 0.3, 150))

    cart = Cart()
    cart.add_item(inventory.get_product('Apple'), 3)
    cart.add_item(inventory.get_product('Banana'), 2)

    total = cart.calculate_total()
    print(cart.generate_receipt())

    # Cash payment
    success, change = Payment.process_cash_payment(total, 5.0)
    if success:
        print(f"Payment successful. Change: ${change:.2f}")
    else:
        print("Payment failed. Insufficient cash.")

    # Card payment
    if Payment.process_card_payment(total):
        print("Payment successful.")
    else:
        print("Payment failed.")


def main():
    inventory = Inventory()
    inventory.add_product(Product('Apple', 0.5, 100))
    inventory.add_product(Product('Banana', 0.3, 150))
    inventory.add_product(Product('Orange', 0.7, 80))

    cart = Cart()

    while True:
        print("\n1. Add item to cart")
        print("2. View cart and checkout")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            product_name = input("Enter product name: ")
            product = inventory.get_product(product_name)
            if product:
                quantity = int(input("Enter quantity: "))
                if quantity <= product.quantity:
                    cart.add_item(product, quantity)
                    inventory.update_quantity(product_name, quantity)
                    print(f"Added {quantity} x {product_name} to cart.")
                else:
                    print("Insufficient quantity in stock.")
            else:
                print("Product not found.")
        
        elif choice == '2':
            if not cart.items:
                print("Cart is empty.")
            else:
                print(cart.generate_receipt())
                total = cart.calculate_total()
                print(f"Total: ${total:.2f}")
                payment_method = input("Enter payment method (cash/card): ").strip().lower()

                if payment_method == 'cash':
                    cash_given = float(input("Enter cash amount: "))
                    success, change = Payment.process_cash_payment(total, cash_given)
                    if success:
                        print(f"Payment successful. Change: ${change:.2f}")
                        break
                    else:
                        print("Payment failed. Insufficient cash.")
                elif payment_method == 'card':
                    if Payment.process_card_payment(total):
                        print("Payment successful.")
                        break
                    else:
                        print("Payment failed.")
                else:
                    print("Invalid payment method.")
        
        elif choice == '3':
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

