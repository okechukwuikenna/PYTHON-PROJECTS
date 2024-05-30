#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Number guess game
import random

def main():
    number_to_guess = random.randint(1, 100)
    attempts = 0

    print("Welcome to the Guess the Number game!")
    print("I have chosen a number between 1 and 100. Try to guess it.")

    while True:
        guess = input("Enter your guess (or 'q' to quit): ")

        if guess.lower() == 'q':
            print("Quitting the game. Goodbye!")
            break

        try:
            guess = int(guess)
            attempts += 1

            if guess < number_to_guess:
                print("Too low! Try again.")
            elif guess > number_to_guess:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number {number_to_guess} in {attempts} attempts.")
                break
        except ValueError:
            print("Invalid input! Please enter a number or 'q' to quit.")

if __name__ == "__main__":
    main()


