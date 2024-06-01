#!/usr/bin/env python
# coding: utf-8

#getting the weather:
import requests
import json

API_KEY = 'your_api_key_here'
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

def get_weather(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

if __name__ == "__main__":
    city = input("Enter city name: ")
    weather_data = get_weather(city)
    if weather_data:
        print(json.dumps(weather_data, indent=4))
    else:
        print("Failed to get weather data.")

if __name__ == "__main__":def display_weather(data):
    print(f"Weather in {data['name']}, {data['sys']['country']}:")
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Weather: {data['weather'][0]['description']}")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Wind Speed: {data['wind']['speed']} m/s")

if __name__ == "__main__":
    city = input("Enter city name: ")
    weather_data = get_weather(city)
    if weather_data:
        display_weather(weather_data)
    else:
        print("Failed to get weather data.")

    main()

def save_weather(data, filename='weather_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_weather(filename='weather_data.json'):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    city = input("Enter city name: ")
    weather_data = get_weather(city)
    if weather_data:
        display_weather(weather_data)
        save_weather(weather_data)
    else:
        print("Failed to get weather data.")


import matplotlib.pyplot as plt

def plot_weather(data):
    temps = [data['main']['temp']]
    humidity = [data['main']['humidity']]
    labels = [data['name']]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(labels, temps, color='blue')
    plt.title('Temperature (°C)')

    plt.subplot(1, 2, 2)
    plt.bar(labels, humidity, color='green')
    plt.title('Humidity (%)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    city = input("Enter city name: ")
    weather_data = get_weather(city)
    if weather_data:
        display_weather(weather_data)
        save_weather(weather_data)
        plot_weather(weather_data)
    else:
        print("Failed to get weather data.")


