# ⚡ EcoRoute Planner

**EcoRoute Planner** is a smart web application that helps Electric Vehicle (EV) users find the most energy-efficient route to their destination while showing charging stations along the way. It takes user inputs like start and destination locations (place names), battery level, and EV range, and intelligently recommends a route with charging stations shown both on a map and in card format.

---

## 🚀 Features

- 🔍 Enter location names instead of coordinates  
- 🔋 Input current battery level and vehicle range  
- 🗺️ Route planning using Google Maps Directions API  
- ⚡ Real-time EV charging station data using OpenStreetMao API  
- 📍 Geocoding place names to coordinates via Google Geocoding API  
- 🧠 Route optimization based on energy efficiency (DQN model)  
- 🖼️ Charging station info shown on an interactive Leaflet map  
- 🧾 Clean card view under the map displaying station details  
- ✅ No database required — works with just frontend + backend  

---

## 📦 Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Leaflet.js  
- **Backend**: Python (Flask or FastAPI)  
- **APIs Used**:
  - Google Maps Directions API  
  - Google Maps Geocoding API  
  - OpenStreetMap API  
- **AI/ML**: Deep Q-Learning (DQN) for route optimization  

---

## 🛠️ How It Works

1. **User Inputs:**
   - Start Location (e.g., “Bangalore”)
   - Destination Location (e.g., “Mysore”)
   - Current battery level (in %)
   - EV range on full charge (in km)

2. **Behind the Scenes:**
   - Place names are converted to coordinates using Geocoding API
   - The route is generated using the Google Directions API
   - Charging stations are fetched around those points 
   - Details are shown as markers on the map and as cards below

3. **Output:**
   - Optimized route with map visualization
   - Charging station details (title, address, distance) in a list of cards

---

## 📍 Sample Use Case

> Start: **San Francisco**  
> Destination: **Los Angeles**  
> Battery: **60%**  
> Range: **300 km**  
✅ The app returns a route with EV chargers mapped and detailed for stops like San Jose, Monterey, and Bakersfield.

---

## 📷 UI Highlights

- Interactive Leaflet map with green markers for charging stations
- Card layout below the map with:
  - 📌 Title of the station  
  - 📍 Address  
  - 📏 Distance from route  

---

