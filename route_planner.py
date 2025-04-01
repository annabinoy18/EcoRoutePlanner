import googlemaps
import torch
import numpy as np
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.2:5500"])  # ✅ Enable CORS for all routes

# ✅ Load the trained DQN model
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Load trained model
model = DQN(4, 3)  # 4 input features: (distance, battery, range, traffic_delay), 3 actions
model.load_state_dict(torch.load("dqn_ev_route.pth"))
model.eval()

# ✅ Google Maps API Key (Replace with a secure method)
gmaps = googlemaps.Client(key="AIzaSyChLb_13Z8w_KwTuI0mntk1toqt5PNrQ0Y")

def get_routes(start, end):
    """Fetch possible routes from Google Maps Directions API."""
    directions = gmaps.directions(start, end, mode="driving", departure_time=datetime.now())
    return directions

def extract_features(route, battery, range_km):
    """Extract required route features and combine with EV state info."""
    distance = route["legs"][0]["distance"]["value"] / 1000  # Convert meters to km
    duration = route["legs"][0]["duration"]["value"] / 60  # Convert seconds to minutes
    traffic_delay = route["legs"][0].get("duration_in_traffic", {}).get("value", 0) / 60
    return np.array([distance, battery, range_km, traffic_delay], dtype=np.float32)

def choose_best_route(routes, battery, range_km):
    """Use the DQN model to select the most energy-efficient route."""
    best_route = None
    best_score = -np.inf
    for route in routes:
        features = extract_features(route, battery, range_km)
        state = torch.FloatTensor(features).unsqueeze(0)
        score = model(state).max().item()
        if score > best_score:
            best_score = score
            best_route = route
    return best_route

@app.route('/plan_route', methods=['POST'])
def plan_route():
    data = request.json
    print("Received request data:", data)  # Debugging statement
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400
    start = (data['startLatitude'], data['startLongitude'])
    end = (data['destLatitude'], data['destLongitude'])
    battery = data['battery']
    range_km = data['range']

    routes = get_routes(start, end)
    if not routes:
        return jsonify({"error": "No routes found!"}), 400

    best_route = choose_best_route(routes, battery, range_km)
    
    # ✅ Extract route details for frontend
    route_data = {
        "distance": best_route["legs"][0]["distance"]["value"] / 1000,
        "duration": best_route["legs"][0]["duration"]["text"],
        "battery_usage": (best_route["legs"][0]["distance"]["value"] / 1000) / range_km * 100,
        "steps": [
            {
                "start_location": step["start_location"],
                "end_location": step["end_location"],
                "instructions": step["html_instructions"],
            }
            for step in best_route["legs"][0]["steps"]
        ]
    }

    print("Route Data being sent to frontend:")  # ✅ Debugging print statement
    print(route_data)  # ✅ Debugging print statement

    return jsonify(route_data)

if __name__ == "__main__":
    app.run(debug=True)
