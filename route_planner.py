import googlemaps
import torch
import numpy as np
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# ✅ Load the trained DQN model
class DuelingDQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        
        # Value Stream
        self.value_stream = torch.nn.Linear(64, 1)
        # Advantage Stream
        self.advantage_stream = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = self.value_stream(x)  # Scalar value for the state
        advantage = self.advantage_stream(x)  # Advantage for each action
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean())
        return q_values

# ✅ Load trained model
model = DuelingDQN(4, 3)  # 4 input features: (distance, battery, range, traffic_delay), 3 actions
model.load_state_dict(torch.load("dqn_ev_route.pth"))
model.eval()

# ✅ Load Google Maps API Key securely
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAP_API_KEY"  # Set this in your environment variables
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_routes(start, end):
    """Fetch possible routes from Google Maps Directions API, ensuring valid roads."""
    directions = gmaps.directions(
        start,
        end,
        mode="driving",
        departure_time=datetime.now(),
        avoid=["ferries", "indoor", "unpaved"]  # ✅ Avoid invalid roads
    )
    return directions if directions else None

def extract_features(route, battery, range_km):
    """Extract required route features and combine with EV state info."""
    leg = route["legs"][0]  
    distance = leg["distance"]["value"] / 1000  # Convert meters to km
    duration = leg["duration"]["value"] / 60  # Convert seconds to minutes
    traffic_delay = leg.get("duration_in_traffic", {}).get("value", 0) / 60  # Fallback to 0 if missing
    return np.array([distance, battery, range_km, traffic_delay], dtype=np.float32)

def choose_best_route(routes, battery, range_km):
    """Use the DQN model to select the most energy-efficient route."""
    if not routes:
        return None

    best_route = None
    best_score = -np.inf
    for route in routes:
        features = extract_features(route, battery, range_km)
        state = torch.FloatTensor(features).unsqueeze(0)
        score = model(state).max().item()
        if score > best_score:
            best_score = score
            best_route = route
    return best_route if best_route else routes[0]  # ✅ Fallback to first valid route

@app.route('/plan_route', methods=['POST'])
def plan_route():
    try:
        data = request.json
        print("Received request data:", data)  # Debugging statement
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        start = f"{data['startLatitude']},{data['startLongitude']}"
        end = f"{data['destLatitude']},{data['destLongitude']}"
        battery = float(data['battery'])
        range_km = float(data['range'])

        routes = get_routes(start, end)
        if not routes:
            return jsonify({"error": "No routes found!"}), 400

        best_route = choose_best_route(routes, battery, range_km)
        if not best_route:
            return jsonify({"error": "No valid route found!"}), 400

        # ✅ Extract route details for frontend
        leg = best_route["legs"][0]
        route_data = {
            "distance": leg["distance"]["value"] / 1000,
            "duration": leg["duration"]["text"],
            "battery_usage": (leg["distance"]["value"] / 1000) / range_km * 100,
            "steps": [
                {
                    "start_location": step["start_location"],
                    "end_location": step["end_location"],
                    "instructions": step["html_instructions"],
                }
                for step in leg["steps"]
            ]
        }

        print("Route Data being sent to frontend:", json.dumps(route_data, indent=2))  # ✅ Debugging print statement

        return jsonify(route_data)

    except Exception as e:
        print("Error:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
