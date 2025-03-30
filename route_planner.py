import googlemaps
import torch
import numpy as np
import json
from datetime import datetime

# ✅ Load the trained DQN model
class DQN(torch.nn.Module):
    def _init_(self, input_dim, output_dim):
        super(DQN, self)._init_()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Load trained model
model = DQN(4, 3)  # 4 input features: (lat, long, battery, range), 3 actions
model.load_state_dict(torch.load("dqn_ev_route.pth"))
model.eval()

# ✅ Google Maps API Key
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

def main():
    # ✅ User Inputs for EV Route Planning
    start_lat = float(input("Enter starting latitude: "))
    start_lng = float(input("Enter starting longitude: "))
    dest_lat = float(input("Enter destination latitude: "))
    dest_lng = float(input("Enter destination longitude: "))
    battery = float(input("Enter battery level (%): "))
    range_km = float(input("Enter range on full charge (km): "))

    start = (start_lat, start_lng)
    end = (dest_lat, dest_lng)

    # ✅ Fetch routes and find the best one
    routes = get_routes(start, end)
    if not routes:
        print("No routes found!")
        return
    best_route = choose_best_route(routes, battery, range_km)

    output_file = "optimal_ev_route.json"
    with open(output_file, "w") as f:
        json.dump(best_route, f, indent=2)
    
    print(f"\nOptimal EV Route has been saved to '{output_file}'.")
if _name_ == "_main_":
    main()          
