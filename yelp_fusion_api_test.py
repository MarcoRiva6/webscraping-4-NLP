import requests

# Set up API credentials
API_KEY = "3lX3EqE4bLsHCwaN8tyZ3kNNg_tykrIiw8cgEDcbNOeGYo9m22YYW5as-1dPp-f0Gy_X8_12CDEiqVgbM0SdKgKE2x94_w4-_PLu8Kfufdj-kBvbYWCGNmUUjyZHZ3Yx"  # Replace with your Yelp API key
API_URL = "https://api.yelp.com/v3/businesses/search"

# Define the search parameters
headers = {"Authorization": f"Bearer {API_KEY}"}
params = {
    # "term": "creperies",
    "location": "Italy",
    "categories": "restaurants",
    "limit": 10,  # Number of results to fetch
}

# Make the request
response = requests.get(API_URL, headers=headers, params=params)

# Parse the response
if response.status_code == 200:
    data = response.json()
    for business in data.get("businesses", []):
        print(f"Name: {business['name']}")
        print(f"Rating: {business['rating']}")
        print(f"Address: {', '.join(business['location']['display_address'])}")
        print(f"Phone: {business.get('phone', 'N/A')}")
        print("-" * 40)
else:
    print(f"Error: {response.status_code} - {response.text}")
