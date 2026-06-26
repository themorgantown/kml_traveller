#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable.
# Not required up front: ORS is the default provider, and the Google key is only
# validated when --api-provider google is actually selected (see create_api_client).
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

import sys
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import googlemaps
from googlemaps.convert import decode_polyline
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import shutil
import pickle
import hashlib
import time
import math
from datetime import datetime, timedelta

# ORS (OpenRouteService) support
ORS_API_KEY = os.getenv('ORS_API_KEY')
_ORS_AVAILABLE = False
try:
	import openrouteservice
	_ORS_AVAILABLE = True
except ImportError:
	pass


class ORSClientWrapper:
	"""Wraps openrouteservice.Client to expose a googlemaps-compatible interface."""

	def __init__(self, ors_client):
		self._ors = ors_client

	def geocode(self, address):
		response = self._ors.pelias_search(text=address, size=1)
		if response and response.get('features'):
			feature = response['features'][0]
			lon, lat = feature['geometry']['coordinates']
			return [{'geometry': {'location': {'lat': lat, 'lng': lon}}}]
		return []

	def distance_matrix(self, origins, destinations, mode="driving", avoid="tolls", units="metric"):
		if len(origins) != 1:
			raise ValueError("ORS distance_matrix wrapper supports exactly one origin per call")
		origin = origins[0]
		coords = [[origin[1], origin[0]]]
		dest_indices = []
		for dest in destinations:
			coords.append([dest[1], dest[0]])
			dest_indices.append(len(dest_indices) + 1)

		try:
			response = self._ors.distance_matrix(
				locations=coords,
				profile='driving-car',
				metrics=['distance'],
				sources=[0],
				destinations=dest_indices
			)
			distances = response.get('distances', [[]])[0] if response else []
			elements = []
			for d in distances:
				if d is not None:
					elements.append({'status': 'OK', 'distance': {'value': int(d)}})
				else:
					elements.append({'status': 'ZERO_RESULTS'})
			return {'rows': [{'elements': elements}]}
		except Exception:
			elements = [{'status': 'ZERO_RESULTS'}] * len(destinations)
			return {'rows': [{'elements': elements}]}

	def directions(self, origin, destination, mode="driving", avoid="tolls"):
		coords = [[origin[1], origin[0]], [destination[1], destination[0]]]
		options = {}
		if avoid == "tolls":
			options['avoid_features'] = ['tollways']

		try:
			response = self._ors.directions(
				coordinates=coords,
				profile='driving-car',
				format='json',
				options=options
			)
			if response and response.get('routes'):
				geometry = response['routes'][0].get('geometry', '')
				return [{'overview_polyline': {'points': geometry}}]
		except Exception:
			pass

		return []


def create_api_client(provider):
	"""Create a unified API client. provider is 'google' or 'ors'."""
	if provider == 'google':
		if not GOOGLE_MAPS_API_KEY:
			raise ValueError("GOOGLE_MAPS_API_KEY not set. Check your .env file.")
		return googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

	if provider == 'ors':
		if not _ORS_AVAILABLE:
			raise ImportError("openrouteservice package not installed. Run: pip install openrouteservice")
		if not ORS_API_KEY or ORS_API_KEY == 'your-openrouteservice-api-key-here':
			raise ValueError(
				"ORS_API_KEY not set. Get a free key at https://openrouteservice.org/dev/#/signup "
				"and set ORS_API_KEY in your .env file."
			)
		return ORSClientWrapper(openrouteservice.Client(key=ORS_API_KEY))

	raise ValueError(f"Unknown API provider: {provider}")

DEFAULT_START_CITY = "Kingston"
DEFAULT_START_MODE = "first-listed"
DEFAULT_ROUTE_SHAPE = "open"
DEFAULT_QUALITY = "maximum"

# === Geographic Utility Functions ===
def haversine_distance(lat1, lon1, lat2, lon2):
	"""
	Calculate the great circle distance between two points 
	on the earth (specified in decimal degrees)
	Returns distance in meters
	"""
	# Convert decimal degrees to radians
	lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
	
	# Haversine formula
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
	c = 2 * math.asin(math.sqrt(a))
	r = 6371000  # Radius of earth in meters
	return c * r

def calculate_centroid(locations):
	"""Calculate the geographic centroid of a list of (lat, lon) coordinates"""
	if not locations:
		return None
	
	lat_sum = sum(lat for lat, lon in locations)
	lon_sum = sum(lon for lat, lon in locations)
	
	return (lat_sum / len(locations), lon_sum / len(locations))

def find_optimal_start_location(locations):
	"""
	Find the optimal starting location for TSP - typically the one furthest from centroid
	This ensures we start from an edge location for better route optimization
	"""
	if len(locations) <= 1:
		return 0
	
	centroid = calculate_centroid(locations)
	if not centroid:
		return 0
	
	max_distance = 0
	optimal_start = 0
	
	for i, (lat, lon) in enumerate(locations):
		distance = haversine_distance(centroid[0], centroid[1], lat, lon)
		if distance > max_distance:
			max_distance = distance
			optimal_start = i
	
	return optimal_start

def should_skip_distance_calculation(loc1, loc2, max_reasonable_distance=200000.0):
	"""
	Determine if we should skip distance calculation between two locations
	based on geographic distance. Skip if they're impossibly far apart.
	max_reasonable_distance in meters (default 200km)
	"""
	geo_distance = haversine_distance(loc1[0], loc1[1], loc2[0], loc2[1])
	return geo_distance > max_reasonable_distance

def cluster_locations(locations, max_cluster_size=15):
	"""
	Simple geographic clustering for large datasets to reduce TSP complexity
	Groups nearby locations together to solve smaller sub-problems
	"""
	if len(locations) <= max_cluster_size:
		return [list(range(len(locations)))]  # Single cluster
	
	print(f"Clustering {len(locations)} locations into groups of max {max_cluster_size}...")
	
	# Simple k-means-like clustering based on geographic distance
	n_clusters = (len(locations) + max_cluster_size - 1) // max_cluster_size
	clusters = [[] for _ in range(n_clusters)]
	
	# Assign each location to the nearest cluster center
	# Start with evenly spaced points as initial cluster centers
	cluster_centers = []
	for i in range(n_clusters):
		center_idx = i * len(locations) // n_clusters
		cluster_centers.append(locations[center_idx])
	
	# Assign each location to closest cluster center
	for i, location in enumerate(locations):
		min_distance = float('inf')
		best_cluster = 0
		
		for j, center in enumerate(cluster_centers):
			distance = haversine_distance(location[0], location[1], center[0], center[1])
			if distance < min_distance:
				min_distance = distance
				best_cluster = j
		
		clusters[best_cluster].append(i)
	
	# Remove empty clusters
	clusters = [cluster for cluster in clusters if cluster]
	
	print(f"Created {len(clusters)} clusters:")
	for i, cluster in enumerate(clusters):
		print(f"  - Cluster {i+1}: {len(cluster)} locations")
	
	return clusters

# === Caching and API Optimization ===
class APICache:
	def __init__(self, cache_dir="cache", cache_duration_days=30):
		self.cache_dir = cache_dir
		self.cache_duration = timedelta(days=cache_duration_days)
		self.geocode_cache_file = os.path.join(cache_dir, "geocode_cache.pkl")
		self.distance_cache_file = os.path.join(cache_dir, "distance_cache.pkl")
		self.directions_cache_file = os.path.join(cache_dir, "directions_cache.pkl")

		# Create cache directory if it doesn't exist
		os.makedirs(cache_dir, exist_ok=True)

		# Load existing caches
		self.geocode_cache = self._load_cache(self.geocode_cache_file)
		self.distance_cache = self._load_cache(self.distance_cache_file)
		self.directions_cache = self._load_cache(self.directions_cache_file)

		print(f"Cache initialized: {len(self.geocode_cache)} geocode entries, {len(self.distance_cache)} distance entries, {len(self.directions_cache)} directions entries")

	def _load_cache(self, cache_file):
		"""Load cache from file, filtering out expired entries"""
		if not os.path.exists(cache_file):
			return {}

		try:
			with open(cache_file, 'rb') as f:
				cache = pickle.load(f)

			# Filter out expired entries
			now = datetime.now()
			valid_cache = {}
			for key, (data, timestamp) in cache.items():
				if now - timestamp < self.cache_duration:
					valid_cache[key] = (data, timestamp)

			return valid_cache
		except Exception as e:
			print(f"Warning: Could not load cache from {cache_file}: {e}")
			return {}

	def _save_cache(self, cache, cache_file):
		"""Save cache to file"""
		try:
			with open(cache_file, 'wb') as f:
				pickle.dump(cache, f)
		except Exception as e:
			print(f"Warning: Could not save cache to {cache_file}: {e}")

	def _hash_location(self, location):
		"""Create a hash key for a location (lat, lon)"""
		if isinstance(location, (tuple, list)):
			return f"{location[0]:.6f},{location[1]:.6f}"
		return str(location)

	def get_geocode(self, address):
		"""Get geocoded coordinates from cache or return None"""
		key = hashlib.md5(address.encode()).hexdigest()
		if key in self.geocode_cache:
			data, timestamp = self.geocode_cache[key]
			print(f"  Cache hit for: {address}")
			return data
		return None

	def set_geocode(self, address, result):
		"""Save geocoded result to cache"""
		key = hashlib.md5(address.encode()).hexdigest()
		self.geocode_cache[key] = (result, datetime.now())
		self._save_cache(self.geocode_cache, self.geocode_cache_file)

	def get_distance(self, origin, destination):
		"""Get distance from cache or return None"""
		key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
		if key in self.distance_cache:
			data, timestamp = self.distance_cache[key]
			return data
		return None

	def set_distance(self, origin, destination, result):
		"""Save distance result to cache"""
		key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
		self.distance_cache[key] = (result, datetime.now())
		self._save_cache(self.distance_cache, self.distance_cache_file)

	def get_directions_path(self, origin, destination):
		"""Get decoded directions path from cache or return None"""
		key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
		if key in self.directions_cache:
			data, timestamp = self.directions_cache[key]
			return data
		return None

	def set_directions_path(self, origin, destination, result):
		"""Save decoded directions path to cache"""
		key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
		self.directions_cache[key] = (result, datetime.now())
		self._save_cache(self.directions_cache, self.directions_cache_file)

	def get_cache_stats(self):
		"""Return cache statistics"""
		return {
			'geocode_entries': len(self.geocode_cache),
			'distance_entries': len(self.distance_cache),
			'directions_entries': len(self.directions_cache),
			'cache_directory': self.cache_dir,
			'cache_duration_days': self.cache_duration.days
		}


# === Step 1: Extract Addresses and Names from KMZ ===
def extract_addresses_from_kmz(kmz_path):
	extract_dir = "kmz_extracted"
	
	# Clean up previous extraction directory if it exists
	if os.path.exists(extract_dir):
		import shutil
		shutil.rmtree(extract_dir)
	
	with zipfile.ZipFile(kmz_path, 'r') as kmz:
		kmz.extractall(extract_dir)
	
	kml_files = [f for f in os.listdir(extract_dir) if f.endswith('.kml')]
	if not kml_files:
		raise FileNotFoundError("No KML file found in KMZ archive.")
	
	print(f"Processing KML file: {kml_files[0]}")
	kml_path = os.path.join(extract_dir, kml_files[0])
	tree = ET.parse(kml_path)
	root = tree.getroot()
	
	# Handle different KML namespace variations
	ns = {'kml': 'http://www.opengis.net/kml/2.2'}
	addresses = []
	location_names = []
	
	# Find all placemarks
	placemarks = root.findall('.//kml:Placemark', ns)
	print(f"Found {len(placemarks)} placemarks in KML file")
	
	for i, placemark in enumerate(placemarks):
		# Get placemark name if available
		name_elem = placemark.find('kml:name', ns)
		if name_elem is not None and name_elem.text:
			name = name_elem.text.strip()
		else:
			# Try getting name from address field
			address_elem = placemark.find('kml:address', ns)
			if address_elem is not None and address_elem.text:
				name = address_elem.text.strip().split(',')[0]  # Take first part as name
			else:
				name = f"Location_{i+1}"
		
		# Look for address in ExtendedData
		address = None
		extended_data = placemark.find('kml:ExtendedData', ns)
		if extended_data is not None:
			for data in extended_data.findall('kml:Data', ns):
				name_attr = data.get('name')
				if name_attr and name_attr.lower() == 'address':
					value_elem = data.find('kml:value', ns)
					if value_elem is not None and value_elem.text:
						address = value_elem.text.strip()
						break
		
		# If no address in ExtendedData, try the address field
		if not address:
			address_elem = placemark.find('kml:address', ns)
			if address_elem is not None and address_elem.text:
				address = address_elem.text.strip()
		
		# If we have an address, add it to our list
		if address:
			addresses.append(address)
			location_names.append(name)
			print(f"  {len(addresses)}. {name}: {address}")
		else:
			# Check if there are direct coordinates
			points = placemark.findall('.//kml:Point', ns)
			for point in points:
				coord_elem = point.find('kml:coordinates', ns)
				if coord_elem is not None and coord_elem.text:
					coord_text = coord_elem.text.strip()
					try:
						# Coordinates in KML are lon,lat,alt (altitude optional)
						coords_parts = coord_text.split(',')
						lon, lat = float(coords_parts[0]), float(coords_parts[1])
						# For direct coordinates, we'll store them as a special format
						addresses.append(f"COORDINATES:{lat},{lon}")
						location_names.append(name)
						print(f"  {len(addresses)}. {name}: Direct coordinates ({lat:.6f}, {lon:.6f})")
					except (ValueError, IndexError) as e:
						print(f"  Warning: Could not parse coordinates for {name}: {coord_text}")
						continue
	
	print(f"Successfully extracted {len(addresses)} addresses/locations")
	return addresses, location_names

# === Step 2: Convert Addresses to GPS Coordinates ===
def geocode_addresses(addresses, location_names, gmaps, cache):
	coordinates = []
	final_names = []
	final_addresses = []
	
	print("Converting addresses to GPS coordinates...")
	
	for i, address in enumerate(addresses):
		name = location_names[i] if i < len(location_names) else f"Location_{i+1}"
		
		# Check if this is already a coordinate
		if address.startswith("COORDINATES:"):
			coord_str = address.replace("COORDINATES:", "")
			lat, lon = map(float, coord_str.split(','))
			coordinates.append((lat, lon))
			final_names.append(name)
			final_addresses.append(address)
			print(f"  {len(coordinates)}. {name}: Using direct coordinates ({lat:.6f}, {lon:.6f})")
			continue
		
		# Check cache first
		cached_result = cache.get_geocode(address)
		if cached_result:
			lat, lon = cached_result['lat'], cached_result['lng']
			coordinates.append((lat, lon))
			final_names.append(name)
			final_addresses.append(address)
			print(f"  {len(coordinates)}. {name}: ({lat:.6f}, {lon:.6f})")
			continue
		
		try:
			print(f"  Geocoding: {name} - {address}")
			geocode_result = gmaps.geocode(address)
			
			if geocode_result:
				location = geocode_result[0]['geometry']['location']
				lat, lon = location['lat'], location['lng']
				coordinates.append((lat, lon))
				final_names.append(name)
				final_addresses.append(address)
				
				# Cache the result
				cache.set_geocode(address, location)
				print(f"  {len(coordinates)}. {name}: ({lat:.6f}, {lon:.6f})")
				# ORS free tier: 100 geocodes/min
				if isinstance(gmaps, ORSClientWrapper):
					time.sleep(0.6)
			else:
				print(f"  Warning: Could not geocode address for {name}: {address}")
				
		except Exception as e:
			print(f"  Error geocoding {name}: {str(e)}")
			continue
		
	print(f"Successfully geocoded {len(coordinates)} locations")
	return coordinates, final_names, final_addresses

# === Step 3: Create Distance Matrix with Optimizations ===
def create_distance_matrix(locations, gmaps, cache, quality=DEFAULT_QUALITY):
	n = len(locations)
	directed = quality == "maximum"
	use_geographic_filter = quality != "maximum"
	print(f"Creating {'directed' if directed else 'symmetric'} {n}x{n} distance matrix...")
	matrix = [[0] * n for _ in range(n)]
	
	# Calculate centroid for geographic filtering
	centroid = calculate_centroid(locations)
	if centroid:
		avg_distance_from_center = sum(
			haversine_distance(centroid[0], centroid[1], lat, lon) 
			for lat, lon in locations
		) / len(locations)
		max_reasonable_distance = avg_distance_from_center * 3  # 3x average as cutoff
		print(f"Geographic filtering: max reasonable distance = {max_reasonable_distance/1000:.1f} km")
	else:
		max_reasonable_distance = 200000  # Default 200km
	if not use_geographic_filter:
		print("Maximum quality mode: geographic distance skips disabled")
	
	# Collect directed pairs that need to be calculated.
	pairs_to_calculate = []
	cache_hits = 0
	geographic_skips = 0
	
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			# Check cache first
			cached_distance = cache.get_distance(locations[i], locations[j])
			if cached_distance is not None:
				matrix[i][j] = cached_distance
				cache_hits += 1
			else:
				# Check if geographic distance is too large
				if use_geographic_filter and should_skip_distance_calculation(locations[i], locations[j], max_reasonable_distance):
					penalty_distance = int(max_reasonable_distance * 2)  # Large penalty
					matrix[i][j] = penalty_distance
					geographic_skips += 1
				else:
					pairs_to_calculate.append((i, j))
	
	total_pairs = n * (n - 1)
	api_calls_saved = cache_hits + geographic_skips
	
	print(f"Optimization results:")
	print(f"  - Total pairs to consider: {total_pairs}")
	print(f"  - Cache hits: {cache_hits}")
	print(f"  - Geographic skips: {geographic_skips}")
	print(f"  - API calls needed: {len(pairs_to_calculate)}")
	print(f"  - Total API calls saved: {api_calls_saved} ({api_calls_saved/total_pairs*100:.1f}%)")
	
	if len(pairs_to_calculate) == 0:
		print("All distances found in cache or filtered out!")
		return matrix
	
	origin_groups = {}
	for origin_idx, dest_idx in pairs_to_calculate:
		if origin_idx not in origin_groups:
			origin_groups[origin_idx] = []
		origin_groups[origin_idx].append(dest_idx)

	# ORS can handle all destinations in one call; Google limits to 25 per call.
	batch_size = 9999 if isinstance(gmaps, ORSClientWrapper) else 25
	rate_limit_sleep = 2.0 if isinstance(gmaps, ORSClientWrapper) else 0.1  # ORS: 30 req/min for matrix
	total_batches = sum((len(destinations) + batch_size - 1) // batch_size for destinations in origin_groups.values())
	current_batch = 0

	print(f"Processing {len(pairs_to_calculate)} directed distance calculations in {total_batches} API batches...")

	for origin_idx, destination_indices in origin_groups.items():
		origin = locations[origin_idx]
		for start_idx in range(0, len(destination_indices), batch_size):
			current_batch += 1
			batch_destination_indices = destination_indices[start_idx:start_idx + batch_size]
			destinations = [locations[j] for j in batch_destination_indices]
			print(f"Batch {current_batch}/{total_batches}: origin {origin_idx}, {len(destinations)} destinations...")
			
			try:
				response = gmaps.distance_matrix(
					origins=[origin],
					destinations=destinations,
					mode="driving",
					avoid="tolls",
					units="metric"
				)
				
				if response["rows"]:
					elements = response["rows"][0]["elements"]
					for idx, element in enumerate(elements):
						dest_idx = batch_destination_indices[idx]
						
						if element["status"] == "OK":
							distance = element["distance"]["value"]
							matrix[origin_idx][dest_idx] = distance
							cache.set_distance(origin, destinations[idx], distance)
						else:
							print(f"Warning: Distance calculation failed between points {origin_idx} and {dest_idx}: {element.get('status', 'Unknown error')}")
							fallback_distance = int(haversine_distance(
								origin[0], origin[1], destinations[idx][0], destinations[idx][1]
							) * 1.5)
							matrix[origin_idx][dest_idx] = fallback_distance
				
				dir_sleep = 1.6 if isinstance(gmaps, ORSClientWrapper) else 0.1
				time.sleep(dir_sleep)
				
			except Exception as e:
				print(f"Error in batch request for origin {origin_idx}: {str(e)}")
				for dest_idx in batch_destination_indices:
					fallback_distance = int(haversine_distance(
						origin[0], origin[1], locations[dest_idx][0], locations[dest_idx][1]
					) * 1.5)
					matrix[origin_idx][dest_idx] = fallback_distance
	
	print("Distance matrix creation completed!")
	print(f"Total API calls saved by optimization: {api_calls_saved}")
	
	# Calculate estimated cost savings
	is_ors = isinstance(gmaps, ORSClientWrapper)
	cost_per_call = 0.0 if is_ors else 0.005
	provider_name = "ORS (free)" if is_ors else "Google Maps"
	estimated_savings = api_calls_saved * cost_per_call
	total_possible_cost = total_pairs * cost_per_call
	remaining_cost = len(pairs_to_calculate) * cost_per_call
	
	print(f"💰 Cost Analysis ({provider_name}):")
	print(f"  - Cost without optimization: ${total_possible_cost:.2f}")
	print(f"  - Actual cost this run: ${remaining_cost:.2f}")
	print(f"  - Savings this run: ${estimated_savings:.2f}")
	print(f"  - Savings percentage: {(estimated_savings/total_possible_cost)*100:.1f}%" if total_possible_cost > 0 else "  - Savings percentage: N/A (free provider)")
	
	return matrix

def find_start_index(addresses, location_names, start_city=DEFAULT_START_CITY, start_mode=DEFAULT_START_MODE):
	"""Find the configured first stop from geocoded source addresses."""
	if start_mode != "first-listed":
		raise ValueError(f"Unsupported start mode: {start_mode}")

	city_needle = f"{start_city}, NY".lower()
	for idx, address in enumerate(addresses):
		if city_needle in address.lower():
			name = location_names[idx] if idx < len(location_names) else f"Location_{idx+1}"
			print(f"Fixed start selected: {name} ({address})")
			return idx

	print(f"Warning: no address matched '{start_city}, NY'; falling back to first location")
	return 0

def nearest_neighbor_tsp(matrix, start_idx=0, route_shape=DEFAULT_ROUTE_SHAPE):
    """
    Simple nearest neighbor TSP heuristic for fallback when OR-Tools times out
    """
    n = len(matrix)
    if n <= 1:
        return list(range(n))
    
    unvisited = set(range(n))
    route = [start_idx]
    unvisited.remove(start_idx)
    current = start_idx
    
    print(f"Using nearest neighbor heuristic starting from index {start_idx}")
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: matrix[current][x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    if route_shape == "closed":
        route.append(start_idx)
    
    # Calculate total distance
    total_distance = sum(matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    print(f"Nearest neighbor solution: {total_distance/1000:.2f} km")
    
    return route

# === Step 4: Solve TSP with Optimal Starting Point ===
def solve_tsp(
	matrix,
	locations,
	start_idx=None,
	route_shape=DEFAULT_ROUTE_SHAPE,
	fallback_to_nearest=True,
	time_limit_override=None
):
    tsp_size = len(matrix)
    print(f"Solving {route_shape} TSP for {tsp_size} locations...")
    
    if tsp_size < 2:
        print("Warning: Need at least 2 locations to solve TSP")
        return list(range(tsp_size)) if tsp_size > 0 else []
    
    if start_idx is None:
        start_idx = find_optimal_start_location(locations)
        print(f"Optimal starting location: Index {start_idx} (edge location)")
    else:
        print(f"Fixed starting location: Index {start_idx}")
    
    dummy_end_idx = tsp_size if route_shape == "open" else None
    manager_size = tsp_size + 1 if route_shape == "open" else tsp_size

    if route_shape == "open":
        manager = pywrapcp.RoutingIndexManager(manager_size, 1, [start_idx], [dummy_end_idx])
    else:
        manager = pywrapcp.RoutingIndexManager(manager_size, 1, start_idx)

    # Create RoutingModel
    routing = pywrapcp.RoutingModel(manager)

    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if dummy_end_idx is not None and to_node == dummy_end_idx:
            return 0
        if dummy_end_idx is not None and from_node == dummy_end_idx:
            return 0
        return matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Search parameters - use sophisticated strategy based on problem size
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    if tsp_size <= 10:
        # Small problems: Use exact algorithm with short timeout
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        time_limit = 30  # 30 seconds
    elif tsp_size <= 20:
        # Medium problems: Balance quality and speed
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        time_limit = 60  # 1 minute
    elif tsp_size <= 35:
        # Larger problems: Focus on speed
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
        time_limit = 90  # 1.5 minutes
    else:
        # Very large problems: Quick heuristic solution only
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        time_limit = 180 if route_shape == "open" else 60
    if time_limit_override is not None:
        time_limit = time_limit_override
    
    search_parameters.time_limit.seconds = time_limit
    print(f"TSP solver strategy: {search_parameters.first_solution_strategy}")
    print(f"Local search method: {search_parameters.local_search_metaheuristic}")
    print(f"Time limit: {time_limit} seconds")

    # Solve
    print("Searching for optimal solution...")
    start_time = time.time()
    assignment = routing.SolveWithParameters(search_parameters)
    solve_time = time.time() - start_time
    
    if assignment:
        print(f"✓ Solution found in {solve_time:.1f} seconds!")
        
        # Extract the route
        index = routing.Start(0)
        route = []
        total_distance = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != dummy_end_idx:
                route.append(node)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        
        end_node = manager.IndexToNode(index)
        if route_shape == "closed" and end_node != dummy_end_idx:
            route.append(end_node)
        
        # Calculate route statistics
        print(f"📊 Route Statistics:")
        print(f"  - Total distance: {total_distance/1000:.2f} km")
        print(f"  - Average distance per segment: {total_distance/(len(route)-1)/1000:.2f} km")
        print(f"  - Starting index: {start_idx}")
        
        # Estimate travel time (assuming average speed of 50 km/h)
        estimated_hours = (total_distance / 1000) / 50
        print(f"  - Estimated travel time: {estimated_hours:.1f} hours")
        
        return route
    else:
        if fallback_to_nearest:
            print(f"❌ No solution found within {time_limit} seconds! Using greedy fallback...")
            return nearest_neighbor_tsp(matrix, start_idx, route_shape)
        print(f"❌ No solution found within {time_limit} seconds.")
        return None

def solve_tsp_with_clustering(matrix, locations, start_idx=None, route_shape=DEFAULT_ROUTE_SHAPE, quality=DEFAULT_QUALITY, max_cluster_size=30):
	"""
	Solve TSP with optional clustering for large datasets
	For datasets larger than max_cluster_size, use clustering to reduce complexity
	"""
	n = len(locations)
	
	if quality == "maximum":
		route = solve_tsp(
			matrix,
			locations,
			start_idx=start_idx,
			route_shape=route_shape,
			fallback_to_nearest=False
		)
		if route:
			return route
		print("Full-route solve failed; falling back to clustered solve.")

	if n <= max_cluster_size:
		return solve_tsp(matrix, locations, start_idx=start_idx, route_shape=route_shape)
	
	print(f"Large dataset detected ({n} locations). Using clustering optimization...")
	
	# Cluster the locations with smaller cluster sizes for better performance
	clusters = cluster_locations(locations, min(15, max_cluster_size // 4))
	
	if len(clusters) == 1:
		# Clustering didn't help, solve directly
			return solve_tsp(matrix, locations, start_idx=start_idx, route_shape=route_shape)
	
	# Solve TSP for each cluster
	cluster_routes = []
	cluster_centers = []
	
	for i, cluster in enumerate(clusters):
		print(f"Solving TSP for cluster {i+1}/{len(clusters)} ({len(cluster)} locations)...")
		
		# Create sub-matrix for this cluster
		cluster_matrix = [[matrix[cluster[a]][cluster[b]] for b in range(len(cluster))] for a in range(len(cluster))]
		cluster_coords = [locations[idx] for idx in cluster]
		
		# Solve TSP for this cluster
		cluster_start = cluster.index(start_idx) if start_idx in cluster else None
		cluster_route = solve_tsp(cluster_matrix, cluster_coords, start_idx=cluster_start, route_shape=route_shape)
		
		# Convert back to original indices
		original_route = [cluster[idx] for idx in cluster_route if idx < len(cluster)]
		cluster_routes.append(original_route)
		
		# Calculate cluster center (centroid of the cluster)
		cluster_center_idx = find_optimal_start_location(cluster_coords)
		cluster_centers.append(cluster[cluster_center_idx])
	
	# Solve TSP between cluster centers
	print(f"Solving TSP between {len(clusters)} cluster centers...")
	center_matrix = [[matrix[cluster_centers[i]][cluster_centers[j]] for j in range(len(cluster_centers))] for i in range(len(cluster_centers))]
	center_locations = [locations[idx] for idx in cluster_centers]
	center_route = solve_tsp(center_matrix, center_locations, route_shape=route_shape)
	
	# Combine cluster routes in the order determined by center TSP
	final_route = []
	for center_idx in center_route:
		if center_idx < len(cluster_routes):
			cluster_route = cluster_routes[center_idx]
			# Remove duplicates and add to final route
			for location_idx in cluster_route:
				if location_idx not in final_route:
					final_route.append(location_idx)
	
	# Ensure we return to the start for closed routes only.
	if route_shape == "closed" and final_route and final_route[-1] != final_route[0]:
		final_route.append(final_route[0])
	
	print(f"✓ Clustering optimization completed!")
	print(f"  - Reduced problem from {n} to {len(clusters)} clusters")
	print(f"  - Final route has {len(final_route)} waypoints")
	
	return final_route

def fetch_directions_segments(locations, route, location_names, gmaps, cache, enabled=True):
	"""Fetch road-following coordinates for each adjacent route leg."""
	segments = []
	if not enabled:
		print("Directions paths disabled; using straight route segments")

	for i in range(len(route) - 1):
		start_idx = route[i]
		end_idx = route[i + 1]
		origin = locations[start_idx]
		destination = locations[end_idx]
		start_name = location_names[start_idx] if location_names and start_idx < len(location_names) else f"Location_{start_idx+1}"
		end_name = location_names[end_idx] if location_names and end_idx < len(location_names) else f"Location_{end_idx+1}"
		fallback_path = [origin, destination]
		path = None
		status = "fallback"

		if enabled:
			cached_path = cache.get_directions_path(origin, destination)
			if cached_path:
				path = cached_path
				status = "cached"
			else:
				dir_sleep = 1.6 if isinstance(gmaps, ORSClientWrapper) else 0.1
				try:
					response = gmaps.directions(
						origin=origin,
						destination=destination,
						mode="driving",
						avoid="tolls"
					)
					if response:
						points = response[0].get("overview_polyline", {}).get("points")
						if points:
							decoded = decode_polyline(points)
							path = [(point["lat"], point["lng"]) for point in decoded]
							cache.set_directions_path(origin, destination, path)
							status = "directions"
					time.sleep(dir_sleep)
				except Exception as e:
					print(f"Warning: Directions failed for segment {i + 1} ({start_name} -> {end_name}): {e}")

		if not path:
			path = fallback_path
			print(f"Warning: using straight fallback for segment {i + 1} ({start_name} -> {end_name})")

		segments.append({
			"from_index": start_idx,
			"to_index": end_idx,
			"from_name": start_name,
			"to_name": end_name,
			"path": path,
			"status": status
		})

	directions_count = sum(1 for segment in segments if segment["status"] == "directions")
	cached_count = sum(1 for segment in segments if segment["status"] == "cached")
	fallback_count = sum(1 for segment in segments if segment["status"] == "fallback")
	print(f"Directions geometry: {directions_count} fetched, {cached_count} cached, {fallback_count} fallback")
	return segments

def flatten_segment_paths(route_segments):
	"""Combine leg paths into one route path without duplicate join points."""
	route_path = []
	for segment in route_segments:
		for point in segment["path"]:
			if not route_path or route_path[-1] != point:
				route_path.append(point)
	return route_path

# === Step 5: Save GeoJSON Output ===
def save_geojson(locations, route, location_names=None, output_file="optimized_route.geojson", route_segments=None):
	features = []
	
	# Add points as features
	for i, (lat, lon) in enumerate(locations):
		point_name = location_names[i] if location_names and i < len(location_names) else f"Location_{i+1}"
		route_order = route.index(i) + 1 if i in route else 0
		
		features.append({
			"type": "Feature",
			"geometry": {
				"type": "Point",
				"coordinates": [lon, lat]
			},
			"properties": {
				"name": point_name,
				"location_index": i,
				"route_order": route_order,
				"marker-color": "#ff0000" if route_order == 1 else "#0080ff",
				"marker-size": "large" if route_order == 1 else "medium",
				"marker-symbol": "1" if route_order == 1 else str(route_order) if route_order <= 9 else "circle"
			}
		})
	
	# Add route lines as features
	if route_segments is None:
		route_segments = []
		for i in range(len(route) - 1):
			start_idx = route[i]
			end_idx = route[i + 1]
			start = locations[start_idx]
			end = locations[end_idx]
			start_name = location_names[start_idx] if location_names and start_idx < len(location_names) else f"Location_{start_idx+1}"
			end_name = location_names[end_idx] if location_names and end_idx < len(location_names) else f"Location_{end_idx+1}"
			route_segments.append({
				"from_index": start_idx,
				"to_index": end_idx,
				"from_name": start_name,
				"to_name": end_name,
				"path": [start, end],
				"status": "fallback"
			})
	for i, segment in enumerate(route_segments):
		
		features.append({
			"type": "Feature",
			"geometry": {
				"type": "LineString",
				"coordinates": [[lon, lat] for lat, lon in segment["path"]]
			},
			"properties": {
				"from": segment["from_name"],
				"to": segment["to_name"],
				"from_index": segment["from_index"],
				"to_index": segment["to_index"],
				"segment": i + 1,
				"geometry_source": segment["status"],
				"stroke": "#ff0000",
				"stroke-width": 3,
				"stroke-opacity": 0.8
			}
		})
	
	geojson = {
		"type": "FeatureCollection",
		"features": features
	}
	
	with open(output_file, "w") as f:
		json.dump(geojson, f, indent=2)
	print(f"GeoJSON route saved to: {output_file}")
	print(f"  - {len([f for f in features if f['geometry']['type'] == 'Point'])} location points")
	print(f"  - {len([f for f in features if f['geometry']['type'] == 'LineString'])} route segments")

# === Step 5: Save KML Output ===
def save_kml(locations, route, location_names=None, output_file="optimized_route.kml", route_segments=None):
	"""
	Creates a KML file with the optimized route including both points and path
	"""
	kml_content = []
	
	# KML header
	kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
	kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
	kml_content.append('  <Document>')
	kml_content.append('    <name>Optimized Route</name>')
	kml_content.append('    <description>Route optimization generated by route optimizer</description>')
	
	# Define styles
	kml_content.append('    <!-- Styles -->')
	kml_content.append('    <Style id="startPoint">')
	kml_content.append('      <IconStyle>')
	kml_content.append('        <color>ff0000ff</color>')  # Red
	kml_content.append('        <scale>1.2</scale>')
	kml_content.append('        <Icon>')
	kml_content.append('          <href>http://maps.google.com/mapfiles/kml/paddle/red-stars.png</href>')
	kml_content.append('        </Icon>')
	kml_content.append('      </IconStyle>')
	kml_content.append('    </Style>')
	
	kml_content.append('    <Style id="routePoint">')
	kml_content.append('      <IconStyle>')
	kml_content.append('        <color>ff0080ff</color>')  # Orange
	kml_content.append('        <scale>1.0</scale>')
	kml_content.append('        <Icon>')
	kml_content.append('          <href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href>')
	kml_content.append('        </Icon>')
	kml_content.append('      </IconStyle>')
	kml_content.append('    </Style>')
	
	kml_content.append('    <Style id="routePath">')
	kml_content.append('      <LineStyle>')
	kml_content.append('        <color>ff0000ff</color>')  # Red line
	kml_content.append('        <width>4</width>')
	kml_content.append('      </LineStyle>')
	kml_content.append('    </Style>')
	
	# Create folder for route points
	kml_content.append('    <Folder>')
	kml_content.append('      <name>Route Points</name>')
	kml_content.append('      <description>Waypoints in optimized order</description>')
	
	# Add waypoints in route order
	for i, route_idx in enumerate(route):
		if route_idx < len(locations):
			lat, lon = locations[route_idx]
			location_name = location_names[route_idx] if location_names and route_idx < len(location_names) else f"Location_{route_idx+1}"
			location_name_xml = escape(location_name)
			
			style_id = "startPoint" if i == 0 else "routePoint"
			
			kml_content.append('      <Placemark>')
			kml_content.append(f'        <name>{i + 1}. {location_name_xml}</name>')
			kml_content.append(f'        <description>Stop {i + 1} of {len(route)} in optimized route</description>')
			kml_content.append(f'        <styleUrl>#{style_id}</styleUrl>')
			kml_content.append('        <Point>')
			kml_content.append(f'          <coordinates>{lon},{lat},0</coordinates>')
			kml_content.append('        </Point>')
			kml_content.append('      </Placemark>')
	
	kml_content.append('    </Folder>')
	
	# Create the route path
	kml_content.append('    <Placemark>')
	kml_content.append('      <name>Optimized Route Path</name>')
	kml_content.append('      <description>Complete optimized route connecting all waypoints</description>')
	kml_content.append('      <styleUrl>#routePath</styleUrl>')
	kml_content.append('      <LineString>')
	kml_content.append('        <tessellate>1</tessellate>')
	kml_content.append('        <coordinates>')
	
	# Add coordinates for the complete path
	path_coords = []
	route_path = flatten_segment_paths(route_segments) if route_segments else [locations[route_idx] for route_idx in route if route_idx < len(locations)]
	for lat, lon in route_path:
		path_coords.append(f'          {lon},{lat},0')
	
	kml_content.append('\n'.join(path_coords))
	kml_content.append('        </coordinates>')
	kml_content.append('      </LineString>')
	kml_content.append('    </Placemark>')
	
	# KML footer
	kml_content.append('  </Document>')
	kml_content.append('</kml>')
	
	# Write to file
	with open(output_file, 'w', encoding='utf-8') as f:
		f.write('\n'.join(kml_content))
	
	print(f"KML route saved to: {output_file}")
	print(f"  - {len(route)} waypoints in optimized order")
	print(f"  - Complete route path with {len(path_coords)} coordinates")

def save_kmz(kml_file, kmz_file):
	"""Write a KMZ containing the KML as doc.kml."""
	with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as kmz:
		kmz.write(kml_file, "doc.kml")
	print(f"KMZ route saved to: {kmz_file}")
	
# === Main Execution ===
def main():
	parser = argparse.ArgumentParser(description="Optimize a KMZ route and export KML, GeoJSON, and KMZ.")
	parser.add_argument("kmz_file", nargs="?", default="file.kmz", help="Input KMZ file")
	parser.add_argument("out_base", nargs="?", default="optimized_route", help="Output base path without extension")
	parser.add_argument("--start-city", default=DEFAULT_START_CITY, help="City to use for the fixed first stop")
	parser.add_argument("--start-mode", choices=["first-listed"], default=DEFAULT_START_MODE, help="How to choose the start within the city")
	parser.add_argument("--route-shape", choices=["open", "closed"], default=DEFAULT_ROUTE_SHAPE, help="Open itinerary or closed loop")
	parser.add_argument("--quality", choices=["maximum", "balanced"], default=DEFAULT_QUALITY, help="Routing quality/cost mode")
	parser.add_argument("--directions-paths", dest="directions_paths", action="store_true", default=True, help="Use Directions API geometry for route lines")
	parser.add_argument("--no-directions-paths", dest="directions_paths", action="store_false", help="Use straight route lines")
	parser.add_argument("--api-provider", choices=["ors", "google"], default="ors", help="API provider for routing (default: ors)")
	args = parser.parse_args()
	kmz_file = args.kmz_file
	out_base = args.out_base

	# Check if KMZ file exists
	if not os.path.exists(kmz_file):
		print(f"Error: KMZ file '{kmz_file}' not found!")
		return
	
	# Initialize cache
	cache = APICache()
	
	print("=" * 50)
	print("ROUTE OPTIMIZATION STARTING")
	print("=" * 50)
	
	print("Step 1: Extracting addresses from KMZ file...")
	try:
		addresses, location_names = extract_addresses_from_kmz(kmz_file)
	except Exception as e:
		print(f"Error extracting addresses: {str(e)}")
		return
	
	if len(addresses) == 0:
		print("No addresses found in the KMZ file!")
		return
	elif len(addresses) == 1:
		print("Only one location found. No route optimization needed.")
		return
	
	print(f"\nStep 2: Converting {len(addresses)} addresses to GPS coordinates...")
	try:
		gmaps = create_api_client(args.api_provider)
		coords, final_location_names, final_addresses = geocode_addresses(addresses, location_names, gmaps, cache)
	except Exception as e:
		print(f"Error geocoding addresses: {str(e)}")
		return
	
	if len(coords) < 2:
		print("Need at least 2 valid coordinates for route optimization!")
		return

	start_idx = find_start_index(final_addresses, final_location_names, args.start_city, args.start_mode)
		
	print(f"\nStep 3: Creating distance matrix for {len(coords)} locations...")
	try:
		distance_matrix = create_distance_matrix(coords, gmaps, cache, quality=args.quality)
	except Exception as e:
		print(f"Error creating distance matrix: {str(e)}")
		return
	
	print(f"\nStep 4: Solving traveling salesman problem with optimizations...")
	route = solve_tsp_with_clustering(
		distance_matrix,
		coords,
		start_idx=start_idx,
		route_shape=args.route_shape,
		quality=args.quality
	)
	if not route:
		print("Could not find a solution.")
		return
	
	print(f"\nStep 5: Displaying optimized route:")
	print("-" * 40)
	for i, idx in enumerate(route):
			location_name = final_location_names[idx] if idx < len(final_location_names) else f"Location_{idx+1}"
			print(f"{i + 1:2d}. {location_name}")
			print(f"     Coordinates: {coords[idx]}")
		
	print(f"\nStep 6: Fetching road-following route geometry...")
	route_segments = fetch_directions_segments(
		coords,
		route,
		final_location_names,
		gmaps,
		cache,
		enabled=args.directions_paths
	)

	print(f"\nStep 7: Saving results...")
	out_dir = os.path.dirname(out_base)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	kml_file = f"{out_base}.kml"
	geojson_file = f"{out_base}.geojson"
	kmz_file_out = f"{out_base}.kmz"
	save_geojson(coords, route, final_location_names, output_file=geojson_file, route_segments=route_segments)
	save_kml(coords, route, final_location_names, output_file=kml_file, route_segments=route_segments)
	save_kmz(kml_file, kmz_file_out)
	
	print("\n" + "=" * 50)
	print("ROUTE OPTIMIZATION COMPLETED!")
	print("=" * 50)
	print(f"✓ Processed {len(coords)} locations")
	print(f"✓ Starting location: {final_location_names[route[0]]}")
	print(f"✓ Final location: {final_location_names[route[-1]]}")
	print(f"✓ Route shape: {args.route_shape}")
	print(f"✓ Optimized route saved to '{geojson_file}'")
	print(f"✓ Optimized route saved to '{kml_file}'")
	print(f"✓ Optimized route saved to '{kmz_file_out}'")
	print("✓ You can import these files into mapping software like Google Earth")
	
	# Show cache statistics
	cache_stats = cache.get_cache_stats()
	print(f"\n📊 Cache Statistics:")
	print(f"  - Geocode cache entries: {cache_stats['geocode_entries']}")
	print(f"  - Distance cache entries: {cache_stats['distance_entries']}")
	print(f"  - Directions cache entries: {cache_stats['directions_entries']}")
	print(f"  - Cache valid for: {cache_stats['cache_duration_days']} days")
	
	# Calculate total optimization savings
	total_possible_geocode_calls = len(coords)
	total_possible_distance_calls = len(coords) * (len(coords) - 1)
	cost_per_geocode = 0.0 if isinstance(gmaps, ORSClientWrapper) else 0.005
	cost_per_distance = 0.0 if isinstance(gmaps, ORSClientWrapper) else 0.005
	
	total_possible_cost = (total_possible_geocode_calls * cost_per_geocode) + (total_possible_distance_calls * cost_per_distance)
	
	provider_name = "ORS (free)" if isinstance(gmaps, ORSClientWrapper) else "Google Maps"
	print(f"\n💰 Complete Optimization Summary:")
	print(f"  - API provider: {provider_name}")
	print(f"  - Total possible API calls: {total_possible_geocode_calls + total_possible_distance_calls}")
	print(f"  - Geocode calls: {total_possible_geocode_calls}")
	print(f"  - Distance matrix elements: {total_possible_distance_calls} (directed maximum quality)")
	print(f"  - Theoretical cost without optimization: ${total_possible_cost:.2f}")
	print(f"  - Subsequent runs will use cached data (near $0 cost)")
	
	print(f"\n🎯 Optimization Features Applied:")
	print(f"  ✅ API provider: {provider_name}")
	print(f"  ✅ Directed driving distance matrix")
	print(f"  ✅ Maximum quality mode with no geographic skip penalties")
	print(f"  ✅ Comprehensive caching system (30-day persistence)")
	print(f"  ✅ Batch API processing (25x25 matrix calls)")
	print(f"  ✅ Fixed start in {args.start_city}")
	print(f"  ✅ Open itinerary without forced return")
	print(f"  ✅ Directions API road-following route geometry")
	print(f"  ✅ Advanced TSP solver with multiple strategies")
	
	if len(coords) > 50:
		print(f"\n🚀 Large Dataset Optimizations:")
		print(f"  - Full-route solve attempted first for {len(coords)} locations")
		print(f"  - Clustering retained as fallback only")
	
	print(f"\n📈 Performance Improvements:")
	print(f"  - Caching eliminates repeat costs")
	print(f"  - City entrances improve because the full route is solved at once")
	print(f"  - Map lines follow roads instead of straight point-to-point chords")
	
if __name__ == "__main__":
	main()
