#!/usr/bin/env python3
"""
Route Optimization Utility Functions

Shared utilities for KMZ processing, geographic calculations, clustering,
and file output generation used by both Google Maps and OpenRouteService versions.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import json
import math
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any


# === Geographic Utility Functions ===

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def calculate_centroid(locations: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Calculate the geographic centroid of a list of (lat, lon) coordinates"""
    if not locations:
        return None
    
    lat_sum = sum(lat for lat, lon in locations)
    lon_sum = sum(lon for lat, lon in locations)
    
    return (lat_sum / len(locations), lon_sum / len(locations))


def find_optimal_start_location(locations: List[Tuple[float, float]]) -> int:
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


def should_skip_distance_calculation(
    loc1: Tuple[float, float], 
    loc2: Tuple[float, float], 
    max_reasonable_distance: float = 200000.0
) -> bool:
    """
    Determine if we should skip distance calculation between two locations
    based on geographic distance. Skip if they're impossibly far apart.
    max_reasonable_distance in meters (default 200km)
    """
    geo_distance = haversine_distance(loc1[0], loc1[1], loc2[0], loc2[1])
    return geo_distance > max_reasonable_distance


# === Clustering Functions ===

def cluster_locations(locations: List[Tuple[float, float]], max_cluster_size: int = 25) -> List[List[int]]:
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


# === KMZ/KML Processing Functions ===

def extract_addresses_from_kmz(kmz_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract addresses and location names from KMZ file
    Returns: (addresses, location_names)
    """
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


# === Caching System ===

class APICache:
    """
    Generic caching system for API results with configurable duration
    Supports both geocoding and distance matrix caching
    """
    
    def __init__(self, cache_dir: str = "cache", cache_duration_days: int = 30):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(days=cache_duration_days)
        self.geocode_cache_file = os.path.join(cache_dir, "geocode_cache.pkl")
        self.distance_cache_file = os.path.join(cache_dir, "distance_cache.pkl")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing caches
        self.geocode_cache = self._load_cache(self.geocode_cache_file)
        self.distance_cache = self._load_cache(self.distance_cache_file)

        print(f"Cache initialized: {len(self.geocode_cache)} geocode entries, {len(self.distance_cache)} distance entries")
    
    def _load_cache(self, cache_file: str) -> Dict[str, Tuple[Any, datetime]]:
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
    
    def _save_cache(self, cache: Dict[str, Tuple[Any, datetime]], cache_file: str):
        """Save cache to file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_file}: {e}")
    
    def _hash_location(self, location: Any) -> str:
        """Create a hash key for a location (lat, lon)"""
        if isinstance(location, (tuple, list)):
            return f"{location[0]:.6f},{location[1]:.6f}"
        return str(location)
    
    def get_geocode(self, address: str) -> Optional[Dict[str, float]]:
        """Get geocoded coordinates from cache or return None"""
        key = hashlib.md5(address.encode()).hexdigest()
        if key in self.geocode_cache:
            data, timestamp = self.geocode_cache[key]
            print(f"  Cache hit for: {address}")
            return data
        return None
    
    def set_geocode(self, address: str, result: Dict[str, float]):
        """Save geocoded result to cache"""
        key = hashlib.md5(address.encode()).hexdigest()
        self.geocode_cache[key] = (result, datetime.now())
        self._save_cache(self.geocode_cache, self.geocode_cache_file)
    
    def get_distance(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[float]:
        """Get distance from cache or return None"""
        key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
        if key in self.distance_cache:
            data, timestamp = self.distance_cache[key]
            return data
        return None
    
    def set_distance(self, origin: Tuple[float, float], destination: Tuple[float, float], result: float):
        """Save distance result to cache"""
        key = f"{self._hash_location(origin)}->{self._hash_location(destination)}"
        self.distance_cache[key] = (result, datetime.now())
        self._save_cache(self.distance_cache, self.distance_cache_file)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics"""
        return {
            'geocode_entries': len(self.geocode_cache),
            'distance_entries': len(self.distance_cache),
            'cache_directory': self.cache_dir,
            'cache_duration_days': self.cache_duration.days
        }


# === File Output Generation ===

def save_geojson(
    locations: List[Tuple[float, float]], 
    route: List[int], 
    location_names: Optional[List[str]] = None, 
    output_file: str = "optimized_route.geojson"
):
    """
    Save route as GeoJSON file with points and route lines
    """
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
    for i in range(len(route) - 1):
        start_idx = route[i]
        end_idx = route[i + 1]
        start = locations[start_idx]
        end = locations[end_idx]
        
        start_name = location_names[start_idx] if location_names and start_idx < len(location_names) else f"Location_{start_idx+1}"
        end_name = location_names[end_idx] if location_names and end_idx < len(location_names) else f"Location_{end_idx+1}"
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start[1], start[0]], [end[1], end[0]]]
            },
            "properties": {
                "from": start_name,
                "to": end_name,
                "from_index": start_idx,
                "to_index": end_idx,
                "segment": i + 1,
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


def save_kml(
    locations: List[Tuple[float, float]], 
    route: List[int], 
    location_names: Optional[List[str]] = None, 
    output_file: str = "optimized_route.kml"
):
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
            
            style_id = "startPoint" if i == 0 else "routePoint"
            
            kml_content.append('      <Placemark>')
            kml_content.append(f'        <name>{i + 1}. {location_name}</name>')
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
    for route_idx in route:
        if route_idx < len(locations):
            lat, lon = locations[route_idx]
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
    print(f"  - Complete route path with {len(route)} segments")


# === Geocoding Utilities ===

def process_coordinate_string(address: str) -> Optional[Tuple[float, float]]:
    """
    Process coordinate string in COORDINATES:lat,lon format
    Returns (lat, lon) tuple or None if invalid
    """
    if address.startswith("COORDINATES:"):
        coord_str = address.replace("COORDINATES:", "")
        try:
            lat, lon = map(float, coord_str.split(','))
            return (lat, lon)
        except (ValueError, IndexError):
            return None
    return None


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges
    Latitude: -90 to 90, Longitude: -180 to 180
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def calculate_route_statistics(
    route: List[int], 
    locations: List[Tuple[float, float]], 
    distance_matrix: List[List[float]]
) -> Dict[str, Any]:
    """
    Calculate comprehensive route statistics
    Returns dictionary with distance, time estimates, and other metrics
    """
    if len(route) < 2:
        return {
            'total_distance': 0,
            'total_segments': 0,
            'average_segment_distance': 0,
            'estimated_travel_time_hours': 0
        }
    
    total_distance = 0
    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i + 1]
        if from_idx < len(distance_matrix) and to_idx < len(distance_matrix[from_idx]):
            total_distance += distance_matrix[from_idx][to_idx]
    
    num_segments = len(route) - 1
    avg_segment_distance = total_distance / num_segments if num_segments > 0 else 0
    
    # Estimate travel time assuming average speed of 50 km/h
    estimated_hours = (total_distance / 1000) / 50
    
    return {
        'total_distance': total_distance,
        'total_segments': num_segments,
        'average_segment_distance': avg_segment_distance,
        'estimated_travel_time_hours': estimated_hours,
        'total_distance_km': total_distance / 1000,
        'average_segment_distance_km': avg_segment_distance / 1000
    }
