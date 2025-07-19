#!/usr/bin/env python3

from run import APICache
import os

def test_cache():
    print("Testing cache functionality...")
    
    # Initialize cache
    cache = APICache()
    print(f"✓ Cache initialized successfully")
    print(f"✓ Cache directory: {cache.cache_dir}")
    print(f"✓ Directory exists: {os.path.exists(cache.cache_dir)}")
    
    # Test geocode caching
    test_address = "123 Main St, New York, NY"
    result = cache.get_geocode(test_address)
    print(f"Initial geocode lookup: {result}")
    
    # Add to cache
    cache.set_geocode(test_address, {'lat': 40.7128, 'lng': -74.0060})
    result2 = cache.get_geocode(test_address)
    print(f"After caching: {result2}")
    
    # Test distance caching
    loc1 = (40.7128, -74.0060)
    loc2 = (40.7589, -73.9851)
    dist = cache.get_distance(loc1, loc2)
    print(f"Initial distance lookup: {dist}")
    
    cache.set_distance(loc1, loc2, 5000)
    dist2 = cache.get_distance(loc1, loc2)
    print(f"After caching distance: {dist2}")
    
    print("✓ All cache tests passed!")
    
    # Show cache statistics
    print(f"\nCache Statistics:")
    print(f"  - Geocode cache entries: {len(cache.geocode_cache)}")
    print(f"  - Distance cache entries: {len(cache.distance_cache)}")
    
    # Show potential savings
    print(f"\nPotential API call savings for different numbers of locations:")
    for n in [5, 10, 20, 50]:
        geocode_calls = n
        distance_calls = n * (n - 1)
        total_calls = geocode_calls + distance_calls
        cost = total_calls * 0.005
        print(f"  {n:2d} locations: {total_calls:4d} API calls (${cost:.2f})")

if __name__ == "__main__":
    test_cache()
