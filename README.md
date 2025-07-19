# Route Optimization Tool

Sample file is upstate art weekend 2025.
Final map: https://www.google.com/maps/d/u/0/edit?mid=1xfY0kFyNcUoP8r05mY_pL_jyWYx4IKw&usp=sharing

A powerful Python tool that extracts locations from KML/KMZ files and finds the optimal traveling salesman route with minimum distance. Features comprehensive cost optimizations and supports both Google Maps and OpenRouteService APIs.


## Features

- **KML/KMZ File Processing**: Extracts addresses and coordinates from Google Earth files
- **Intelligent Geocoding**: Converts addresses to GPS coordinates with caching
- **TSP Route Optimization**: Finds the shortest route visiting all locations
- **Cost Optimization**: Up to 80% API cost reduction through smart optimizations
- **Multiple Output Formats**: Generates both GeoJSON and KML files
- **Dual API Support**: Google Maps API and OpenRouteService

## Two Versions Available

### 1. `run.py` - Advanced Optimization Version (Recommended)
- **API**: Google Maps (requires API key)
- **Optimizations**: Comprehensive cost-saving features
- **Best for**: Large datasets, production use, cost-conscious applications

### 2. `run_ors.py` - Simple OpenRouteService Version
- **API**: OpenRouteService (requires API key)
- **Optimizations**: Basic route optimization
- **Best for**: Small datasets, testing, alternative to Google Maps

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Dependencies
- `googlemaps` - Google Maps API client (for run.py)
- `ortools` - Google's optimization tools for TSP solving
- `openrouteservice` - OpenRouteService API client (for run_ors.py)
- `python-dotenv` - Environment variable management
- `zipfile`, `xml.etree.ElementTree` - Built-in Python libraries

## Setup

### Environment Variables (Recommended)
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your API keys:
   ```bash
   # For Google Maps version (run.py)
   GOOGLE_MAPS_API_KEY=your-actual-google-maps-api-key
   
   # For OpenRouteService version (run_ors.py)
   ORS_API_KEY=your-actual-openrouteservice-api-key
   ```

### Option 1: Google Maps API (run.py)
1. Get a Google Maps API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable these APIs:
   - Geocoding API
   - Distance Matrix API
3. Add your key to `.env` file:
   ```
   GOOGLE_MAPS_API_KEY=your-api-key-here
   ```

### Option 2: OpenRouteService API (run_ors.py)
1. Get a free API key from [OpenRouteService](https://openrouteservice.org/)
2. Add your key to `.env` file:
   ```
   ORS_API_KEY=your-api-key-here
   ```

## Usage

### Basic Usage
1. Place your KML or KMZ file in the same directory as the script
2. Rename it to `file.kmz` or edit the filename in the script
3. Run the optimization:

```bash
# For advanced optimization (recommended)
python run.py

# For OpenRouteService version
python run_ors.py
```

## Security & Best Practices

### API Key Security
- **Never commit API keys to version control**
- Use the `.env` file for local development
- Add `.env` to your `.gitignore` file
- For production deployment, use your platform's environment variable system
- Restrict API keys to specific domains/IPs when possible

### File Structure
```
project/
├── run.py                    # Advanced optimization version
├── run_ors.py               # OpenRouteService version
├── .env                     # Your API keys (DO NOT COMMIT)
├── .env.example             # Template for environment variables
├── requirements.txt         # Python dependencies
├── file.kmz                 # Your input KML/KMZ file
├── cache/                   # Auto-created cache directory
│   ├── geocode_cache.pkl    # Cached geocoding results
│   └── distance_cache.pkl   # Cached distance calculations
├── optimized_route.geojson  # Output: Web mapping format
└── optimized_route.kml      # Output: Google Earth format
```

## How It Works

### Step-by-Step Process

#### 1. **KML/KMZ Extraction**
- Extracts and parses KML files from KMZ archives
- Supports both address-based and coordinate-based locations
- Handles ExtendedData fields for complex KML structures

#### 2. **Geocoding** 
- Converts addresses to GPS coordinates
- Caches results for 30 days to avoid repeat API calls
- Handles direct coordinates without geocoding

#### 3. **Distance Matrix Creation** (run.py only)
- **Symmetric Optimization**: Only calculates A→B, assumes B→A is equal (50% reduction)
- **Geographic Filtering**: Skips impossible long-distance calculations
- **Batch Processing**: Groups API calls for efficiency
- **Comprehensive Caching**: Stores all distance calculations

#### 4. **TSP Optimization**
- **Edge-based Starting Point**: Finds optimal starting location (furthest from centroid)
- **Adaptive Algorithms**: Different strategies based on problem size:
  - Small (≤15): Exact algorithms with guided local search
  - Medium (16-50): Simulated annealing for quality/speed balance  
  - Large (>50): Tabu search with clustering for speed
- **Smart Clustering**: Automatically groups nearby locations for large datasets

#### 5. **Output Generation**
- **GeoJSON**: For web mapping applications
- **KML**: For Google Earth visualization
- Both include route order, waypoints, and path visualization

## Advanced Features (run.py)

### Cost Optimizations
- **Symmetric Matrix**: 50% API call reduction
- **Geographic Filtering**: Skip impossible distances
- **Caching System**: 100% savings on repeat runs
- **Batch Processing**: Efficient API usage

### Performance Statistics
For 168 locations:
- **Without optimization**: ~$140 first run
- **With optimization**: ~$67 first run (52% savings)
- **Subsequent runs**: Near $0 (cached)

### Smart Clustering
- Automatically activates for datasets > 50 locations
- Groups nearby locations to reduce complexity
- Maintains solution quality while improving performance

## Output Files

### GeoJSON Format
- Compatible with web mapping libraries (Leaflet, Mapbox)
- Includes route order and waypoint properties
- Color-coded markers for start/end points

### KML Format  
- Compatible with Google Earth
- Styled waypoints with route order
- Complete path visualization

## Configuration Options

### Customizable Parameters
```python
# Cache duration (run.py)
cache = APICache(cache_duration_days=30)

# Clustering threshold (run.py)
max_cluster_size = 50

# Geographic filtering distance (run.py)
max_reasonable_distance = 200000  # meters

# Time limits for TSP solving
time_limit = 300  # seconds
```

## API Rate Limits & Costs

### Google Maps API (run.py)
- **Geocoding**: $0.005 per request
- **Distance Matrix**: $0.005 per element
- **Rate Limit**: 50 requests/second
- **Optimization**: Up to 80% cost reduction

### OpenRouteService (run_ors.py)
- **Free Tier**: 2,000 requests/day
- **Geocoding**: Free
- **Routing**: Free with limits
- **Rate Limit**: 40 requests/minute

## Troubleshooting

### Common Issues

#### "No KML file found"
- Ensure your KMZ file is properly formatted
- Check that it contains a valid KML file inside

#### "API key invalid"
- Verify your API key is correct
- Check that required APIs are enabled (Google Cloud Console)
- Ensure billing is set up for Google Maps API

#### "Rate limit exceeded"
- The tool includes automatic rate limiting
- For large datasets, the process may take time
- Consider using caching between runs

#### Memory issues with large datasets
- Use clustering optimization (automatic in run.py)
- Consider splitting very large datasets

### Performance Tips
1. **Use caching**: Run the same dataset multiple times with near-zero cost
2. **Start small**: Test with a subset of locations first
3. **Monitor costs**: Check API usage in your provider's dashboard
4. **Use run.py for production**: Better optimizations and caching

## Contributing

This is an open-source project. Contributions welcome!

### Development Setup
```bash
git clone <repository>
cd route-optimization-tool
pip install -r requirements.txt
```

### Testing
- Test with small datasets first
- Verify API keys are working
- Check output file generation

## License

[Add your chosen license here]

## Support

For issues and questions:
- Check troubleshooting section above
- Review API provider documentation
- Submit issues with sample data (no API keys)

## Roadmap

Potential future features:
- Additional mapping service APIs
- Real-time traffic consideration
- Multi-vehicle routing
- Web interface
- Docker containerization

---

**Note**: Remember to keep your API keys secure and never commit them to version control!
