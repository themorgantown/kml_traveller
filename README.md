# Route Optimization Tool

Sample file is upstate art weekend 2026.
Final map: https://www.google.com/maps/d/edit?mid=1Itebu15hqRqvCmXIoNYEkuUrp3_EF-k&usp=sharing

A powerful Python tool that extracts locations from KML/KMZ files and finds the optimal traveling salesman route with minimum distance. Features comprehensive cost optimizations and supports both Google Maps and OpenRouteService APIs.


## Features

- **KML/KMZ File Processing**: Extracts addresses and coordinates from Google Earth files
- **Intelligent Geocoding**: Converts addresses to GPS coordinates with caching
- **TSP Route Optimization**: Finds the shortest route visiting all locations
- **Cost Optimization**: Up to 80% API cost reduction through smart optimizations
- **Multiple Output Formats**: Generates both GeoJSON and KML files
- **Dual API Support**: Google Maps API and OpenRouteService



 `run.py` - Advanced Optimization Version 
- **API**: Google Maps (requires API key)
- **Optimizations**: Comprehensive cost-saving features
- **Best for**: Large datasets, production use, cost-conscious applications



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
   - Directions API
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

Inputs and outputs are organized by year under `inputs/<year>/` and `outputs/<year>/`.

### 1. One-time setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then add your GOOGLE_MAPS_API_KEY
```

### 2. Run the optimizer

`run.py` takes two optional arguments: the input KMZ and the output base path
(no extension — it writes `<base>.kml`, `<base>.geojson`, and `<base>.kmz`).
Missing output folders are created automatically.

```bash
# usage: python run.py [input.kmz] [output_base]

# 2026
.venv/bin/python run.py inputs/2026/upstate_art_weekend_2026_full.kmz outputs/2026/optimized_route_2026

# 2025
.venv/bin/python run.py inputs/2025/file.kmz outputs/2025/optimized_route
```

With no arguments it falls back to `file.kmz` → `optimized_route.*` in the
current directory. By default, the route starts at the first listed Kingston
venue, uses an open itinerary, solves against directed Google driving distances,
and draws road-following paths with the Directions API. Geocoding, distance, and
directions results are cached in `cache/`, so re-runs are near-free.

Optional flags:

```bash
.venv/bin/python run.py inputs/2026/upstate_art_weekend_2026_full.kmz outputs/2026/optimized_route_2026 \
  --start-city Kingston \
  --start-mode first-listed \
  --route-shape open \
  --quality maximum \
  --directions-paths
```

### 3. Import into Google Maps (My Maps)

The optimizer writes a `.kmz` automatically. To import into
[Google My Maps](https://www.google.com/maps/d/), use the KMZ because it is a
zipped KML and is smaller than the raw KML.

Then in My Maps: **Create a new map → Import → upload the `.kmz`**.

### Preparing input from a venue list

The yearly input KMZ is built from a raw venue list. The 2026 pipeline
(`inputs/2026/`) shows the steps: a scraped text list (`2026.txt`) → parsed to
`Name,Address` CSV → enriched with coordinates pulled from the source map
(`upstate_art_weekend_2026_full.csv`) → imported into Google My Maps and
exported as `upstate_art_weekend_2026_full.kmz`, which is then fed to `run.py`.

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
├── run.py                    # Route optimizer (Google Maps API)
├── route_utils.py            # Shared helpers
├── .env                      # Your API keys (DO NOT COMMIT)
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependencies
├── inputs/                   # Source data, by year
│   ├── 2025/file.kmz
│   └── 2026/                 # venue list → CSV → KMZ pipeline + final KMZ
├── outputs/                  # Optimized routes, by year
│   ├── 2025/optimized_route.{kml,geojson,md}
│   └── 2026/optimized_route_2026.{kml,geojson,kmz}
├── cache/                    # Auto-created API cache
│   ├── geocode_cache.pkl     # Cached geocoding results
│   └── distance_cache.pkl    # Cached distance calculations
└── kmz_extracted/            # Scratch dir (gitignored; regenerated each run)
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
- **Directed Driving Matrix**: Calculates A→B independently from B→A for better real-world routing
- **Maximum Quality Mode**: Uses real Google driving distances across the full dataset by default
- **Batch Processing**: Groups API calls for efficiency
- **Comprehensive Caching**: Stores all distance calculations

#### 4. **TSP Optimization**
- **Fixed Open Start**: Defaults to the first listed Kingston venue and does not force a return loop
- **Adaptive Algorithms**: Different strategies based on problem size:
  - Small (≤15): Exact algorithms with guided local search
  - Medium (16-50): Simulated annealing for quality/speed balance  
  - Large (>50): Tabu search with clustering for speed
- **Cluster Fallback**: Keeps clustering as a fallback if the full-route solve fails

#### 5. **Output Generation**
- **GeoJSON**: For web mapping applications
- **KML**: For Google Earth visualization
- Both include route order, waypoints, and road-following path visualization

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

### Road-Following Geometry
- Uses the Directions API after optimization to draw the final route along roads
- Falls back to straight segments only for individual legs where Directions fails
- Stores decoded directions paths in `cache/directions_cache.pkl`

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
- **Directions**: used once per final route leg for road-following geometry
- **Rate Limit**: 50 requests/second
- **Optimization**: caching keeps repeat runs inexpensive

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
 

## Support

For issues and questions:
- Check troubleshooting section above
- Review API provider documentation
- Submit issues with sample data (no API keys)

 
