# Route Optimization Tool

Sample file is upstate art weekend 2026.
Final map: https://www.google.com/maps/d/edit?mid=1Itebu15hqRqvCmXIoNYEkuUrp3_EF-k&usp=sharing

A Python tool that extracts locations from KML/KMZ files and finds the optimal
traveling-salesman route with minimum driving distance. Routing, geocoding, and
road-following geometry are powered by [OpenRouteService](https://openrouteservice.org/)
(ORS), which is free within generous daily limits.


## Features

- **KML/KMZ File Processing**: Extracts addresses and coordinates from Google Earth files
- **Geocoding with caching**: Converts addresses to GPS coordinates, cached for 30 days
- **TSP Route Optimization**: Finds the shortest route visiting all locations
- **Road-following geometry**: Draws the final route along real roads
- **Multiple Output Formats**: Generates KML, GeoJSON, and KMZ
- **Free**: Uses the OpenRouteService free tier — no per-request cost


## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dependencies
- `openrouteservice` - OpenRouteService API client (routing, geocoding, directions)
- `ortools` - OR-Tools constraint solver, used for the TSP optimization
- `googlemaps` - used only as an encoded-polyline decoder for route geometry
- `python-dotenv` - Environment variable management
- `zipfile`, `xml.etree.ElementTree` - Built-in Python libraries

## Setup

1. Get a free API key from [OpenRouteService](https://openrouteservice.org/dev/#/signup).
   The free tier allows ~1,000 geocodes/day, 500 matrix requests/day, and
   2,000 directions requests/day.
2. Copy the example environment file and add your key:
   ```bash
   cp .env.example .env
   ```
   ```bash
   # .env
   ORS_API_KEY=your-actual-openrouteservice-api-key
   ```

## Usage

Inputs and outputs are organized by year under `inputs/<year>/` and `outputs/<year>/`.

### 1. One-time setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then add your ORS_API_KEY
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
venue, uses an open itinerary, solves against directed ORS driving distances,
and draws road-following paths with the ORS Directions service. Geocoding,
distance, and directions results are cached in `cache/`, so re-runs are fast and
free.

> **First run is slow.** ORS rate-limits the free tier, so the tool sleeps
> between calls (geocoding, matrix, and directions). A fresh ~165-location run
> takes roughly 10–15 minutes. Once the caches are warm, subsequent runs finish
> in seconds.

Optional flags:

```bash
.venv/bin/python run.py inputs/2026/upstate_art_weekend_2026_full.kmz outputs/2026/optimized_route_2026 \
  --start-city Kingston \
  --start-mode first-listed \
  --route-shape open \
  --quality maximum \
  --directions-paths
```

### 3. Import into Google My Maps

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

### File Structure
```
project/
├── run.py                    # Route optimizer (OpenRouteService)
├── route_utils.py            # Shared helpers
├── .env                      # Your API key (DO NOT COMMIT)
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
│   ├── distance_cache.pkl    # Cached distance calculations
│   └── directions_cache.pkl  # Cached road-following geometry
└── kmz_extracted/            # Scratch dir (gitignored; regenerated each run)
```

## How It Works

### Step-by-Step Process

#### 1. **KML/KMZ Extraction**
- Extracts and parses KML files from KMZ archives
- Supports both address-based and coordinate-based locations
- Handles ExtendedData fields for complex KML structures

#### 2. **Geocoding**
- Converts addresses to GPS coordinates via the ORS Pelias geocoder
- Caches results for 30 days to avoid repeat API calls
- Handles direct coordinates without geocoding

#### 3. **Distance Matrix Creation**
- **Directed Driving Matrix**: Calculates A→B independently from B→A for better real-world routing
- **Maximum Quality Mode**: Uses real ORS driving distances across the full dataset by default
- **Rate-limited batching**: One matrix call per origin, spaced to respect ORS limits
- **Comprehensive Caching**: Stores all distance calculations
- On a failed call, falls back to a straight-line (haversine) estimate for that leg

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
- **KMZ**: Zipped KML for Google My Maps import
- All include route order, waypoints, and road-following path visualization

## Road-Following Geometry
- Uses the ORS Directions service after optimization to draw the final route along roads
- Falls back to straight segments only for individual legs where directions fail
- Stores decoded directions paths in `cache/directions_cache.pkl`

## Output Files

### GeoJSON Format
- Compatible with web mapping libraries (Leaflet, Mapbox)
- Includes route order and waypoint properties
- Color-coded markers for start/end points

### KML / KMZ Format
- Compatible with Google Earth and Google My Maps
- Styled waypoints with route order
- Complete road-following path visualization

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

## OpenRouteService Limits

The free tier is generous but rate-limited. Approximate daily quotas:
- **Geocoding**: ~1,000 requests/day
- **Distance Matrix**: ~500 requests/day
- **Directions**: ~2,000 requests/day

The tool sleeps between calls to stay under the per-minute limits, and caches
every result, so only the first run on a new dataset consumes meaningful quota.

## Troubleshooting

### Common Issues

#### "No KML file found"
- Ensure your KMZ file is properly formatted
- Check that it contains a valid KML file inside

#### "ORS_API_KEY not set"
- Verify your key is present in `.env`
- Get a free key at https://openrouteservice.org/dev/#/signup

#### "Rate limit exceeded"
- The tool includes automatic rate limiting, but a very large fresh dataset can
  still exceed the daily free-tier quota
- Re-run later; cached results from the partial run are reused for free

#### Memory issues with large datasets
- Clustering optimization kicks in automatically for large datasets
- Consider splitting very large datasets

### Performance Tips
1. **Use caching**: Re-run the same dataset for free in seconds
2. **Start small**: Test with a subset of locations first
3. **Be patient on the first run**: Rate limiting makes the initial pass slow

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
- Verify your ORS key is working
- Check output file generation


## Support

For issues and questions:
- Check the troubleshooting section above
- Review the [OpenRouteService documentation](https://openrouteservice.org/dev/#/api-docs)
- Submit issues with sample data (no API keys)
