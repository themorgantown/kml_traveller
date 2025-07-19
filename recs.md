# Route Optimization Tool - Improvement Recommendations (Updated)

This document provides an updated analysis of the route optimization tool, assessing the original recommendations against the current state of `run.py`.

---

## Easy Improvements

### 1. Centralize Configuration Management
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The script still uses hardcoded default values for parameters within function definitions (e.g., `max_cluster_size`, `cache_duration_days`). This makes configuration changes difficult and error-prone.
- **Developer Notes**:
  - Create a `config.py` file.
  - Move the following parameters from `run.py` into `config.py`:
    ```python
    # config.py
    # TSP Solver Configuration
    MAX_CLUSTER_SIZE = 30
    TSP_TIME_LIMITS = {
        'small': 30,   # up to 10 locations
        'medium': 60,  # up to 20 locations
        'large': 90,   # up to 35 locations
        'xlarge': 60   # over 35 locations
    }

    # API and Data Configuration
    MAX_REASONABLE_DISTANCE_M = 200000  # 200km
    CACHE_DURATION_DAYS = 30
    KMZ_EXTRACT_DIR = "kmz_extracted"
    CACHE_DIR = "cache"

    # Output file names
    GEOJSON_OUTPUT_FILE = "optimized_route.geojson"
    KML_OUTPUT_FILE = "optimized_route.kml"
    ```
  - In `run.py`, import these values: `from config import MAX_CLUSTER_SIZE, CACHE_DIR`.

### 2. Improve Error Handling for API Calls
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The script uses basic `try...except` blocks but lacks a robust retry mechanism for transient network or API errors.
- **Developer Notes**:
  - Use a library like `tenacity` to add exponential backoff to API-calling functions.
  - Install it: `pip install tenacity`.
  - Apply it to the `geocode_addresses` and `create_distance_matrix` functions where `gmaps.geocode` and `gmaps.distance_matrix` are called.
  - **Example**:
    ```python
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    import googlemaps.exceptions

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((googlemaps.exceptions.ApiError, googlemaps.exceptions.Timeout, googlemaps.exceptions.TransportError))
    )
    def call_google_maps_api(...):
        # Place the gmaps.distance_matrix or gmaps.geocode call here
        ...
    ```

### 3. Add Requirements Version Pinning
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The `requirements.txt` file lacks specific versions, which can lead to breaking changes if dependencies are updated.
- **Developer Notes**:
  - Generate a pinned `requirements.txt` file from your current working environment.
  - Run `pip freeze > requirements.txt`.
  - The resulting file should look like this (versions are examples):
    ```text
    googlemaps==4.10.0
    ortools==9.8.3296
    python-dotenv==1.0.0
    # Add other dependencies like tenacity
    ```

---

## Medium Improvements

### 1. Create Shared Utility Module
- **Status**: `PARTIALLY IMPLEMENTED`
- **Analysis**: A `route_utils.py` file exists, but `run.py` does not use it. It contains its own local implementations of functions that should be shared.
- **Developer Notes**:
  - **Consolidate**: Move the following functions from `run.py` into `route_utils.py`, ensuring there is only one version:
    - `haversine_distance`
    - `calculate_centroid`
    - `find_optimal_start_location`
    - `should_skip_distance_calculation`
    - `cluster_locations`
    - `extract_addresses_from_kmz`
    - `save_geojson`
    - `save_kml`
  - **Refactor**: Update `run.py` to import these functions from `route_utils` (e.g., `from route_utils import haversine_distance, cluster_locations`).

### 2. Implement Proper Logging
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The script uses `print()` for all output, which is inflexible. A proper logging setup is needed for better diagnostics.
- **Developer Notes**:
  - Use the `logging` module.
  - **Setup**: Configure logging at the start of `main()` in `run.py`.
    ```python
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("route_optimization.log"),
            logging.StreamHandler() # To also print to console
        ]
    )
    ```
  - **Usage**: Replace `print()` calls with `logging.info()`, `logging.warning()`, or `logging.error()`. For example, `logging.info("Solving TSP for %d locations...", tsp_size)`.

### 3. Add Input Validation
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The script does not validate the KML file's structure, making it vulnerable to errors from malformed inputs.
- **Developer Notes**:
  - Use a library like `lxml` for schema validation.
  - **Action**: In `extract_addresses_from_kmz`, before parsing, validate the KML file against the official OGC KML 2.2 schema. This will require downloading the XSD files from OGC.
  - This will prevent runtime errors if, for example, a `<Placemark>` is missing a required tag.

---

## Advanced Improvements

### 1. Implement Parallel Processing
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The distance matrix calculation, when it needs to call the API, is sequential. This is a major performance bottleneck for uncached datasets.
- **Developer Notes**:
  - Use `concurrent.futures.ThreadPoolExecutor` to make parallel API calls in `create_distance_matrix`.
  - The target for parallelization is the loop that processes `pairs_to_calculate`.
  - **Example Structure**:
    ```python
    from concurrent.futures import ThreadPoolExecutor

    # In create_distance_matrix...
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create futures for each batch or origin group
        future_to_pair = {executor.submit(gmaps.distance_matrix, ...): pair for pair in ...}
        for future in concurrent.futures.as_completed(future_to_pair):
            # Process result
            ...
    ```

### 2. Add Alternative Routing Engines
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The logic is tightly coupled to `googlemaps` and `ortools`.
- **Developer Notes**:
  - Define an abstract base class for a routing engine.
    ```python
    from abc import ABC, abstractmethod

    class RoutingService(ABC):
        @abstractmethod
        def get_distance_matrix(self, locations):
            pass

        @abstractmethod
        def solve_tsp(self, matrix, locations):
            pass
    ```
  - Create concrete implementations: `class GoogleORToolsRouter(RoutingService): ...`.
  - The `main` function would then instantiate a specific router, making the system pluggable.

### 3. Implement Time-Based Routing
- **Status**: `NOT IMPLEMENTED`
- **Analysis**: The TSP solver is configured to optimize for distance only. The Google Maps API returns duration, but it is not used.
- **Developer Notes**:
  - **Modify `create_distance_matrix`**: This function currently only returns a distance matrix. It should be updated to also return a *duration matrix*. The Google Maps API response contains a `duration` field in each element. Store this in a separate matrix.
  - **Update `solve_tsp`**: Add a parameter to `solve_tsp` to let the user choose the optimization metric (e.g., `optimize_by='distance'`).
  - **Update `distance_callback`**: Based on the `optimize_by` parameter, the callback should return a value from either the distance matrix or the duration matrix.
    ```python
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if optimize_by == 'time':
            return duration_matrix[from_node][to_node]
        else:
            return distance_matrix[from_node][to_node]
    ```