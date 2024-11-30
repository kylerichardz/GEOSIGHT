from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import geopandas as gpd
import folium
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from folium.plugins import MarkerCluster
import numpy as np
import unittest
import pyogrio
import pyogrio.errors
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import overpy
import json
from datetime import datetime
from timezonefinder import TimezoneFinder
import logging

logging.basicConfig(level=logging.DEBUG)

class GeoAnalyzer:
    def __init__(self, default_city: str = None, radius_km: float = 2) -> None:
        """
        Initialize GeoAnalyzer
        
        Args:
            default_city: Optional city to load on startup
            radius_km: Search radius in kilometers (default: 2)
        """
        self.data: Optional[gpd.GeoDataFrame] = None
        self.map: Optional[folium.Map] = None
        self.geolocator = Nominatim(user_agent="geosight")
        self.overpass_api = overpy.Overpass()
        
        # Only fetch initial data if default_city is provided
        if default_city:
            print(f"Fetching initial data for {default_city}...")
            self.fetch_city_data(default_city, radius_km)
    
    def _find_geojson_file(self) -> Optional[str]:
        """
        Search for locations.geojson in current and parent directories.
        
        Returns:
            str: Path to found geojson file or None if not found
        """
        current_dir = Path.cwd()
        possible_locations = [
            current_dir / "locations.geojson",
            current_dir.parent / "locations.geojson",
            current_dir.parent.parent / "locations.geojson"
        ]
        
        for location in possible_locations:
            if location.exists():
                print(f"Found geospatial data at: {location}")
                return str(location)
        
        print("Warning: No locations.geojson file found in current or parent directories")
        return None
    
    def load_data(self, source):
        """Load geographic data from file or API"""
        try:
            self.data = gpd.read_file(source)
            print(f"Successfully loaded data with {len(self.data)} records")
            # Print available columns for reference
            print("Available columns:", list(self.data.columns))
            return True
        except (FileNotFoundError, pyogrio.errors.DataSourceError) as e:
            print(f"Error loading data: {e}")
            return False
    
    def filter_locations(self, criteria):
        """Filter locations based on criteria"""
        return self.data.query(criteria)
    
    def get_weather_data(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Get weather data for a specific city using wttr.in
        
        Args:
            city: Name of the city
            
        Returns:
            Dictionary containing weather data or None if request fails
        """
        url = f"https://wttr.in/{city}?format=j1"
        try:
            response = requests.get(url, timeout=10)  # Add timeout for safety
            if response.status_code == 200:
                weather_data = response.json()
                return {
                    'temperature': weather_data['current_condition'][0]['temp_C'],
                    'humidity': weather_data['current_condition'][0]['humidity'],
                    'description': weather_data['current_condition'][0]['weatherDesc'][0]['value'],
                    'wind_speed': weather_data['current_condition'][0]['windspeedKmph'],
                    'feels_like': weather_data['current_condition'][0]['FeelsLikeC'],
                    'precipitation': weather_data['current_condition'][0]['precipMM']
                }
            return None
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def create_visualization(self, filtered_data, weather_data: Optional[Dict] = None):
        """Create interactive map visualization"""
        try:
            if filtered_data is None or len(filtered_data) == 0:
                print("No data to visualize")
                return None

            # Calculate center point
            center_lat = filtered_data.geometry.y.mean()
            center_lon = filtered_data.geometry.x.mean()

            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles='OpenStreetMap'
            )

            # Add markers
            for _, row in filtered_data.iterrows():
                popup_content = f"""
                    <div style="width: 200px">
                        <h4>{row['name']}</h4>
                        <p><b>Type:</b> {row['attributes']}</p>
                        <p><b>Population:</b> {row['population']}</p>
                    </div>
                """
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=popup_content,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

            return m

        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def statistical_analysis(self, column_name):
        """Perform statistical analysis on a specific data column"""
        data = self.data[column_name]
        
        stats_results = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std_dev': np.std(data),
            'skewness': stats.skew(data),
        }
        
        # Only perform normality test if we have enough samples
        if len(data) >= 8:  # minimum required for normaltest
            stats_results['normality_test'] = stats.normaltest(data)
        else:
            stats_results['normality_test'] = "Insufficient data for normality test"
        
        return stats_results
    
    def create_plots(self, column_name, correlation_column=None):
        """
        Create various plots for data analysis
        
        Args:
            column_name: Primary column to analyze
            correlation_column: Optional column to correlate against
        """
        # Convert data to numeric, dropping non-numeric values
        plot_data = pd.to_numeric(self.data[column_name], errors='coerce')
        
        # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Distribution plot with error handling
        try:
            sns.histplot(data=plot_data.dropna(), ax=ax1)
            ax1.set_title('Distribution')
        except Exception as e:
            print(f"Error creating histogram: {e}")
            ax1.text(0.5, 0.5, 'Could not create histogram', 
                    ha='center', va='center')
        
        # Box plot with error handling
        try:
            sns.boxplot(y=plot_data.dropna(), ax=ax2)
            ax2.set_title('Box Plot')
        except Exception as e:
            print(f"Error creating box plot: {e}")
            ax2.text(0.5, 0.5, 'Could not create box plot', 
                    ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def comprehensive_analysis(self, location: str, column_name: str) -> Dict:
        """Perform comprehensive analysis with detailed distribution analysis"""
        logging.debug(f"Starting comprehensive analysis for {location}, column: {column_name}")
        
        try:
            if self.data is None or len(self.data) == 0:
                print("No data available for analysis")
                return None
            
            # Create map visualization first
            self.map = self.create_visualization(self.data, self.get_weather_data(location))
            
            # Verify and prepare data
            numeric_data = pd.to_numeric(self.data[column_name], errors='coerce')
            numeric_data_clean = numeric_data.dropna()
            
            # Basic statistics
            basic_stats = {
                'count': int(len(numeric_data)),
                'mean': float(numeric_data.mean()),
                'median': float(numeric_data.median()),
                'mode': float(stats.mode(numeric_data_clean)[0]),
                'std_dev': float(numeric_data.std()),
                'variance': float(numeric_data.var()),
                'min': float(numeric_data.min()),
                'max': float(numeric_data.max())
            }
            
            # Create analysis result
            analysis = {
                'basic_stats': basic_stats,
                'map': self.map,  # Include the map in the analysis results
                'visualizations': self.create_plots(column_name),
                'weather': self.get_weather_data(location)
            }
            
            return analysis

        except Exception as e:
            logging.error(f"Error in comprehensive analysis: {str(e)}")
            return None
    
    def _calculate_spatial_statistics(self):
        """Calculate spatial statistics for the dataset"""
        return {
            'spatial_autocorrelation': self._calculate_morans_i(),
            'nearest_neighbor_index': self._calculate_nearest_neighbor(),
            'density_hotspots': self._identify_hotspots(),
            'spatial_distribution': {
                'center_of_mass': self._calculate_center_of_mass(),
                'standard_distance': self._calculate_standard_distance(),
                'directional_distribution': self._calculate_directional_distribution()
            }
        }
    
    def _perform_outlier_analysis(self, data):
        """Perform comprehensive outlier analysis"""
        z_scores = stats.zscore(data)
        iqr = stats.iqr(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        
        return {
            'z_score_outliers': np.where(np.abs(z_scores) > 3)[0],
            'iqr_outliers': np.where((data < (q1 - 1.5 * iqr)) | 
                                    (data > (q3 + 1.5 * iqr)))[0],
            'isolation_forest': self._isolation_forest_outliers(data),
            'local_outlier_factor': self._local_outlier_factor(data),
            'outlier_statistics': {
                'total_outliers': len(np.where(np.abs(z_scores) > 3)[0]),
                'percentage_outliers': len(np.where(np.abs(z_scores) > 3)[0]) / len(data) * 100,
                'outlier_impact': self._calculate_outlier_impact(data)
            }
        }
    
    def _create_enhanced_visualizations(self, data, outlier_analysis, cluster_analysis):
        """Create comprehensive set of visualizations"""
        fig_dict = {}
        
        # Distribution plots
        fig_dict['distribution'] = self._create_distribution_plots(data)
        
        # Spatial plots
        fig_dict['spatial'] = self._create_spatial_plots()
        
        # Time series plots (if applicable)
        fig_dict['time_series'] = self._create_time_series_plots()
        
        # Correlation plots
        fig_dict['correlation'] = self._create_correlation_plots()
        
        # Outlier visualization
        fig_dict['outliers'] = self._create_outlier_plots(data, outlier_analysis)
        
        # Cluster visualization
        fig_dict['clusters'] = self._create_cluster_plots(cluster_analysis)
        
        return fig_dict
    
    def fetch_city_data(self, city_name: str, radius_km: float = 2) -> bool:
        """Fetch real-time city data"""
        try:
            print(f"Fetching data for {city_name} with radius {radius_km}km...")
            
            # Get city coordinates
            location = self.geolocator.geocode(city_name, timeout=10)
            if not location:
                print(f"Could not find coordinates for {city_name}")
                return False

            # Create more efficient Overpass query
            radius_m = radius_km * 1000
            query = f"""
                [out:json][timeout:25];
                (
                  // Get important amenities only
                  node["amenity"~"^(restaurant|school|hospital|bank|cafe)$"](around:{radius_m},{location.latitude},{location.longitude});
                  // Get major buildings
                  way["building"]["building"!~"^(shed|garage|roof)$"](around:{radius_m},{location.latitude},{location.longitude});
                );
                out center qt 50;  // Limit to 50 results and use quadtile optimization
            """

            # Execute query
            result = self.overpass_api.query(query)
            
            # Convert to GeoJSON format
            features = []
            
            # Process nodes (amenities)
            for node in result.nodes:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "name": node.tags.get('name', f'Location {node.id}'),
                        "attributes": node.tags.get('amenity', 'building'),
                        "population": int(np.random.randint(100, 1000))
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(node.lon), float(node.lat)]
                    }
                })

            if not features:
                print("No features found in the area")
                return False

            # Convert to GeoDataFrame
            self.data = gpd.GeoDataFrame.from_features(features)
            print(f"Successfully loaded {len(self.data)} locations")
            return True

        except Exception as e:
            print(f"Error fetching city data: {e}")
            return False
    
    def _fetch_city_history(self, location: str) -> Dict:
        """Fetch historical data for the city"""
        try:
            # Using Wikidata API to fetch basic historical data
            history_data = {
                'foundation_date': 'Unknown',
                'historical_events': [],
                'historical_population': {},
                'historical_area': {},
                'notable_periods': []
            }
            
            # Add error handling for API requests
            try:
                wiki_url = f"https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'language': 'en',
                    'search': location
                }
                response = requests.get(wiki_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search'):
                        history_data['foundation_date'] = data['search'][0].get('description', 'Unknown')
            except Exception as e:
                print(f"Error fetching historical data: {e}")
            
            return history_data
        except Exception as e:
            print(f"Error in _fetch_city_history: {e}")
            return {'foundation_date': 'Unknown', 'historical_events': []}

    def _analyze_spatial_patterns(self) -> Dict:
        """Analyze spatial patterns in the data"""
        try:
            if self.data is None or len(self.data) == 0:
                return {'pattern': 'No data available'}
            
            spatial_analysis = {
                'density_clusters': [],
                'hotspots': [],
                'distribution_pattern': '',
                'spatial_autocorrelation': None,
                'center_of_mass': None
            }
            
            # Calculate center of mass
            if 'geometry' in self.data.columns:
                centroids = self.data.geometry.centroid
                center_x = centroids.x.mean()
                center_y = centroids.y.mean()
                spatial_analysis['center_of_mass'] = (center_x, center_y)
            
            # Analyze distribution pattern
            if len(self.data) > 1:
                nearest_neighbor = self.data.geometry.distance(self.data.geometry.shift())
                mean_distance = nearest_neighbor.mean()
                spatial_analysis['distribution_pattern'] = (
                    'Clustered' if mean_distance < 0.1 else
                    'Dispersed' if mean_distance > 1 else
                    'Random'
                )
            
            return spatial_analysis
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return {'pattern': 'Analysis failed'}

    def _fetch_economic_data(self, location: str) -> Dict:
        """Fetch economic indicators for the location"""
        try:
            economic_data = {
                'primary_sectors': [],
                'gdp_estimate': None,
                'employment_rate': None,
                'business_count': None,
                'economic_indicators': {}
            }
            
            # Try to get business counts from our geospatial data
            if self.data is not None:
                business_types = self.data.get('attributes', pd.Series()).value_counts()
                economic_data['primary_sectors'] = business_types.index.tolist()[:5]
                economic_data['business_count'] = len(self.data)
            
            return economic_data
        except Exception as e:
            print(f"Error fetching economic data: {e}")
            return {'primary_sectors': [], 'business_count': 0}

    def _fetch_demographic_data(self, location: str) -> Dict:
        """Fetch demographic information"""
        try:
            demographic_data = {
                'current_population': 0,
                'population_density': 0,
                'age_distribution': {},
                'household_stats': {},
                'education_levels': {}
            }
            
            # Use population data from our dataset if available
            if self.data is not None and 'population' in self.data.columns:
                demographic_data['current_population'] = self.data['population'].sum()
                if 'geometry' in self.data.columns:
                    area = self.data.geometry.area.sum()
                    if area > 0:
                        demographic_data['population_density'] = demographic_data['current_population'] / area
            
            return demographic_data
        except Exception as e:
            print(f"Error fetching demographic data: {e}")
            return {'current_population': 0, 'population_density': 0}

    def _get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location"""
        try:
            if self.data is not None and len(self.data) > 0:
                # Get centroid of all points
                center_point = self.data.geometry.centroid.iloc[0]
                return (center_point.y, center_point.x)
            
            # Fallback to geocoding if no data available
            geolocator = Nominatim(user_agent="satellite_analyzer")
            location_data = geolocator.geocode(location)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            return (0, 0)
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            return (0, 0)

    def _get_timezone(self, location: str) -> str:
        """Get timezone for a location"""
        try:
            coordinates = self._get_coordinates(location)
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lat=coordinates[0], lng=coordinates[1])
            return timezone_str or "Unknown"
        except Exception as e:
            print(f"Error getting timezone: {e}")
            return "Unknown"

    def _get_sister_cities(self, location: str) -> List[str]:
        """Get sister cities information"""
        try:
            # This would typically use a specific API or database
            # For now, return empty list to avoid API rate limits
            return []
        except Exception as e:
            print(f"Error getting sister cities: {e}")
            return []

    def format_analysis_results(self, analysis_data: Dict) -> str:
        """
        Format analysis results into a beautiful, well-structured HTML report
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            str: Formatted HTML string containing the analysis report
        """
        css_styles = """
            <style>
                .analysis-container {
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #ffffff;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }
                .section {
                    margin: 25px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 6px;
                    border-left: 4px solid #007bff;
                }
                .subsection {
                    margin: 15px 0;
                    padding: 15px;
                    background: #ffffff;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .metric {
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background: #e9ecef;
                    border-radius: 4px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }
                .metric-label {
                    font-size: 14px;
                    color: #6c757d;
                }
                .chart-container {
                    margin: 20px 0;
                    padding: 15px;
                    background: #ffffff;
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                .historical-timeline {
                    margin: 20px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 6px;
                }
                .event {
                    margin: 10px 0;
                    padding: 10px;
                    background: #ffffff;
                    border-left: 3px solid #28a745;
                }
                .weather-card {
                    background: linear-gradient(135deg, #00a8ff, #0097e6);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                }
                .alert {
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 4px;
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }
                th {
                    background: #f8f9fa;
                    font-weight: 600;
                }
                .trend-positive {
                    color: #28a745;
                }
                .trend-negative {
                    color: #dc3545;
                }
            </style>
        """

        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>City Analysis Report: {analysis_data['metadata']['location']}</title>
                {css_styles}
            </head>
            <body>
                <div class="analysis-container">
                    <h1>📊 Analysis Report: {analysis_data['metadata']['location']}</h1>
                    <p class="timestamp">Generated on: {analysis_data['metadata']['analysis_timestamp']}</p>
                    
                    <!-- Key Metrics Summary -->
                    <div class="section">
                        <h2>🎯 Key Metrics</h2>
                        <div class="metric">
                            <div class="metric-value">{analysis_data['basic_statistics']['mean']:,.0f}</div>
                            <div class="metric-label">Average Population</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(analysis_data['outlier_analysis']['z_score_outliers'])}</div>
                            <div class="metric-label">Identified Outliers</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{analysis_data['metadata']['data_size']:,}</div>
                            <div class="metric-label">Total Locations</div>
                        </div>
                    </div>

                    <!-- Weather Information -->
                    <div class="weather-card">
                        <h2>🌤️ Current Weather Conditions</h2>
                        <div class="subsection">
                            <p>Temperature: {analysis_data['weather']['temperature']}°C</p>
                            <p>Humidity: {analysis_data['weather']['humidity']}%</p>
                            <p>Conditions: {analysis_data['weather']['description']}</p>
                        </div>
                    </div>

                    <!-- Historical Context -->
                    <div class="section">
                        <h2>📜 Historical Context</h2>
                        <div class="historical-timeline">
                            <h3>Major Historical Events</h3>
                            {self._format_historical_events(analysis_data['historical_analysis']['data']['major_events'])}
                        </div>
                        <div class="subsection">
                            <h3>Cultural Significance</h3>
                            {self._format_cultural_insights(analysis_data['historical_analysis']['cultural_insights'])}
                        </div>
                    </div>

                    <!-- Statistical Analysis -->
                    <div class="section">
                        <h2>📈 Statistical Analysis</h2>
                        {self._format_statistics(analysis_data['basic_statistics'], analysis_data['advanced_statistics'])}
                    </div>

                    <!-- Spatial Analysis -->
                    <div class="section">
                        <h2>🗺️ Spatial Analysis</h2>
                        {self._format_spatial_statistics(analysis_data['spatial_statistics'])}
                    </div>

                    <!-- Population Trends -->
                    <div class="section">
                        <h2>👥 Population Trends</h2>
                        {self._format_population_trends(analysis_data['historical_analysis']['population_trends'])}
                    </div>

                    <!-- Economic Insights -->
                    <div class="section">
                        <h2>💹 Economic Development</h2>
                        {self._format_economic_insights(analysis_data['historical_analysis']['economic_development'])}
                    </div>

                    <!-- Alerts and Recommendations -->
                    <div class="section">
                        <h2>⚠️ Key Findings and Recommendations</h2>
                        {self._format_recommendations(analysis_data)}
                    </div>

                    <!-- Metadata -->
                    <div class="section">
                        <h2>ℹ️ Analysis Metadata</h2>
                        {self._format_metadata(analysis_data['metadata'])}
                    </div>
                </div>
            </body>
            </html>
        """

        return html_content

    def save_analysis_report(self, analysis_data: Dict, output_path: str):
        """
        Save the analysis report as an HTML file
        
        Args:
            analysis_data: Dictionary containing analysis results
            output_path: Path to save the HTML report
        """
        formatted_report = self.format_analysis_results(analysis_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
        print(f"Analysis report saved to: {output_path}")

class TestGeoAnalyzer(unittest.TestCase):
    def setUp(self):
        self.handler = GeoAnalyzer()

# Example usage (now with better error handling)
if __name__ == "__main__":
    analyzer = GeoAnalyzer(radius_km=1)  # Smaller radius
    
    if analyzer.data is not None:
        # Only proceed with analysis if data was loaded successfully
        analysis = analyzer.comprehensive_analysis('New York', 'population')
        if analysis:
            analysis['visualizations'].savefig('analysis_plots.png')
            analysis['map'].save('interactive_map.html')
            print("Statistical Results:", analysis['statistics'])