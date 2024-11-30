import customtkinter as ctk
from .data_handler import GeoAnalyzer
import tkinter as tk
from tkinter import messagebox
import webbrowser
import folium
import tempfile
import os
import logging
from datetime import datetime
from scipy import stats
import numpy as np

class GeoSightGUI:
    def __init__(self):
        # Add logger initialization
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        
        # Configure window
        self.window = ctk.CTk()
        self.window.title("GeoSight")
        self.window.geometry("800x600")
        
        # Initialize analyzer
        self.analyzer = GeoAnalyzer()
        
        # Create main frame and GUI
        self.setup_gui()
        
        self.map_file = None
    
    def setup_gui(self):
        """Setup the main GUI elements"""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self.header = ctk.CTkLabel(
            self.main_frame, 
            text="GeoSight",
            font=("Helvetica", 24)
        )
        self.header.pack(pady=20)
        
        # Create frames
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=5)
        
        self.search_frame = ctk.CTkFrame(self.main_frame)
        self.search_frame.pack(fill="x", padx=10, pady=5)
        
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # List of popular cities
        self.cities = {
            "Select a city": None,
            "New York, USA": (40.7128, -74.0060),
            "London, UK": (51.5074, -0.1278),
            "Tokyo, Japan": (35.6762, 139.6503),
            "Paris, France": (48.8566, 2.3522),
            "Sydney, Australia": (-33.8688, 151.2093),
            "Dubai, UAE": (25.2048, 55.2708),
            "Singapore": (1.3521, 103.8198),
            "Hong Kong": (22.3193, 114.1694),
            "Mumbai, India": (19.0760, 72.8777),
            "Rio de Janeiro, Brazil": (-22.9068, -43.1729)
        }
        
        # Setup quick select dropdown
        self.quick_select_frame = ctk.CTkFrame(self.search_frame)
        self.quick_select_frame.pack(side="left", padx=5)
        
        self.quick_select_label = ctk.CTkLabel(
            self.quick_select_frame,
            text="Quick Select:"
        )
        self.quick_select_label.pack(side="left", padx=5)
        
        self.city_var = tk.StringVar(value="Select a city")
        self.city_dropdown = ctk.CTkOptionMenu(
            self.quick_select_frame,
            values=list(self.cities.keys()),
            variable=self.city_var,
            command=self.on_city_select
        )
        self.city_dropdown.pack(side="left", padx=5)
        
        # Setup search controls
        self.search_controls_frame = ctk.CTkFrame(self.search_frame)
        self.search_controls_frame.pack(side="left", padx=20)
        
        self.city_label = ctk.CTkLabel(
            self.search_controls_frame,
            text="Or search:"
        )
        self.city_label.pack(side="left", padx=5)
        
        self.city_entry = ctk.CTkEntry(self.search_controls_frame)
        self.city_entry.pack(side="left", padx=5)
        
        self.radius_label = ctk.CTkLabel(
            self.search_controls_frame,
            text="Radius (km):"
        )
        self.radius_label.pack(side="left", padx=5)
        
        self.radius_entry = ctk.CTkEntry(
            self.search_controls_frame,
            width=60
        )
        self.radius_entry.insert(0, "2")
        self.radius_entry.pack(side="left", padx=5)
        
        self.fetch_button = ctk.CTkButton(
            self.search_controls_frame,
            text="Fetch Data",
            command=self.fetch_city_data
        )
        self.fetch_button.pack(side="left", padx=5)
        
        # Setup analysis controls
        self.location_label = ctk.CTkLabel(self.controls_frame, text="Location:")
        self.location_label.pack(side="left", padx=5)
        
        self.location_entry = ctk.CTkEntry(self.controls_frame)
        self.location_entry.pack(side="left", padx=5)
        
        self.column_label = ctk.CTkLabel(self.controls_frame, text="Analyze:")
        self.column_label.pack(side="left", padx=5)
        
        self.column_var = tk.StringVar(value="population")
        self.column_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            variable=self.column_var,
            values=["population"]
        )
        self.column_menu.pack(side="left", padx=5)
        
        self.analyze_button = ctk.CTkButton(
            self.controls_frame,
            text="Analyze",
            command=self.run_analysis
        )
        self.analyze_button.pack(side="left", padx=5)
        
        # Setup results area
        self.stats_text = ctk.CTkTextbox(
            self.results_frame,
            height=300,
            width=700
        )
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Map view button
        self.view_map_button = ctk.CTkButton(
            self.results_frame,
            text="View Map",
            command=self.view_map
        )
        self.view_map_button.pack(pady=5)
        
        # Initial message
        self.stats_text.insert("1.0", "Welcome to GeoSight!\n\nEnter a city name and radius, then click 'Fetch Data' to begin.")
    
    def on_city_select(self, city):
        """Handle city selection from dropdown"""
        if city != "Select a city":
            # Update the search entry with selected city
            self.city_entry.delete(0, tk.END)
            self.city_entry.insert(0, city)
            
            # Auto-fetch data for the selected city
            self.fetch_city_data()
    
    def run_analysis(self):
        """Run the comprehensive analysis"""
        try:
            if not self.analyzer or self.analyzer.data is None:
                messagebox.showerror("Error", "No data available. Please fetch city data first.")
                return

            location = self.city_entry.get()
            if not location:
                messagebox.showerror("Error", "Please enter a city name")
                return

            # Show loading message
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", "Analyzing data...\nPlease wait...")
            self.window.update()

            # Create visualization
            map_obj = self.analyzer.create_visualization(self.analyzer.data)
            if map_obj:
                # Save map
                self.map_file = os.path.abspath('interactive_map.html')
                map_obj.save(self.map_file)
                print(f"Map saved to: {self.map_file}")

                # Display statistics
                stats_text = " Analysis Results\n"
                stats_text += "=" * 40 + "\n\n"
                
                # Location Information
                stats_text += "📍 Location Information\n"
                stats_text += "-" * 20 + "\n"
                stats_text += f"• Analyzing: {location}\n"
                stats_text += f"• Total Locations: {len(self.analyzer.data)}\n"
                stats_text += f"• Types of Places: {self.analyzer.data['attributes'].nunique()}\n\n"
                
                # Population Statistics
                pop_stats = self.analyzer.data['population'].describe()
                stats_text += "👥 Population Statistics\n"
                stats_text += "-" * 20 + "\n"
                stats_text += f"• Average: {pop_stats['mean']:,.0f}\n"
                stats_text += f"• Median: {pop_stats['50%']:,.0f}\n"
                stats_text += f"• Range: {pop_stats['min']:,.0f} - {pop_stats['max']:,.0f}\n"
                stats_text += f"• Standard Deviation: {pop_stats['std']:,.0f}\n\n"
                
                # Distribution Analysis
                stats_text += "📈 Distribution Analysis\n"
                stats_text += "-" * 20 + "\n"
                skewness = self.analyzer.data['population'].skew()
                stats_text += f"• Skewness: {skewness:.2f} "
                if skewness > 1:
                    stats_text += "(Highly right-skewed)\n"
                elif skewness < -1:
                    stats_text += "(Highly left-skewed)\n"
                else:
                    stats_text += "(Approximately symmetric)\n"
                
                # Get weather data
                weather = self.analyzer.get_weather_data(location)
                if weather:
                    stats_text += "\n🌡️ Weather Conditions\n"
                    stats_text += "-" * 20 + "\n"
                    stats_text += f"• Temperature: {weather.get('temperature', 'N/A')}°C\n"
                    stats_text += f"• Feels Like: {weather.get('feels_like', 'N/A')}°C\n"
                    stats_text += f"• Conditions: {weather.get('description', 'N/A')}\n"
                    stats_text += f"• Humidity: {weather.get('humidity', 'N/A')}%\n"
                    stats_text += f"• Wind Speed: {weather.get('wind_speed', 'N/A')} km/h\n"
                
                # Data Quality
                stats_text += "\n📋 Data Quality\n"
                stats_text += "-" * 20 + "\n"
                quality_score = len(self.analyzer.data)
                if quality_score > 100:
                    stats_text += "• Excellent data coverage\n"
                elif quality_score > 50:
                    stats_text += "• Good data coverage\n"
                else:
                    stats_text += "• Limited data coverage\n"
                stats_text += f"• Sample Size: {quality_score} locations\n"

                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert("1.0", stats_text)

                messagebox.showinfo("Success", "Analysis completed! Click 'View Map' to see the visualization.")
            else:
                messagebox.showerror("Error", "Failed to create visualization")

        except Exception as e:
            print(f"Error in analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def _get_data_quality_status(self, stats):
        """Helper method to determine geographic data quality"""
        if not stats:
            return "No data available"
        
        sample_size = stats.get('count', 0)
        if sample_size == 0:
            return "No data points"
        elif sample_size < 10:
            return f"Limited data ({sample_size} points)"
        elif sample_size < 50:
            return f"Moderate data ({sample_size} points)"
        else:
            return f"Good data quality ({sample_size} points)"

    def _get_weather_quality_status(self, weather):
        """Helper method to determine weather data quality"""
        if not weather:
            return "No weather data available"
        
        required_fields = ['temperature', 'description', 'humidity', 'wind_speed']
        available_fields = sum(1 for field in required_fields if field in weather)
        
        if available_fields == 0:
            return "No weather data"
        elif available_fields < len(required_fields):
            return f"Partial weather data ({available_fields}/{len(required_fields)} metrics)"
        else:
            return "Complete weather data"
    
    def view_map(self):
        """Open the interactive map in default web browser"""
        try:
            map_path = os.path.abspath('interactive_map.html')
            if os.path.exists(map_path):
                print(f"Opening map: {map_path}")
                webbrowser.open(f'file:///{map_path}')
            else:
                messagebox.showerror("Error", "Map not found. Please run the analysis first.")
        except Exception as e:
            print(f"Error displaying map: {e}")
            messagebox.showerror("Error", f"Failed to display map: {str(e)}")
    
    def fetch_city_data(self):
        """Fetch data for the specified city"""
        try:
            city = self.city_entry.get()
            radius = float(self.radius_entry.get())
            
            if not city:
                messagebox.showerror("Error", "Please enter a city name")
                return
            
            # Show loading message
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", f"Fetching data for {city}...\nThis may take a few moments...")
            self.window.update()
            
            # Fetch the data
            success = self.analyzer.fetch_city_data(city, radius)
            
            if success:
                # Update available columns for analysis
                self.update_column_menu()
                
                # Update location entry for analysis
                self.location_entry.delete(0, tk.END)
                self.location_entry.insert(0, city)
                
                # Run initial analysis
                self.run_analysis()
                
                messagebox.showinfo("Success", f"Successfully fetched and analyzed data for {city}")
            else:
                messagebox.showerror("Error", f"Failed to fetch data for {city}")
        
        except ValueError as e:
            messagebox.showerror("Error", "Please enter a valid radius (number)")
        except Exception as e:
            self.logger.error(f"Error in fetch_city_data: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        try:
            # Add a flag to track if the application is running
            self.is_running = True
            
            while self.is_running:
                try:
                    # Check if window exists and isn't destroyed
                    if self.window and hasattr(self.window, '_w'):
                        self.window.update()
                    else:
                        self.is_running = False
                        break
                except tk.TclError:
                    # Window was destroyed
                    self.is_running = False
                    self.logger.info("Application interrupted by user")
                    break
                
        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'window') and self.window:
                try:
                    self.window.destroy()
                except:
                    pass
            
            # Cleanup map file if it exists
            if self.map_file and os.path.exists(self.map_file):
                try:
                    os.remove(self.map_file)
                except Exception as e:
                    self.logger.error(f"Failed to cleanup map file: {str(e)}")

    def update_column_menu(self):
        """Update the column menu with available columns from the data"""
        try:
            if self.analyzer and self.analyzer.data is not None:
                # Get available numeric columns
                numeric_columns = self.analyzer.data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Always ensure 'population' is in the list if available
                if 'population' in self.analyzer.data.columns and 'population' not in numeric_columns:
                    numeric_columns.append('population')
                
                # Update the menu with available columns
                if numeric_columns:
                    self.column_menu.configure(values=numeric_columns)
                    # Set to population by default if available, otherwise first numeric column
                    default_column = 'population' if 'population' in numeric_columns else numeric_columns[0]
                    self.column_var.set(default_column)
                else:
                    self.column_menu.configure(values=['population'])
                    self.column_var.set('population')
                    
                print(f"Updated column menu with: {numeric_columns}")  # Debug print
                
        except Exception as e:
            self.logger.error(f"Error updating column menu: {e}")
            # Set default values if update fails
            self.column_menu.configure(values=['population'])
            self.column_var.set('population')

if __name__ == "__main__":
    app = GeoSightGUI()
    app.run() 