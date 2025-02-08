#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class OccupancyMapParser:
    def __init__(self, filename):
        self.filename = filename
        self.width = 794
        self.height = 1224
        self.expected_size = self.width * self.height
        
    def parse_map(self):
        """Parse and visualize the occupancy map"""
        try:
            # Read exact amount of data needed
            with open(self.filename, 'rb') as f:
                map_data = f.read(self.expected_size)
            
            # Convert to numpy array
            map_array = np.frombuffer(map_data, dtype=np.uint8).reshape(self.height, self.width)
            
            # Create visualization
            plt.figure(figsize=(20, 10))
            
            # Plot 1: Raw occupancy values
            plt.subplot(121)
            plt.title('Raw Occupancy Values')
            plt.imshow(map_array, cmap='gray_r')
            plt.colorbar(label='Occupancy Value')
            
            # Plot 2: Thresholded map
            plt.subplot(122)
            plt.title('Thresholded Map')
            
            # Create thresholded map
            thresholded = np.zeros_like(map_array)
            thresholded[map_array == 0] = 255        # Free space (white)
            thresholded[map_array >= 100] = 0        # Definitely occupied (black)
            thresholded[map_array > 0] = 127         # Partially occupied (gray)
            
            plt.imshow(thresholded, cmap='gray')
            
            # Add grid overlay
            grid_spacing = 50  # pixels
            plt.grid(True, color='blue', alpha=0.3, linestyle='--')
            plt.xticks(np.arange(0, self.width, grid_spacing))
            plt.yticks(np.arange(0, self.height, grid_spacing))
            
            # Save visualization
            output_file = os.path.splitext(self.filename)[0] + '_visualization.png'
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            print(f"Visualization saved to: {output_file}")
            
            # Print statistics
            print("\nMap Statistics:")
            print(f"Dimensions: {self.width}x{self.height} pixels")
            total_cells = self.width * self.height
            
            # Count different types of cells
            free_cells = np.sum(map_array == 0)
            occupied_cells = np.sum(map_array >= 100)
            partial_cells = np.sum((map_array > 0) & (map_array < 100))
            
            print(f"\nCell Distribution:")
            print(f"Free space: {free_cells} cells ({free_cells/total_cells*100:.1f}%)")
            print(f"Occupied: {occupied_cells} cells ({occupied_cells/total_cells*100:.1f}%)")
            print(f"Partially occupied: {partial_cells} cells ({partial_cells/total_cells*100:.1f}%)")
            
            # Save thresholded map as binary file
            threshold_file = os.path.splitext(self.filename)[0] + '_thresholded.bin'
            thresholded.tofile(threshold_file)
            print(f"\nThresholded map saved to: {threshold_file}")
            
            return map_array
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
            
    def generate_svg(self, map_array):
        """Generate SVG visualization of the map"""
        try:
            # Threshold the map for SVG
            threshold = 50
            obstacles = map_array >= threshold
            
            # Create SVG header
            svg_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                        f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">',
                        '<g fill="none" stroke="black" stroke-width="1">']
            
            # Add obstacles as rectangles
            for y in range(self.height):
                for x in range(self.width):
                    if obstacles[y, x]:
                        svg_lines.append(f'<rect x="{x}" y="{y}" width="1" height="1"/>')
            
            svg_lines.append('</g></svg>')
            
            # Save SVG file
            svg_file = os.path.splitext(self.filename)[0] + '_map.svg'
            with open(svg_file, 'w') as f:
                f.write('\n'.join(svg_lines))
            print(f"SVG map saved to: {svg_file}")
            
        except Exception as e:
            print(f"Error generating SVG: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python map_parser.py <filename>")
        return
        
    parser = OccupancyMapParser(sys.argv[1])
    map_array = parser.parse_map()
    if map_array is not None:
        parser.generate_svg(map_array)

if __name__ == '__main__':
    main()
