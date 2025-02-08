#!/usr/bin/env python3
import os
import sys
import struct
import numpy as np

class STCMParser:
    def __init__(self, filename):
        self.filename = filename
        
    def find_value(self, data, keyword):
        """Find value after keyword in data"""
        try:
            pos = data.find(keyword.encode())
            if pos < 0:
                return None
                
            # Find the next number after the keyword
            start = pos + len(keyword)
            # Skip until we find a digit or minus sign
            while start < len(data) and not (data[start:start+1].isdigit() or data[start:start+1] == b'-'):
                start += 1
                
            if start >= len(data):
                return None
                
            # Read until we find a non-number character
            end = start
            while end < len(data) and (data[end:end+1].isdigit() or data[end:end+1] in [b'.', b'-']):
                end += 1
                
            value_str = data[start:end].decode()
            try:
                return float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                return None
        except:
            return None

    def parse_file(self):
        """Parse STCM file with better text handling"""
        try:
            with open(self.filename, 'rb') as f:
                data = f.read()
                
            print("=== STCM File Analysis ===")
            print(f"File size: {len(data)} bytes\n")
            
            # Extract dimensions from text
            width = self.find_value(data, "dimension_width")
            height = self.find_value(data, "dimension_height")
            
            # Extract resolution
            res_x = self.find_value(data, "resolution_x")
            res_y = self.find_value(data, "resolution_y")
            
            # Extract origin
            origin_x = self.find_value(data, "origin_x")
            origin_y = self.find_value(data, "origin_y")
            
            print("Found Metadata:")
            print(f"Dimensions: {width}x{height} pixels")
            print(f"Resolution: {res_x}x{res_y} meters/pixel")
            print(f"Origin: ({origin_x}, {origin_y}) meters")
            
            # Look for the start of binary map data
            # Find a section of consistent binary data
            chunk_size = 1000
            best_start = None
            most_zeros = 0
            
            for i in range(0, len(data) - chunk_size, chunk_size):
                chunk = data[i:i+chunk_size]
                zeros = chunk.count(b'\x00')
                if zeros > most_zeros:
                    most_zeros = zeros
                    best_start = i
                    
            if best_start is not None:
                print(f"\nFound potential map data at offset {best_start}")
                
                # Analyze sample of data at this position
                sample = data[best_start:best_start+chunk_size]
                print("\nData sample analysis:")
                value_counts = {}
                for b in sample:
                    value_counts[b] = value_counts.get(b, 0) + 1
                    
                print("Value distribution in sample:")
                for value, count in sorted(value_counts.items()):
                    if count > chunk_size/100:  # Show only significant values
                        print(f"Value {value:3d}: {count:4d} occurrences ({count/chunk_size*100:5.1f}%)")
                        
            # Try to extract a sample of the map
            if width and height and best_start:
                try:
                    # Extract a small section of the map
                    sample_width = min(100, width)
                    sample_height = min(100, height)
                    map_sample = np.frombuffer(
                        data[best_start:best_start + (sample_width * sample_height)],
                        dtype=np.uint8
                    ).reshape(sample_height, sample_width)
                    
                    print("\nMap sample statistics:")
                    unique_values = np.unique(map_sample)
                    print("Unique values found:", sorted(unique_values))
                    
                    # Save a small ASCII preview
                    print("\nMap preview (100x100 or smaller):")
                    for y in range(min(20, sample_height)):
                        line = ""
                        for x in range(min(50, sample_width)):
                            value = map_sample[y, x]
                            if value == 0:
                                line += "."  # Free space
                            elif value == 100:
                                line += "#"  # Wall/obstacle
                            else:
                                line += " "  # Unknown
                        print(line)
                        
                except Exception as e:
                    print(f"Error creating map preview: {e}")
            
        except Exception as e:
            print(f"Error processing file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python stcm_parser.py <filename>")
        return
        
    parser = STCMParser(sys.argv[1])
    parser.parse_file()

if __name__ == '__main__':
    main()