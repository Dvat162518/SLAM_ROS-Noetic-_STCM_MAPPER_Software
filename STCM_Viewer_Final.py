#!/usr/bin/env python3
import os
import sys
import math
import struct
import numpy as np
import svgwrite
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Initialize GLUT before creating QApplication
from OpenGL.GLUT import glutInit
glutInit([])

from OpenGL.GL import *
from OpenGL.GLU import *

class STCMParser:
    @staticmethod
    def find_value(data, keyword):
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

    @staticmethod
    def parse_metadata(filename):
        """Parse STCM file metadata"""
        try:
            with open(filename, 'rb') as f:
                data = f.read()
                
            print("=== STCM File Analysis ===")
            print(f"File size: {len(data)} bytes\n")
            
            # Extract dimensions from text
            width = STCMParser.find_value(data, "dimension_width") or STCMParser.find_value(data, "pixels_width")
            height = STCMParser.find_value(data, "dimension_height") or STCMParser.find_value(data, "pixels_height")
            
            # Extract resolution
            res_x = STCMParser.find_value(data, "resolution_x") or 0.05
            res_y = STCMParser.find_value(data, "resolution_y") or 0.05
            
            # Extract origin
            origin_x = STCMParser.find_value(data, "origin_x") or -23.69
            origin_y = STCMParser.find_value(data, "origin_y") or -45.84
            
            # Find potential start of binary map data
            chunk_size = 1000
            best_start = None
            most_zeros = 0
            
            for i in range(0, len(data) - chunk_size, chunk_size):
                chunk = data[i:i+chunk_size]
                zeros = chunk.count(b'\x00')
                if zeros > most_zeros:
                    most_zeros = zeros
                    best_start = i
            
            # Default to last resort calculation if no dimensions found
            if width is None or height is None:
                # Try to estimate dimensions from total file size
                if best_start is not None:
                    total_map_size = len(data) - best_start
                    # Try to find most square-like dimensions
                    side_length = int(math.sqrt(total_map_size))
                    width = side_length
                    height = total_map_size // side_length
            
            return {
                'width': int(width) if width is not None else 1478,
                'height': int(height) if height is not None else 1295,
                'resolution': float(res_x),
                'origin_x': float(origin_x),
                'origin_y': float(origin_y),
                'map_start': best_start
            }
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            # Return default values if parsing fails
            return {
                'width': 1478,
                'height': 1295,
                'resolution': 0.05,
                'origin_x': -23.69,
                'origin_y': -45.84,
                'map_start': None
            }

class DrawingTool:
    """Base class for drawing tools"""
    def __init__(self, color=255, thickness=1):
        self.color = color
        self.thickness = thickness
        self.points = []

    def add_point(self, x, y):
        """Add a point to the drawing"""
        # Prevent duplicate points
        if not self.points or (x, y) != self.points[-1]:
            self.points.append((x, y))

    def reset(self):
        """Reset the current drawing"""
        self.points = []

    def draw(self, map_data):
        """
        Abstract method to draw on the map data
        Should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement draw method")

class VirtualWallTool(DrawingTool):
    def draw(self, map_data):
        """Draw the virtual wall"""
        if len(self.points) < 2:
            return map_data.copy()
        
        modified_data = map_data.copy()
        
        for x, y in self.points:
            # Ensure we're within map boundaries
            if (0 <= x < map_data.shape[1] and 0 <= y < map_data.shape[0]):
                modified_data[y, x] = self.color
        
        return modified_data

class RectangleTool(DrawingTool):
    def draw(self, map_data):
        """Draw the rectangle"""
        if len(self.points) < 2:
            return map_data.copy()
        
        modified_data = map_data.copy()
        
        x0, y0 = self.points[0]
        x1, y1 = self.points[-1]
        
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                # Ensure we're within map boundaries
                if (0 <= x < map_data.shape[1] and 0 <= y < map_data.shape[0]):
                    modified_data[y, x] = self.color
        
        return modified_data

class CircleTool(DrawingTool):
    def draw(self, map_data):
        """Draw the circle"""
        if len(self.points) < 2:
            return map_data.copy()
        
        modified_data = map_data.copy()
        
        x0, y0 = self.points[0]
        x1, y1 = self.points[-1]
        
        radius = int(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
        
        for y in range(y0 - radius, y0 + radius + 1):
            for x in range(x0 - radius, x0 + radius + 1):
                # Ensure we're within map boundaries
                if (0 <= x < map_data.shape[1] and 0 <= y < map_data.shape[0]):
                    if (x - x0)**2 + (y - y0)**2 <= radius**2:
                        modified_data[y, x] = self.color
        
        return modified_data

class DrawingManager:
    """Manages drawing tools and map modifications"""
    def __init__(self, map_data):
        self.original_map = map_data.copy()
        self.current_map = map_data.copy()
        self.drawing_tools = {
            'wall': VirtualWallTool(color=255, thickness=3),  # Red wall (high occupancy)
            'rectangle': RectangleTool(color=200),  # Darker red for rectangles
            'circle': CircleTool(color=180)  # Even darker for circles
        }
        self.current_tool = 'wall'
    
    def add_point(self, x, y):
        """Add a point to the current drawing tool"""
        self.drawing_tools[self.current_tool].add_point(x, y)
    
    def draw(self):
        """Apply the current drawing tool to the map"""
        # Reset to original map
        self.current_map = self.original_map.copy()
        
        # Apply each tool
        for tool_name, tool in self.drawing_tools.items():
            if tool.points:
                self.current_map = tool.draw(self.current_map)
        
        return self.current_map
    
    def reset_tool(self, tool_name=None):
        """Reset a specific tool or the current tool"""
        if tool_name:
            self.drawing_tools[tool_name].reset()
        else:
            self.drawing_tools[self.current_tool].reset()
    
    def set_current_tool(self, tool_name):
        """Set the current active drawing tool"""
        if tool_name in self.drawing_tools:
            self.current_tool = tool_name

class MapExporter:
    @staticmethod
    def export_stcm(stcm_map, filename):
        """
        Export map data in original STCM format with enhanced metadata
        Includes the modified map data with drawn elements
        """
        try:
            # Get the modified map data from the STCMMap instance
            modified_map = stcm_map.data

            # Ensure data is in the correct format
            map_data = modified_map.astype(np.uint8)

            # Write raw bytes
            with open(filename, 'wb') as f:
                # Add comprehensive metadata header
                metadata_header = (
                    f"dimension_width {stcm_map.width}\n"
                    f"dimension_height {stcm_map.height}\n"
                    f"resolution_x {stcm_map.resolution}\n"
                    f"resolution_y {stcm_map.resolution}\n"
                    f"origin_x {stcm_map.origin_x}\n"
                    f"origin_y {stcm_map.origin_y}\n"
                    f"total_map_points {map_data.size}\n"
                    f"unique_values {len(np.unique(map_data))}\n"
                    "MAP_DATA_START\n".encode()
                )
                f.write(metadata_header)

                # Write raw map data
                f.write(map_data.tobytes())

            print(f"STCM map saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving STCM map: {e}")
            return False

    @staticmethod
    def analyze_map_modifications(original_map, modified_map):
        """
        Analyze the differences between original and modified maps
        """
        # Find where modifications occurred
        diff_mask = original_map != modified_map
        
        # Calculate modification statistics
        modifications = {
            'total_changes': np.sum(diff_mask),
            'change_percentage': np.sum(diff_mask) / diff_mask.size * 100,
            'added_walls': np.sum((modified_map >= 100) & (original_map < 100)),
            'unique_new_values': np.setdiff1d(
                np.unique(modified_map), 
                np.unique(original_map)
            )
        }
        
        # Detailed change report
        print("\n=== Map Modification Analysis ===")
        for key, value in modifications.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return modifications

    @staticmethod
    def export_png(stcm_map, filename):
        """
        Export map as a PNG image with color-coded occupancy
        """
        try:
            # Create color-coded image
            img = np.zeros((stcm_map.height, stcm_map.width, 3), dtype=np.uint8)
            
            # Color mapping
            free_space = stcm_map.data == 0
            occupied = stcm_map.data >= 100
            partially_occupied = (stcm_map.data > 0) & (stcm_map.data < 100)
            
            # White for free space
            img[free_space] = [255, 255, 255]
            # Black for occupied
            img[occupied] = [0, 0, 0]
            # Gray for partially occupied
            img[partially_occupied] = [128, 128, 128]
            
            # Save using Pillow
            Image.fromarray(img).save(filename)
            
            print(f"PNG map saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving PNG map: {e}")
            return False

    @staticmethod
    def export_svg(stcm_map, filename):
        """
        Export map as an SVG with detailed occupancy visualization
        """
        try:
            # Create SVG drawing
            dwg = svgwrite.Drawing(filename, profile='tiny', size=(
                f'{stcm_map.width}px', 
                f'{stcm_map.height}px'
            ))
            
            # Add background
            dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
            
            # Draw occupied and partially occupied cells
            for y in range(stcm_map.height):
                for x in range(stcm_map.width):
                    value = stcm_map.data[y, x]
                    
                    if value >= 100:
                        # Fully occupied - black
                        dwg.add(dwg.rect(
                            insert=(x, y), 
                            size=(1, 1), 
                            fill='black'
                        ))
                    elif value > 0:
                        # Partially occupied - gray with opacity
                        opacity = value / 100
                        dwg.add(dwg.rect(
                            insert=(x, y), 
                            size=(1, 1), 
                            fill='gray',
                            opacity=opacity
                        ))
            
            # Save SVG
            dwg.save()
            
            print(f"SVG map saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving SVG map: {e}")
            return False

    @staticmethod
    def export_bin(stcm_map, filename):
        """
        Export raw binary map data
        """
        try:
            # Write raw bytes of map data
            with open(filename, 'wb') as f:
                f.write(stcm_map.data.tobytes())
            
            print(f"Binary map saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving binary map: {e}")
            return False

    @staticmethod
    def export_pdf(stcm_map, filename):
        """
        Export map as a PDF with detailed visualization
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 10))
            plt.imshow(stcm_map.data, cmap='gray', interpolation='nearest')
            plt.title(f'Map: {stcm_map.width}x{stcm_map.height}')
            plt.colorbar(label='Occupancy')
            
            # Save as PDF
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            print(f"PDF map saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving PDF map: {e}")
            return False

class STCMMap:
    def __init__(self):
        # Default values, will be updated dynamically
        self.width = 1478  # Default map dimensions
        self.height = 1295
        self.resolution = 0.05  # 5cm per pixel
        self.origin_x = -23.69
        self.origin_y = -45.84
        self.data = None
        self.texture_id = None
        self.filename = None

    def decode_stcm(self, raw_data):
        """Decode STCM format data with robust handling and drawn elements"""
        try:
            # Detailed diagnostic information
            print(f"Raw data length: {len(raw_data)} bytes")

            # First try to parse metadata to get base dimensions
            metadata = STCMParser.parse_metadata(self.filename)

            # List of possible dimension variations to try
            dim_variations = [
                (metadata['width'], metadata['height']),
                (1394, 719),  # From the current file
                (1478, 1295),  # Original hardcoded dimensions
            ]

            # Try different dimension combinations
            for width, height in dim_variations:
                try:
                    expected_size = width * height

                    # If data is larger, truncate
                    if len(raw_data) > expected_size:
                        map_data = raw_data[:expected_size]
                    # If data is smaller, pad with zeros
                    elif len(raw_data) < expected_size:
                        map_data = raw_data + b'\x00' * (expected_size - len(raw_data))
                    else:
                        map_data = raw_data

                    # Try to reshape the data
                    self.data = np.frombuffer(map_data, dtype=np.uint8).reshape(height, width)

                    # Update map properties
                    self.width = width
                    self.height = height

                    # Print map details
                    print(f"Map loaded: {self.width}x{self.height} pixels")
                    print(f"Resolution: {self.resolution} meters/pixel")
                    print(f"Origin: ({self.origin_x}, {self.origin_y})")

                    # Additional data validation
                    unique_values = np.unique(self.data)
                    print("Unique pixel values:", unique_values)
                    print("Occupancy distribution:")
                    for val in unique_values:
                        count = np.sum(self.data == val)
                        percentage = count / self.data.size * 100
                        print(f"Value {val}: {count} pixels ({percentage:.2f}%)")

                    # Read the drawn elements from the file (if any)
                    self.drawn_elements = self.read_drawn_elements(raw_data)

                    return True

                except Exception as e:
                    print(f"Failed to load with dimensions {width}x{height}: {e}")
                    continue

            # If no dimensions work
            raise ValueError("Could not find suitable map dimensions")

        except Exception as e:
            print(f"Error decoding STCM: {e}")
            import traceback
            traceback.print_exc()
            return False

    def read_drawn_elements(self, raw_data):
        """
        Read the drawn elements from the STCM file
        """
        try:
            # Find the start of the drawn elements data
            start_index = raw_data.find(b"DRAWN_ELEMENTS_START")
            if start_index == -1:
                # No drawn elements found
                return None

            start_index += len(b"DRAWN_ELEMENTS_START\n")

            # Find the end of the drawn elements data
            end_index = raw_data.find(b"DRAWN_ELEMENTS_END")
            if end_index == -1:
                # Malformed file, unable to find end of drawn elements
                return None

            # Extract the drawn elements data
            drawn_elements_data = raw_data[start_index:end_index]

            # Parse the drawn elements data
            drawn_elements = []
            for line in drawn_elements_data.decode().strip().split("\n"):
                tool_type, *points = line.split(",")
                points = [(int(x), int(y)) for x, y in [point.strip().split(" ") for point in points]]
                drawn_elements.append((tool_type, points))

            return drawn_elements

        except Exception as e:
            print(f"Error reading drawn elements: {e}")
            return None
    
    def load_map_data(self, filename):
        """Load map data from file"""
        try:
            # Store filename for metadata parsing
            self.filename = filename
            
            # Read entire file
            with open(filename, 'rb') as f:
                raw_data = f.read()
            
            # Decode the map data
            return self.decode_stcm(raw_data)
        except Exception as e:
            print(f"Error loading map data: {e}")
            return False

    def create_texture(self):
        """Create OpenGL texture from map data"""
        if self.data is None:
            return None

        if self.texture_id is None:
            self.texture_id = glGenTextures(1)

        # Create texture image
        texture_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Convert map data to RGBA
        free_space = self.data == 0
        occupied = self.data >= 100
        partially_occupied = (self.data > 0) & (self.data < 100)

        # White for free space
        texture_data[free_space] = [0, 0, 0, 255]
        # Black for occupied
        texture_data[occupied] = [255, 255, 255, 255]
        # Gray for partially occupied
        texture_data[partially_occupied] = [127, 127, 127, 255]

        # Bind and set texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        return self.texture_id

class MapGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create OpenGL format
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        
        self.stcm = STCMMap()
        self.camera_distance = 10.0
        self.camera_x = 5.0
        self.camera_y = -5.0
        self.camera_z = 8.0
        self.rotation_x = 45.0
        self.rotation_y = 45.0
        
        # Added missing attributes
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        
        self.show_grid = True
        self.show_walls = True
        self.show_floor = True
        
        self.last_pos = None
        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            
            # Set up lighting for better visibility
            glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1])
            
            glClearColor(0.1, 0.1, 0.1, 1.0)
            print("OpenGL initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")

    def resizeGL(self, width, height):
        try:
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, width/float(height), 0.1, 1000.0)
        except Exception as e:
            print(f"Error in resizeGL: {e}")

        # Drawing management
        self.drawing_mode = False
        self.drawing_manager = None
        
        # Add drawing-related attributes
        self.current_tool = 'wall'
    
    def draw_map(self):
        if self.stcm.data is None:
            return

        try:
            # Create or update texture
            texture_id = self.stcm.create_texture()
            if texture_id is None:
                return
                
            # Calculate physical dimensions
            width_meters = self.stcm.width * self.stcm.resolution
            height_meters = self.stcm.height * self.stcm.resolution
            
            # Enable texturing
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set blending for semi-transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Draw textured quad
            glColor4f(1.0, 1.0, 1.0, 1.0)  # White, full opacity
            glBegin(GL_QUADS)
            
            # Draw map centered at origin
            half_width = width_meters / 2
            half_height = height_meters / 2
            
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-half_width, -half_height, 0)
            
            glTexCoord2f(1.0, 0.0)
            glVertex3f(half_width, -half_height, 0)
            
            glTexCoord2f(1.0, 1.0)
            glVertex3f(half_width, half_height, 0)
            
            glTexCoord2f(0.0, 1.0)
            glVertex3f(-half_width, half_height, 0)
            
            glEnd()
            
            # Disable texturing
            glDisable(GL_TEXTURE_2D)
            
        except Exception as e:
            print(f"Error drawing map: {e}")

    def draw_grid(self):
        if not self.show_grid:
            return
            
        try:
            glBegin(GL_LINES)
            glColor4f(0.5, 0.5, 0.5, 0.3)
            
            size = 20
            step = 1.0
            
            for i in range(-size, size+1):
                glVertex3f(i*step, -size*step, 0)
                glVertex3f(i*step, size*step, 0)
                glVertex3f(-size*step, i*step, 0)
                glVertex3f(size*step, i*step, 0)
            
            glEnd()
        except Exception as e:
            print(f"Error drawing grid: {e}")

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            gluLookAt(self.camera_x, self.camera_y, self.camera_z,
                     self.target_x, self.target_y, self.target_z,
                     0, 0, 1)

            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 0, 1)

            if self.show_grid:
                self.draw_grid()

            # Draw the map
            self.draw_map()

            # Draw the drawn elements on top
            if self.drawing_mode and self.drawing_manager:
                self.draw_drawn_elements()

        except Exception as e:
            print(f"Error in paintGL: {e}")

    def draw_drawn_elements(self):
        """Draw the elements added using the drawing tools"""
        if self.drawing_mode and self.drawing_manager:
            # Get the drawing tools and their points
            tools = self.drawing_manager.drawing_tools
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Draw each tool's elements
            for tool_name, tool in tools.items():
                if tool.points:
                    self.draw_tool_elements(tool)
            
            # Redraw the map with the updated data
            self.stcm.data = self.drawing_manager.current_map
            self.draw_map()
            
            glDisable(GL_BLEND)

    def draw_tool_elements(self, tool):
        """Draw the elements of a specific drawing tool"""
        # Calculate physical dimensions
        width_meters = self.stcm.width * self.stcm.resolution
        height_meters = self.stcm.height * self.stcm.resolution
        
        # Draw the tool's elements
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        
        # Draw map centered at origin
        half_width = width_meters / 2
        half_height = height_meters / 2
        
        for i in range(len(tool.points) - 1):
            x1, y1 = tool.points[i]
            x2, y2 = tool.points[i + 1]
            
            # Translate the map coordinates to 3D coordinates
            px1 = x1 * self.stcm.resolution - half_width
            py1 = y1 * self.stcm.resolution - half_height
            px2 = x2 * self.stcm.resolution - half_width
            py2 = y2 * self.stcm.resolution - half_height
            
            # Set the color based on the tool type
            if tool.color == 255:  # Red for walls
                glColor4f(1.0, 0.0, 0.0, 0.8)
            elif tool.color == 200:  # Darker red for rectangles
                glColor4f(0.8, 0.0, 0.0, 0.6)
            else:  # Even darker for circles
                glColor4f(0.7, 0.0, 0.0, 0.5)
            
            # Draw the line segment
            glVertex3f(px1, py1, 0)
            glVertex3f(px2, py2, 0)
        
        glEnd()

    def load_map(self, filename):
        """Add method to load map from file"""
        if self.stcm.load_map_data(filename):
            # Redraw the map with any previously drawn elements
            if self.drawing_manager:
                self.stcm.data = self.drawing_manager.current_map
            else:
                self.drawing_manager = DrawingManager(self.stcm.data)
            
            self.update()
            return True
        else:
            return False
        
    def wheelEvent(self, event):
        """Enhanced zoom with minimum and maximum limits"""
        delta = event.angleDelta().y()
        zoom_speed = 0.001
        
        # Update camera distance with limits
        self.camera_distance = max(5.0, min(100.0, 
        self.camera_distance * (0.95 if delta > 0 else 1.05)))
        
        # Update camera position
        self.update_camera()
        self.update()
    
    def screen_to_map_coords(self, screen_x, screen_y):
        """Convert screen coordinates to map coordinates"""
        # This is a simplified conversion and might need refinement
        width_meters = self.stcm.width * self.stcm.resolution
        height_meters = self.stcm.height * self.stcm.resolution
        
        # Calculate map coordinates based on current view
        x = ((screen_x / self.width()) * width_meters + 
             (self.stcm.origin_x - width_meters/2))
        y = ((self.height() - screen_y) / self.height() * height_meters + 
             (self.stcm.origin_y - height_meters/2))
        
        # Convert to pixel coordinates
        map_x = int((x - self.stcm.origin_x) / self.stcm.resolution)
        map_y = int((y - self.stcm.origin_y) / self.stcm.resolution)
        
        return map_x, map_y
    
    def start_drawing(self, tool_name='wall'):
        """Enter drawing mode"""
        if self.stcm.data is not None:
            self.drawing_mode = True
            self.current_tool = tool_name  # Add this line
            self.drawing_manager = DrawingManager(self.stcm.data)
            self.drawing_manager.set_current_tool(tool_name)
            self.setCursor(Qt.CrossCursor)
    
    def end_drawing(self):
        """Exit drawing mode"""
        self.drawing_mode = False
        self.drawing_manager = None
        self.setCursor(Qt.ArrowCursor)
    
    def reset_current_tool(self):
        """Reset the current drawing tool"""
        if self.drawing_manager:
            self.drawing_manager.reset_tool()
            self.update()
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for view angle and drawing"""
        # Top view shortcut
        if event.key() == Qt.Key_T:
            self.rotation_x = 90.0  # Looking directly down
            self.rotation_y = 0.0
            self.update_camera()
            self.update()
        
        # Side view shortcuts
        elif event.key() == Qt.Key_L:  # Left view
            self.rotation_x = 45.0
            self.rotation_y = 90.0
            self.update_camera()
            self.update()
        
        elif event.key() == Qt.Key_R:  # Right view
            self.rotation_x = 45.0
            self.rotation_y = -90.0
            self.update_camera()
            self.update()
        
        # Front view
        elif event.key() == Qt.Key_F:
            self.rotation_x = 45.0
            self.rotation_y = 0.0
            self.update_camera()
            self.update()

    def mousePressEvent(self, event):
        """Enhanced mouse event to support drawing"""
        # If drawing mode is active, start drawing
        if self.drawing_mode and self.drawing_manager:
            # Convert screen coordinates to map coordinates
            x, y = self.screen_to_map_coords(event.x(), event.y())
            
            # Reset points for rectangle and circle tools
            if self.current_tool in ['rectangle', 'circle']:
                self.drawing_manager.reset_tool(self.current_tool)
            
            # Add the first point
            self.drawing_manager.add_point(int(x), int(y))
            print(f"Drawing started at: {x}, {y}")  # Debug print
            self.update()
            return
        
        # If not in drawing mode, handle camera control
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()
        elif event.button() == Qt.RightButton:
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Enhanced mouse move to support drawing and camera control"""
        # If drawing mode is active, continue drawing
        if self.drawing_mode and self.drawing_manager:
            # Convert screen coordinates to map coordinates
            x, y = self.screen_to_map_coords(event.x(), event.y())
            
            # Add point for continuous drawing or update last point for shapes
            if self.current_tool == 'wall':
                self.drawing_manager.add_point(int(x), int(y))
                print(f"Drawing wall point: {x}, {y}")  # Debug print
            elif self.current_tool in ['rectangle', 'circle']:
                # For shapes, just update the last point
                if len(self.drawing_manager.drawing_tools[self.current_tool].points) > 0:
                    self.drawing_manager.drawing_tools[self.current_tool].points[-1] = (int(x), int(y))
                    print(f"Updating shape point: {x}, {y}")  # Debug print
            
            self.update()
            return
    
    # Rest of the method remains the same
        
        # Camera control when not in drawing mode
        if not self.last_pos:
            return
        
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        # Left mouse button: rotate view
        if event.buttons() & Qt.LeftButton:
            self.rotation_x = max(-89.0, min(89.0, self.rotation_x + dy * 0.5))
            self.rotation_y = (self.rotation_y + dx * 0.5) % 360.0
        
        # Right mouse button: pan view
        elif event.buttons() & Qt.RightButton:
            pan_speed = 0.05
            self.target_x += dx * pan_speed
            self.target_y += dy * pan_speed
        
        # Update camera and redraw
        self.update_camera()
        self.last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release in drawing mode or camera control"""
        if self.drawing_mode and self.drawing_manager:
            # Convert screen coordinates to map coordinates
            x, y = self.screen_to_map_coords(event.x(), event.y())
            
            # For wall tool, add the final point
            if self.current_tool == 'wall':
                self.drawing_manager.add_point(int(x), int(y))
            
            # Update map data with the drawing
            self.stcm.data = self.drawing_manager.draw()
            self.update()
        
        # Reset last position for camera controls
        self.last_pos = None
    
    def update_camera(self):
        """Update camera position based on parameters"""
        # Convert spherical coordinates to Cartesian
        theta = np.radians(self.rotation_y)
        phi = np.radians(self.rotation_x)
        
        self.camera_x = self.target_x + self.camera_distance * np.cos(theta) * np.cos(phi)
        self.camera_y = self.target_y + self.camera_distance * np.sin(theta) * np.cos(phi)
        self.camera_z = self.camera_distance * np.sin(phi)

    def set_show_grid(self, show):
        self.show_grid = show
        self.update()

    def set_show_walls(self, show):
        self.show_walls = show
        self.update()

    def set_show_floor(self, show):
        self.show_floor = show
        self.update()

class STCMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing STCM Viewer")
        self.initUI()

    def initUI(self):
        """Initialize the user interface"""
        try:
            self.setWindowTitle('STCM Map Viewer (RViz Style)')
            self.setGeometry(100, 100, 1200, 800)

            # Create main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QHBoxLayout(main_widget)

            # Create control panel
            control_panel = QWidget()
            control_panel.setMaximumWidth(300)
            control_layout = QVBoxLayout(control_panel)

             # Add Save Map button
            save_btn = QPushButton('Save Map')
            save_btn.clicked.connect(self.save_map)
            control_layout.addWidget(save_btn)
            
            # Add Load STCM button
            load_btn = QPushButton('Load STCM')
            load_btn.clicked.connect(self.load_stcm)
            control_layout.addWidget(load_btn)
        
            # Add display options
            options_group = QGroupBox('Display Options')
            options_layout = QVBoxLayout()
            
            # Add drawing tools group
            drawing_group = QGroupBox('Drawing Tools')
            drawing_layout = QVBoxLayout()

            # Drawing mode buttons
            drawing_buttons = [
                ('Virtual Wall', 'wall'),
                ('Rectangle', 'rectangle'),
                ('Circle', 'circle')
            ]

            self.drawing_buttons = {}
            for label, tool_name in drawing_buttons:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked, name=tool_name: self.activate_drawing_tool(name))
                drawing_layout.addWidget(btn)
                self.drawing_buttons[tool_name] = btn

            # Reset drawing tool button
            reset_btn = QPushButton('Reset Current Tool')
            reset_btn.clicked.connect(self.reset_drawing_tool)
            drawing_layout.addWidget(reset_btn)

            # End drawing mode button
            end_drawing_btn = QPushButton('End Drawing')
            end_drawing_btn.clicked.connect(self.end_drawing_mode)
            drawing_layout.addWidget(end_drawing_btn)

            drawing_group.setLayout(drawing_layout)
            control_layout.addWidget(drawing_group)
            
            # Create checkboxes with connections
            self.show_grid_cb = QCheckBox('Show Grid')
            self.show_grid_cb.setChecked(True)
            self.show_grid_cb.stateChanged.connect(self.toggle_grid)
            
            self.show_walls_cb = QCheckBox('Show Walls')
            self.show_walls_cb.setChecked(True)
            self.show_walls_cb.stateChanged.connect(self.toggle_walls)
            
            self.show_floor_cb = QCheckBox('Show Floor')
            self.show_floor_cb.setChecked(True)
            self.show_floor_cb.stateChanged.connect(self.toggle_floor)
            
            for cb in [self.show_grid_cb, self.show_walls_cb, self.show_floor_cb]:
                options_layout.addWidget(cb)
            
            options_group.setLayout(options_layout)
            control_layout.addWidget(options_group)

            # Add navigation help
            help_group = QGroupBox('Navigation Controls')
            help_layout = QVBoxLayout()
            help_text = QLabel(
                "Left Mouse: Rotate\n"
                "Right Mouse: Pan\n"
                "Middle Mouse: Height\n"
                "Scroll Wheel: Zoom\n\n"
                "Keyboard Shortcuts:\n"
                "T: Top View\n"
                "F: Front View\n"
                "L: Left View\n"
                "R: Right View"
            )
            help_layout.addWidget(help_text)
            help_group.setLayout(help_layout)
            control_layout.addWidget(help_group)

            control_layout.addStretch()
            layout.addWidget(control_panel)

            # Create OpenGL widget
            self.gl_widget = MapGLWidget()
            layout.addWidget(self.gl_widget)

            self.statusBar().showMessage('Ready')
            print("UI initialization completed")
        except Exception as e:
            print(f"Error initializing UI: {e}")

    def activate_drawing_tool(self, tool_name):
        """Activate a specific drawing tool"""
        # Ensure a map is loaded
        if self.gl_widget.stcm.data is None:
            QMessageBox.warning(self, "Drawing Error", "Load a map first!")
            return

        # Start drawing mode with selected tool
        self.gl_widget.start_drawing(tool_name)

        # Update button states
        for name, btn in self.drawing_buttons.items():
            btn.setChecked(name == tool_name)

    def reset_drawing_tool(self):
        """Reset the current drawing tool"""
        if self.gl_widget.drawing_manager:
            self.gl_widget.reset_current_tool()

    def end_drawing_mode(self):
        """End drawing mode"""
        self.gl_widget.end_drawing()

        # Reset button states
        for btn in self.drawing_buttons.values():
            btn.setChecked(False)
    
    def save_map(self):
        """Open save dialog with multiple format options and modification analysis"""
        # Check if a map is loaded
        if self.gl_widget.stcm.data is None:
            QMessageBox.warning(self, "Save Error", "No map loaded to save.")
            return

        # Check for modifications
        has_modifications = False
        if self.gl_widget.drawing_manager:
            # Analyze differences
            original_map = self.gl_widget.drawing_manager.original_map
            current_map = self.gl_widget.stcm.data
            
            # Compare maps
            has_modifications = not np.array_equal(original_map, current_map)
        
        # Prepare confirmation dialog if modifications exist
        if has_modifications:
            reply = QMessageBox.question(
                self, 
                'Save Modifications', 
                'The map has been modified. Do you want to save these changes?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.No:
                # Revert to original map
                self.gl_widget.stcm.data = original_map
        
        # Prepare file dialog
        filename, file_type = QFileDialog.getSaveFileName(
            self, 
            'Save Map', 
            os.path.expanduser('~'), 
            "STCM Files (*.stcm);;PNG Images (*.png);;SVG Files (*.svg);;"
            "Binary Files (*.bin);;PDF Files (*.pdf)"
        )

        if filename:
            try:
                # If modifications exist, analyze them before saving
                if has_modifications:
                    modifications = MapExporter.analyze_map_modifications(
                        original_map, 
                        self.gl_widget.stcm.data
                    )
                
                # Determine save method based on selected file type
                if file_type == "STCM Files (*.stcm)":
                    success = MapExporter.export_stcm(self.gl_widget.stcm, filename)
                elif file_type == "PNG Images (*.png)":
                    success = MapExporter.export_png(self.gl_widget.stcm, filename)
                elif file_type == "SVG Files (*.svg)":
                    success = MapExporter.export_svg(self.gl_widget.stcm, filename)
                elif file_type == "Binary Files (*.bin)":
                    success = MapExporter.export_bin(self.gl_widget.stcm, filename)
                elif file_type == "PDF Files (*.pdf)":
                    success = MapExporter.export_pdf(self.gl_widget.stcm, filename)
                else:
                    success = False
                    QMessageBox.warning(self, "Save Error", "Unsupported file type.")

                # Show success or error message
                if success:
                    # Create detailed save report
                    report_msg = "Map saved successfully."
                    if has_modifications:
                        report_msg += "\n\nModification Details:\n"
                        for key, value in modifications.items():
                            report_msg += f"- {key.replace('_', ' ').title()}: {value}\n"
                    
                    QMessageBox.information(
                        self, 
                        "Save Successful", 
                        report_msg
                    )
                else:
                    QMessageBox.warning(
                        self, 
                        "Save Error", 
                        "Failed to save the map."
                    )

            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Save Error", 
                    f"An error occurred while saving: {str(e)}"
                )
    
    def load_stcm(self):
        print("Attempting to load STCM")
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open STCM Map',
            os.path.expanduser('~'),
            'All Files (*);;STCM Files (*.stcm);;Binary Files (*.bin)'
        )
        
        if filename:
            try:
                # Print file details for debugging
                print(f"Selected file: {filename}")
                print(f"File exists: {os.path.exists(filename)}")
                print(f"File size: {os.path.getsize(filename)} bytes")
                
                # Attempt to load the map
                if self.gl_widget.load_map(filename):
                    # Update the viewer's camera and view to match map dimensions
                    gl_widget = self.gl_widget
                    stcm = gl_widget.stcm
                    
                    # Calculate map physical dimensions
                    width_meters = stcm.width * stcm.resolution
                    height_meters = stcm.height * stcm.resolution
                    
                    # Adjust camera parameters based on map size
                    gl_widget.camera_distance = max(width_meters, height_meters) * 2
                    gl_widget.target_x = stcm.origin_x + width_meters / 2
                    gl_widget.target_y = stcm.origin_y + height_meters / 2
                    gl_widget.target_z = 0
                    
                    # Update camera position
                    gl_widget.update_camera()
                    
                    # Trigger a redraw
                    gl_widget.update()
                    
                    # Update status bar
                    self.statusBar().showMessage(f'Map loaded from {filename}')
                    print(f"Map loaded successfully from {filename}")
                else:
                    raise Exception("Failed to load map data")
            except Exception as e:
                error_msg = f'Error loading map: {str(e)}'
                print(error_msg)
                self.statusBar().showMessage(error_msg)
                
                # Show more detailed error dialog
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setText("Failed to Load Map")
                error_dialog.setInformativeText(str(e))
                error_dialog.setWindowTitle("Map Loading Error")
                error_dialog.exec_()

    def toggle_grid(self, state):
        self.gl_widget.set_show_grid(state == Qt.Checked)

    def toggle_walls(self, state):
        self.gl_widget.set_show_walls(state == Qt.Checked)

    def toggle_floor(self, state):
        self.gl_widget.set_show_floor(state == Qt.Checked)

def main():
    try:
        # Enable high DPI scaling
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        app = QApplication(sys.argv)
        
        # Set OpenGL format for the entire application
        fmt = QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)
        
        viewer = STCMViewer()
        viewer.show()
        print("Application started")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main()