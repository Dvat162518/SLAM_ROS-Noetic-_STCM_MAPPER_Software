# STCM Map Viewer
A Python-based 3D viewer for STCM (Occupancy Grid Map) files with RViz-style visualization and editing capabilities.

## Features
- 3D visualization of occupancy grid maps
- Interactive camera controls
- Drawing tools for map modification:
  - Virtual walls
  - Rectangles
  - Circles
- Multiple export formats (STCM, PNG, SVG, PDF, Binary)
- Grid overlay and viewing options
- Map metadata analysis

## Components

### Map Decoder (Mapper_Decoder_Final.py)
- Extracts and validates STCM file metadata
- Decodes binary occupancy grid data
- Features:
  - Automatic dimension detection
  - Resolution and origin extraction
  - Data integrity validation
  - ASCII map preview generation
  - Occupancy value analysis

### Map Parser (stcm_parser_final.py)
- Processes decoded map data for visualization
- Handles data transformations and exports
- Features:
  - Occupancy grid thresholding
  - Statistical analysis
  - Multi-format export (PNG, SVG)
  - Grid visualization with matplotlib
  - Resolution scaling support

### Map Viewer (STCM_Viewer_Final.py)
- Main application with 3D visualization
- Interactive editing interface
- Features:
  - OpenGL-based 3D rendering
  - Real-time map modifications
  - Multiple view perspectives
  - Drawing tools integration
  - Export functionality

## Dependencies
```bash
- Python 3.x
- PyQt5
- OpenGL
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- svgwrite
```

## File Structure
```
├── src/
│   ├── Mapper_Decoder_Final.py    # Map decoding utilities
│   ├── stcm_parser_final.py       # Core STCM parsing functionality
│   └── STCM_Viewer_Final.py       # Main viewer application
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Dvat162518/SLAM_ROS-Noetic-_STCM_MAPPER_Software.git
cd stcm-map-viewer
```

2. Install required packages:
```bash
pip install PyQt5 numpy matplotlib Pillow svgwrite PyOpenGL
```

## Usage

### Map Decoder Usage
```bash
python src/Mapper_Decoder_Final.py <input_file.stcm>
```
Outputs:
- File metadata analysis
- Occupancy distribution
- ASCII preview of map

### Map Parser Usage
```bash
python src/stcm_parser_final.py <decoded_map.bin>
```
Outputs:
- Statistical analysis
- Visualization files (PNG/SVG)
- Thresholded binary map

### Map Viewer Usage
Run the main viewer application:
```bash
python src/STCM_Viewer_Final.py
```

### Navigation Controls
- Left Mouse: Rotate view
- Right Mouse: Pan view
- Middle Mouse: Adjust height
- Scroll Wheel: Zoom in/out

### Keyboard Shortcuts
- T: Top view
- F: Front view
- L: Left view
- R: Right view

### Drawing Tools
1. Load a map using the 'Load STCM' button
2. Select a drawing tool (Virtual Wall, Rectangle, or Circle)
3. Draw on the map using mouse clicks and drags
4. Save modifications using the 'Save Map' button

## Data Flow
1. Decoder reads raw STCM file and extracts metadata/map data
2. Parser processes the decoded data for visualization
3. Viewer loads processed map for 3D visualization and editing

## License
Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.
