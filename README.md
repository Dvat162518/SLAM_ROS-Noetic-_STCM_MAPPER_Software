# SLAM_ROS-Noetic-_STCM_MAPPER_Software
STCM Map Viewer: A Python-based 3D visualization tool for occupancy grid maps, featuring RViz-style interactive editing. Supports map analysis, virtual wall drawing, and multiple export formats. Built with PyQt5 and OpenGL, it offers intuitive navigation controls and real-time map modifications for robotics applications.

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
git clone https://github.com/yourusername/stcm-map-viewer.git
cd stcm-map-viewer
```

2. Install required packages:
```bash
pip install PyQt5 numpy matplotlib Pillow svgwrite PyOpenGL
```

## Usage

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
