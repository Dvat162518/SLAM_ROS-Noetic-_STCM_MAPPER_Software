# STCM Map Processing Flow

This document demonstrates the step-by-step processing of STCM map files through our system components.

## 1. Map Decoder (Mapper_Decoder_Final.py)
Input: Raw STCM file
```
$ python Mapper_Decoder_Final.py example_map.stcm

=== STCM File Analysis ===
File size: 1,562,880 bytes

Found Metadata:
Dimensions: 1394x719 pixels
Resolution: 0.05x0.05 meters/pixel
Origin: (-23.69, -45.84) meters

Found potential map data at offset 1024

Data sample analysis:
Value   0: 1823 occurrences (91.2%)  # Free space
Value 100:  156 occurrences (7.8%)   # Walls/obstacles
Value  50:   21 occurrences (1.0%)   # Partial occupancy

Map preview (20x50):
.....###..........
..#####...........
....###...........
...................
........##........
```
![Screenshot from 2025-02-08 17-29-02](https://github.com/user-attachments/assets/8636edb3-6232-4667-a30d-a1fbcaf9165d)


## 2. Map Parser (stcm_parser_final.py)
Input: Decoded map data
```
$ python3 stcm_parser_final.py map1.stcm
Visualization saved to: map1_visualization.png

Map Statistics:
Dimensions: 1478x1295 pixels

Cell Distribution:
Free space: 1452773 cells (75.9%)
Occupied: 307654 cells (16.1%)
Partially occupied: 153583 cells (8.0%)

Thresholded map saved to: map1_thresholded.bin
SVG map saved to: map1_map.svg
```

Generated outputs:
- `map_visualization.png`: Raw occupancy values visualization
- `map_thresholded.bin`: Binary representation of occupied/free space
- `map.svg`: Vector graphics representation

![Screenshot from 2025-02-08 17-33-37](https://github.com/user-attachments/assets/244960c1-fd19-46e0-aa34-a637bb8efb94)
![map1_visualization](https://github.com/user-attachments/assets/8aab869c-5dcf-4628-899c-907007830b5d)

## 3. Map Viewer (STCM_Viewer_Final.py)
Interactive 3D visualization with editing capabilities

```
$ python STCM_Viewer_Final.py

=== Loading Map ===
- Initializing OpenGL context
- Creating texture from map data
- Setting up camera position
- Enabling navigation controls

Available Tools:
- Virtual Wall Drawing
- Rectangle Placement
- Circle Placement

View Controls:
- Left Mouse: Rotate
- Right Mouse: Pan
- Scroll: Zoom
- T: Top View
- F: Front View
```
![Screenshot from 2025-02-08 17-33-37](https://github.com/user-attachments/assets/2eaec09f-4260-47b2-b5e8-37185021f1f1)
![Screenshot from 2025-02-08 18-14-10](https://github.com/user-attachments/assets/1c1146f5-a55b-4deb-9245-9d62837a39c6)
![Screenshot from 2025-02-08 18-14-33](https://github.com/user-attachments/assets/b742b266-8bc2-4039-8777-a317530103b7)
![Screenshot from 2025-02-08 18-14-56](https://github.com/user-attachments/assets/6ecb9199-5be2-48bd-a43c-1f2dab3b3865)
![Screenshot from 2025-02-08 18-15-09](https://github.com/user-attachments/assets/aad47156-9d05-4746-8219-55ec2741b73f)
![Screenshot from 2025-02-08 18-15-13](https://github.com/user-attachments/assets/2693f288-f417-4ca9-adf1-36f9914fa273)


### Output Features:

1. Real-time 3D Visualization
   - Occupancy grid displayed as height map
   - Color-coded occupancy values
   - Interactive camera controls

2. Editing Capabilities
   - Draw virtual walls
   - Add rectangular regions
   - Place circular markers

3. Export Options
   - Modified STCM file
   - PNG/SVG exports
   - Binary format

### Data Flow Diagram:
```
Raw STCM File
      ↓
[Map Decoder]
 - Extract metadata
 - Decode binary data
      ↓
[Map Parser]
 - Process occupancy data
 - Generate visualizations
      ↓
[Map Viewer]
 - 3D visualization
 - Interactive editing
 - Export modifications
```

## Example Processing Times
- Decoder: ~0.5s for 1MB file
- Parser: ~1.2s including visualizations
- Viewer: Real-time updates (60 FPS)

Note: Processing times may vary based on map size and system specifications.
