import numpy as np
# Removed: import matplotlib.pyplot as plt
# Removed: import matplotlib.colors as mcolors
from vispy import scene, app, color # Added VisPy imports
from vispy.scene import visuals # Explicit import for Mesh
from collections import defaultdict
import itertools
import math
import time
# --- Configuration ---
# N_FRAMEWORK will be set by user input
# --- Color Definitions ---
COLOR_SEQUENCE = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
COLOR_MAP_PLOT = {
    'red': '#FF0000', 'orange': '#FFA500', 'yellow': '#FFFF00',
    'green': '#00FF00', 'blue': '#0000FF', 'purple': '#800080',
    'default': '#555555'
}
# --- Data Structures ---
# (Remain the same)
placed_cubes = []
occupied_blocks = set()
vertex_counts = defaultdict(int)
vertex_info = {}
bounds_min_int = None
bounds_max_int = None
# --- Helper Functions ---
# (get_next_color, get_required_parent_color, get_cube_vertices,
#  get_cube_blocks, distance_to_origin, check_placement,
#  get_placement_coords_at_vertex, place_cube remain the same
#  as the last working version)
# --- check_placement (Strict Adjacency Check - Prevents Face AND Edge Contact) ---
# --- Helper Functions ---
def get_next_color(current_color):
    """Gets the next color in the sequence."""
    try:
        idx = COLOR_SEQUENCE.index(current_color)
        return COLOR_SEQUENCE[(idx + 1) % len(COLOR_SEQUENCE)]
    except ValueError:
        return COLOR_SEQUENCE[0] # Default to red if error
def get_required_parent_color(target_placement_color):
    """Gets the color a parent cube must be to place the target color."""
    try:
        idx = COLOR_SEQUENCE.index(target_placement_color)
        parent_idx = (idx - 1 + len(COLOR_SEQUENCE)) % len(COLOR_SEQUENCE)
        return COLOR_SEQUENCE[parent_idx]
    except ValueError:
        return None
def get_cube_vertices(x, y, z, S):
    """Returns the 8 vertices of a cube."""
    vertices = set()
    for dx, dy, dz in itertools.product([0, S], repeat=3):
        vertices.add((x + dx, y + dy, z + dz))
    return vertices
def get_cube_blocks(x, y, z, S):
    """Generator for the 1x1x1 blocks occupied by a cube."""
    for i in range(S):
        for j in range(S):
            for k in range(S):
                yield (x + i, y + j, z + k)
def distance_to_origin(vertex):
    """Calculates Euclidean distance from a vertex to the fractal origin (0.5, 0.5, 0.5)."""
    ox, oy, oz = 0.5, 0.5, 0.5
    vx, vy, vz = vertex
    return math.sqrt((vx - ox)**2 + (vy - oy)**2 + (vz - oz)**2)
# --- check_placement (Strict Adjacency Check - Prevents Face AND Edge Contact) ---
# (check_placement definition follows...)
def get_cube_blocks(x, y, z, S):
    """Generator for the 1x1x1 blocks occupied by a cube."""
    for i in range(S):
        for j in range(S):
            for k in range(S):
                yield (x + i, y + j, z + k)
def check_placement(x, y, z, S, current_occupied_blocks, current_vertex_counts,
                  bounds_min=None, bounds_max=None):
    if S < 1: return False
    if bounds_min is not None and bounds_max is not None: # Bounds Check
        if (x < bounds_min[0] or (x + S) > bounds_max[0] or
            y < bounds_min[1] or (y + S) > bounds_max[1] or
            z < bounds_min[2] or (z + S) > bounds_max[2]): return False
    for face_dim in range(3): # Face Contact Check
        for face_dir in [-1, 1]:
            neighbor_coords_on_face = []
            if face_dir == -1: coord_at_face = (x, y, z)[face_dim] - 1
            else: coord_at_face = (x, y, z)[face_dim] + S
            if face_dim == 0: # X face
                for j in range(S):
                    for k in range(S): neighbor_coords_on_face.append((coord_at_face, y + j, z + k))
            elif face_dim == 1: # Y face
                for i in range(S):
                    for k in range(S): neighbor_coords_on_face.append((x + i, coord_at_face, z + k))
            else: # Z face
                for i in range(S):
                    for j in range(S): neighbor_coords_on_face.append((x + i, y + j, coord_at_face))
            for neighbor_block in neighbor_coords_on_face:
                if neighbor_block in current_occupied_blocks: return False # Face Contact
    for edge_dim1 in range(3): # Edge Contact Check
        for edge_dir1 in [-1, S]:
            coord1 = (x, y, z)[edge_dim1] + edge_dir1
            for edge_dim2 in range(edge_dim1 + 1, 3):
                 for edge_dir2 in [-1, S]:
                    coord2 = (x, y, z)[edge_dim2] + edge_dir2
                    edge_iter_dim = 3 - edge_dim1 - edge_dim2
                    base_coord3 = (x, y, z)[edge_iter_dim]
                    for k in range(S):
                        iter_coord3 = base_coord3 + k
                        neighbor_block = [0, 0, 0]; neighbor_block[edge_dim1] = coord1
                        neighbor_block[edge_dim2] = coord2; neighbor_block[edge_iter_dim] = iter_coord3
                        if tuple(neighbor_block) in current_occupied_blocks: return False # Edge Contact
    for block in get_cube_blocks(x, y, z, S): # Volume Overlap Check
        if block in current_occupied_blocks: return False
    vertices = get_cube_vertices(x, y, z, S) # Vertex Sharing Limit Check
    for v in vertices:
        if current_vertex_counts.get(v, 0) >= 2: return False
    return True
# --- get_placement_coords_at_vertex (Requires Parent Info for Corner Attachment) ---
def get_placement_coords_at_vertex(vertex, cube_size_S, parent_coords, parent_size):
    if parent_coords is None or parent_size is None or parent_size < 1 or cube_size_S < 1: return None
    vx, vy, vz = vertex; px, py, pz = parent_coords; pS = parent_size
    is_valid_vertex = False
    for dx, dy, dz in itertools.product([0, pS], repeat=3):
        if (px + dx, py + dy, pz + dz) == vertex: is_valid_vertex = True; break
    if not is_valid_vertex: return None
    nx = vx - cube_size_S if vx == px else vx
    ny = vy - cube_size_S if vy == py else vy
    nz = vz - cube_size_S if vz == pz else vz
    return nx, ny, nz
# --- place_cube (Simplified - No Parent Info Needed for Checks) ---
def place_cube(x, y, z, S, color):
    global placed_cubes, occupied_blocks, vertex_counts, vertex_info, bounds_min_int, bounds_max_int
    if S < 1: return False
    is_phase2 = bounds_min_int is not None
    if not check_placement(x, y, z, S, occupied_blocks, vertex_counts,
                         bounds_min=bounds_min_int if is_phase2 else None,
                         bounds_max=bounds_max_int if is_phase2 else None): return False
    cube_id = len(placed_cubes); placed_cubes.append((x, y, z, S, color))
    occupied_blocks.update(get_cube_blocks(x, y, z, S))
    new_vertices = get_cube_vertices(x, y, z, S)
    for vertex in new_vertices:
        vertex_counts[vertex] += 1; current_count = vertex_counts[vertex]
        if current_count == 1:
            vertex_info[vertex] = {'parent_id': cube_id, 'parent_size': S, 'parent_color': color}
        elif current_count > 1 and vertex in vertex_info: del vertex_info[vertex]
    return True
# --- NEW: Helper Function to Generate Mesh Data ---
def generate_cube_mesh_data(cubes_list):
    """Generates vertices, faces, and colors for a Mesh visual from a list of cubes."""
    num_cubes = len(cubes_list)
    if num_cubes == 0:
        return None, None, None
    # Define vertices & faces for a unit cube centered at origin (adjust later)
    # Vertices (8 corners)
    unit_v = np.array([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5]
    ])
    # Faces (12 triangles, 2 per face) using vertex indices (0-7)
    unit_f = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face (-Z)
        [4, 7, 6], [4, 6, 5],  # Top face (+Z)
        [0, 4, 5], [0, 5, 1],  # Back face (-Y)
        [3, 2, 6], [3, 6, 7],  # Front face (+Y)
        [0, 3, 7], [0, 7, 4],  # Left face (-X)
        [1, 5, 6], [1, 6, 2]   # Right face (+X)
    ], dtype=np.uint32)
    # Pre-allocate arrays
    all_vertices = np.zeros((num_cubes * 8, 3), dtype=np.float32)
    all_faces = np.zeros((num_cubes * 12, 3), dtype=np.uint32)
    all_colors = np.zeros((num_cubes * 8, 4), dtype=np.float32) # Per-vertex color
    print(f"Generating mesh data for {num_cubes} cubes...")
    start_mesh_gen = time.time()
    for i, (x, y, z, S, color_name) in enumerate(cubes_list):
        # Calculate cube center and scale vertices
        center = np.array([x + S/2.0, y + S/2.0, z + S/2.0])
        scaled_vertices = unit_v * S + center
        # Add to main arrays
        v_offset = i * 8
        f_offset = i * 12
        all_vertices[v_offset : v_offset + 8] = scaled_vertices
        all_faces[f_offset : f_offset + 12] = unit_f + v_offset # Offset face indices
        # Get RGBA color
        hex_color = COLOR_MAP_PLOT.get(color_name, COLOR_MAP_PLOT['default'])
        try:
            rgba = color.Color(hex_color).rgba
        except Exception:
            rgba = color.Color(COLOR_MAP_PLOT['default']).rgba # Fallback
        # Assign color to all 8 vertices of this cube
        all_colors[v_offset : v_offset + 8] = rgba
    gen_duration = time.time() - start_mesh_gen
    print(f"Mesh data generation complete ({gen_duration:.2f}s).")
    return all_vertices, all_faces, all_colors
# ===============================
# --- Main Program Loop ---
# ===============================
while True:
    # --- Reset State Variables ---
    placed_cubes = []; occupied_blocks = set(); vertex_counts = defaultdict(int)
    vertex_info = {}; bounds_min_int = None; bounds_max_int = None
    print("\n" + "="*40)
    # --- Get User Input ---
    print("Welcome to Marlowe's 3D Fractal Art Generator (VisPy Mesh Version)")
    # (Print instructions...)
    print("Enter 'q' to quit.")
    N_FRAMEWORK = None
    while True:
        framework_input = input("Enter the desired value for N (e.g., 3) or 'q' to quit: ").strip().lower()
        if framework_input == 'q': break
        try:
            N_FRAMEWORK_temp = int(framework_input)
            if N_FRAMEWORK_temp > 0: N_FRAMEWORK = N_FRAMEWORK_temp; print(f"Using N_FRAMEWORK = {N_FRAMEWORK}"); break
            else: print("Error: Please enter a positive whole number (greater than 0).")
        except ValueError: print("Error: Invalid input. Please enter a whole number or 'q'.")
    if N_FRAMEWORK is None: print("Exiting program. Goodbye!"); break # Exit main loop
    # --- Phase 1: Framework Generation ---
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    if not place_cube(0, 0, 0, 1, COLOR_SEQUENCE[0]): print("CRITICAL ERROR: Initiator failed."); continue
    framework_paths = {}; origin = np.array([0.5, 0.5, 0.5])
    initiator_vertices = list(get_cube_vertices(0, 0, 0, 1))
    for v in initiator_vertices:
        direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
        if all(c != 0 for c in direction_vector): framework_paths[direction_vector] = 0
    for n in range(1, N_FRAMEWORK + 1): # Framework iterations
        target_size = n + 1; layer_color = COLOR_SEQUENCE[n % len(COLOR_SEQUENCE)]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color}")
        new_framework_paths = {}; placed_count_iter = 0; current_paths = framework_paths.copy()
        for direction_vec, last_cube_id in current_paths.items():
            if not (0 <= last_cube_id < len(placed_cubes)): continue
            lx, ly, lz, lS, _ = placed_cubes[last_cube_id]; parent_coords_phase1 = (lx, ly, lz); parent_size_phase1 = lS
            last_cube_verts = get_cube_vertices(lx, ly, lz, lS); cube_center = np.array([lx + lS/2, ly + lS/2, lz + lS/2])
            try: attachment_vertex = max(last_cube_verts, key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError: continue
            placement_result = get_placement_coords_at_vertex(attachment_vertex, target_size, parent_coords_phase1, parent_size_phase1)
            if placement_result is None: continue
            px, py, pz = placement_result
            if place_cube(px, py, pz, target_size, layer_color):
                new_cube_id = len(placed_cubes) - 1; new_framework_paths[direction_vec] = new_cube_id; placed_count_iter += 1
        framework_paths.update(new_framework_paths); print(f"  Placed {placed_count_iter} framework cubes.")
        if placed_count_iter != 8 and n > 0: print(f"Warning: Placed {placed_count_iter}/8 framework cubes in iteration {n}")
    duration_phase1 = time.time() - start_time_phase1; print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---"); print(f"Total cubes after Phase 1: {len(placed_cubes)}")
    if len(placed_cubes) <= 1 and N_FRAMEWORK > 0: print("Warning: Phase 1 failed. Skipping Phase 2."); continue
    # --- Calculate Final Bounding Box ---
    print("\n--- Calculating Final Bounding Box ---")
    min_coord_f = np.array([float('inf')] * 3); max_coord_f = np.array([float('-inf')] * 3)
    if not placed_cubes: print("Error: No cubes placed. Skipping."); continue
    for x, y, z, S, _ in placed_cubes: min_coord_f = np.minimum(min_coord_f, [x, y, z]); max_coord_f = np.maximum(max_coord_f, [x + S, y + S, z + S])
    bounds_min_int = np.floor(min_coord_f).astype(int); bounds_max_int = np.ceil(max_coord_f).astype(int)
    print(f"Bounding Box Min (int, BBL inclusive): {bounds_min_int}"); print(f"Bounding Box Max (int, exclusive): {bounds_max_int}")
    if np.any(bounds_max_int <= bounds_min_int): print("ERROR: Invalid bounding box. Skipping."); continue
    # --- Phase 2: Color-Based Generational Layering ---
    print(f"\n--- Starting Phase 2: Color-Based Generational Layering ---")
    start_time_phase2 = time.time(); max_color_cycles = 100; cycles_without_placement = 0
    current_target_color_index = 1; total_placed_phase2 = 0; cycle = 0
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles: # Phase 2 Loop
        cycle += 1; target_color_to_place = COLOR_SEQUENCE[current_target_color_index % len(COLOR_SEQUENCE)]
        required_parent_color = get_required_parent_color(target_color_to_place)
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_to_place} onto {required_parent_color}")
        parent_vertices_candidates = []; current_vertex_info_snapshot = vertex_info.copy() # Find candidates
        for V, info in current_vertex_info_snapshot.items():
            if info['parent_color'] == required_parent_color and 0 <= info['parent_id'] < len(placed_cubes):
                 parent_vertices_candidates.append({'vertex': V, 'parent_id': info['parent_id'], 'parent_size': info['parent_size']})
        parent_vertices_found = [] # Pre-filter
        for candidate in parent_vertices_candidates:
            V = candidate['vertex']; parent_id = candidate['parent_id']; parent_size = candidate['parent_size']
            px_chk, py_chk, pz_chk, _, _ = placed_cubes[parent_id]; parent_coords_check = (px_chk, py_chk, pz_chk)
            placement_s1 = get_placement_coords_at_vertex(V, 1, parent_coords_check, parent_size)
            if placement_s1 is None: continue
            px1, py1, pz1 = placement_s1
            if not (px1 < bounds_min_int[0] or (px1 + 1) > bounds_max_int[0] or py1 < bounds_min_int[1] or (py1 + 1) > bounds_max_int[1] or pz1 < bounds_min_int[2] or (pz1 + 1) > bounds_max_int[2]):
                parent_vertices_found.append({'vertex': V, 'distance': distance_to_origin(V), 'parent_size': parent_size, 'parent_id': parent_id})
        if not parent_vertices_found: print(f"  No valid exposed vertices found for {required_parent_color} (after bounds filter)."); cycles_without_placement += 1; current_target_color_index += 1; continue
        print(f"  Found {len(parent_vertices_found)} potential parent vertices ({required_parent_color}).")
        parent_vertices_found.sort(key=lambda x: x['distance']) # Sort
        placed_this_color_cycle = 0; processed_vertices_this_cycle = set() # Attempt placement
        for parent_data in parent_vertices_found:
            V = parent_data['vertex']
            if V in processed_vertices_this_cycle or V not in vertex_info: continue
            S_parent = parent_data['parent_size']; parent_id = parent_data['parent_id']
            parent_x, parent_y, parent_z, pS_check, _ = placed_cubes[parent_id]
            if pS_check != S_parent: print(f"Warning: Mismatch parent size ID {parent_id}")
            current_parent_coords = (parent_x, parent_y, parent_z); current_parent_size = S_parent
            target_size_start = S_parent + 1; best_S_found = 0; placement_coords = None
            for S_try in range(target_size_start, 0, -1): # Find largest size
                px_py_pz = get_placement_coords_at_vertex(V, S_try, current_parent_coords, current_parent_size)
                if px_py_pz is None: continue
                px, py, pz = px_py_pz
                if check_placement(px, py, pz, S_try, occupied_blocks, vertex_counts, bounds_min_int, bounds_max_int):
                    best_S_found = S_try; placement_coords = (px, py, pz); break
            if best_S_found > 0 and placement_coords: # Place cube
                if place_cube(placement_coords[0], placement_coords[1], placement_coords[2], best_S_found, target_color_to_place):
                    placed_this_color_cycle += 1; processed_vertices_this_cycle.add(V)
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_to_place}.")
        if placed_this_color_cycle > 0: cycles_without_placement = 0
        else: cycles_without_placement += 1
        current_target_color_index += 1
    duration_phase2 = time.time() - start_time_phase2; print(f"--- Phase 2 Complete ({duration_phase2:.2f}s in {cycle} color cycles) ---")
    print(f"Total cubes placed in Phase 2: {total_placed_phase2}"); print(f"Total cubes after Phase 2: {len(placed_cubes)}")
    print(f"Total occupied 1x1x1 blocks: {len(occupied_blocks)}"); print(f"Total unique vertices created: {len(vertex_counts)}")
    print(f"Final exposed vertices map size: {len(vertex_info)}")
    # ====================
    # --- VisPy Visualization (Using Mesh) ---
    # ====================
    print("\n--- Preparing VisPy Mesh Visualization ---")
    if not placed_cubes:
         print("No cubes were placed. Nothing to visualize.")
    else:
        # Generate mesh data
        vertices, faces, vertex_colors = generate_cube_mesh_data(placed_cubes)
        if vertices is None:
            print("Mesh data generation failed.")
        else:
            # Create VisPy Canvas and View
            print("Creating VisPy scene...")
            canvas = scene.SceneCanvas(keys='interactive', show=True, title=f"3D Fractal N={N_FRAMEWORK} (Mesh)")
            view = canvas.central_widget.add_view()
            view.bgcolor = '#222222' # Dark background
            # Create Mesh visual
            print("Adding mesh visual...")
            mesh = visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, shading='flat')
            # mesh.transform = scene.transforms.STTransform(translate=[0, 0, 0], scale=[1, 1, 1]) # Optional transform
            view.add(mesh)
            # Set up the camera
            view.camera = scene.TurntableCamera(fov=60, elevation=30, azimuth=-45)
            if bounds_min_int is not None and bounds_max_int is not None:
                center_point = (bounds_min_int + bounds_max_int) / 2.0
                diag_length = np.linalg.norm(bounds_max_int - bounds_min_int)
                view.camera.distance = max(diag_length * 1.5, 10)
                view.camera.center = tuple(center_point)
            else:
                 view.camera.distance = 50; view.camera.center = (0, 0, 0)
            # Add XYZ Axis
            axis = visuals.XYZAxis(parent=view.scene)
            # Run the VisPy application
            print("\nStarting VisPy visualization...")
            print("Rotate: Left Mouse Button + Drag | Zoom: Scroll Wheel | Pan: Shift + Left Button + Drag")
            print("Close the VisPy window to generate another fractal or enter 'q'.")
            app.run()
            print("VisPy window closed.")
# --- This line is now AFTER the main loop ---
print("--- Script Finished ---")
