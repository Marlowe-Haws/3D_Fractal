import numpy as np
import numba
from numba.typed import Dict
from vispy import scene, app, color # Added VisPy imports
from vispy.scene import visuals # Explicit import for Mesh
from collections import defaultdict
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
def get_next_color(current_color):
    """Gets the next color in the sequence."""
    try: idx = COLOR_SEQUENCE.index(current_color); return COLOR_SEQUENCE[(idx + 1) % len(COLOR_SEQUENCE)]
    except ValueError: return COLOR_SEQUENCE[0]
def get_required_parent_color(target_placement_color):
    """Gets the color a parent cube must be to place the target color."""
    try: idx = COLOR_SEQUENCE.index(target_placement_color); parent_idx = (idx - 1 + len(COLOR_SEQUENCE)) % len(COLOR_SEQUENCE); return COLOR_SEQUENCE[parent_idx]
    except ValueError: return None
# --- Numba Accelerated Helper Functions ---
@numba.jit(nopython=True)
def get_cube_vertices_tuple_list(x, y, z, S):
    """Numba-friendly version returning a list of tuples for vertices."""
    vertices_list = []
    for dx in [0, S]:
        for dy in [0, S]:
            for dz in [0, S]: vertices_list.append((x + dx, y + dy, z + dz))
    return vertices_list
@numba.jit(nopython=True)
def get_cube_blocks_list(x, y, z, S):
    """Numba-friendly version returning a list of tuples for occupied blocks."""
    blocks_list = []
    for i in range(S):
        for j in range(S):
            for k in range(S): blocks_list.append((x + i, y + j, z + k))
    return blocks_list
@numba.jit(nopython=True)
def distance_to_origin(vx, vy, vz): # Pass components
    """Numba-friendly distance calculation."""
    ox, oy, oz = 0.5, 0.5, 0.5
    return math.sqrt((vx - ox)**2 + (vy - oy)**2 + (vz - oz)**2)
@numba.jit(nopython=True)
def check_placement(x, y, z, S, current_occupied_blocks_dict, # Accepts TYPED Dict
                  bounds_min_x, bounds_min_y, bounds_min_z,
                  bounds_max_x, bounds_max_y, bounds_max_z):
    """
    Numba-JITted check_placement. Checks bounds, face/edge contact, volume overlap.
    Requires occupied blocks as a Numba typed Dict {block_tuple: bool}.
    Bounds components passed explicitly.
    """
    if S < 1: return False
    has_bounds = bounds_min_x != -9999
    if has_bounds:
        if (x < bounds_min_x or (x + S) > bounds_max_x or
            y < bounds_min_y or (y + S) > bounds_max_y or
            z < bounds_min_z or (z + S) > bounds_max_z): return False
    # Face Contact Check... (using 'in current_occupied_blocks_dict')
    for face_dim in range(3):
        for face_dir in [-1, 1]:
            coord_at_face = 0
            if face_dim == 0: coord_at_face = x + (face_dir * S if face_dir == 1 else face_dir)
            elif face_dim == 1: coord_at_face = y + (face_dir * S if face_dir == 1 else face_dir)
            else: coord_at_face = z + (face_dir * S if face_dir == 1 else face_dir)
            for i in range(S):
                for j in range(S):
                    neighbor_block = (0,0,0)
                    if face_dim == 0: neighbor_block = (coord_at_face, y + i, z + j)
                    elif face_dim == 1: neighbor_block = (x + i, coord_at_face, z + j)
                    else: neighbor_block = (x + i, y + j, coord_at_face)
                    # Check if key exists in the dictionary
                    if neighbor_block in current_occupied_blocks_dict: return False
    # Edge Contact Check... (using 'in current_occupied_blocks_dict')
    for edge_dim1 in range(3):
        for edge_dir1 in [-1, S]:
            coord1 = 0
            if edge_dim1 == 0: coord1 = x + edge_dir1
            elif edge_dim1 == 1: coord1 = y + edge_dir1
            else: coord1 = z + edge_dir1
            for edge_dim2 in range(edge_dim1 + 1, 3):
                 for edge_dir2 in [-1, S]:
                    coord2 = 0
                    if edge_dim2 == 0: coord2 = x + edge_dir2
                    elif edge_dim2 == 1: coord2 = y + edge_dir2
                    else: coord2 = z + edge_dir2
                    edge_iter_dim = 3 - edge_dim1 - edge_dim2
                    base_coord3 = 0
                    if edge_iter_dim == 0: base_coord3 = x
                    elif edge_iter_dim == 1: base_coord3 = y
                    else: base_coord3 = z
                    for k in range(S):
                        iter_coord3 = base_coord3 + k
                        nb0, nb1, nb2 = 0, 0, 0
                        if edge_dim1 == 0: nb0 = coord1
                        elif edge_dim1 == 1: nb1 = coord1
                        else: nb2 = coord1
                        if edge_dim2 == 0: nb0 = coord2
                        elif edge_dim2 == 1: nb1 = coord2
                        else: nb2 = coord2
                        if edge_iter_dim == 0: nb0 = iter_coord3
                        elif edge_iter_dim == 1: nb1 = iter_coord3
                        else: nb2 = iter_coord3
                        neighbor_block_tuple = (nb0, nb1, nb2)
                        # Check if key exists in the dictionary
                        if neighbor_block_tuple in current_occupied_blocks_dict: return False
    # Volume Overlap Check... (using 'in current_occupied_blocks_dict')
    for i in range(S):
        for j in range(S):
            for k in range(S):
                block = (x + i, y + j, z + k)
                # Check if key exists in the dictionary
                if block in current_occupied_blocks_dict: return False
    return True
@numba.jit(nopython=True)
def get_placement_coords_at_vertex(vertex_x, vertex_y, vertex_z, # Pass components
                                 cube_size_S,
                                 parent_coords_x, parent_coords_y, parent_coords_z, # Pass components
                                 parent_size):
    """Numba-friendly version. Determines BBL coords."""
    if parent_size < 1 or cube_size_S < 1: return (-9999, -9999, -9999)
    is_valid_vertex = False
    for dx in [0, parent_size]:
        for dy in [0, parent_size]:
            for dz in [0, parent_size]:
                if (parent_coords_x + dx == vertex_x and
                    parent_coords_y + dy == vertex_y and
                    parent_coords_z + dz == vertex_z):
                    is_valid_vertex = True; break
            if is_valid_vertex: break
        if is_valid_vertex: break
    if not is_valid_vertex: return (-9999, -9999, -9999)
    nx = vertex_x - cube_size_S if vertex_x == parent_coords_x else vertex_x
    ny = vertex_y - cube_size_S if vertex_y == parent_coords_y else vertex_y
    nz = vertex_z - cube_size_S if vertex_z == parent_coords_z else vertex_z
    return (nx, ny, nz)
# --- Non-JITted place_cube (Accepts and Updates Numba Typed Dict) ---
def place_cube(x, y, z, S, color, occupied_blocks_numba_dict): # Takes 6 arguments, accepts Dict
    """Adds a cube with color, updates structures including vertex_info."""
    global placed_cubes, occupied_blocks, vertex_counts, vertex_info
    global bounds_min_int, bounds_max_int
    if S < 1: return False
    # Prepare bounds components
    b_min_x = bounds_min_int[0] if bounds_min_int is not None else -9999
    b_min_y = bounds_min_int[1] if bounds_min_int is not None else -9999
    b_min_z = bounds_min_int[2] if bounds_min_int is not None else -9999
    b_max_x = bounds_max_int[0] if bounds_max_int is not None else -9999
    b_max_y = bounds_max_int[1] if bounds_max_int is not None else -9999
    b_max_z = bounds_max_int[2] if bounds_max_int is not None else -9999
    # Call the JITted check_placement, passing the TYPED Dict
    if not check_placement(x, y, z, S, occupied_blocks_numba_dict, # Pass TYPED Dict
                         b_min_x, b_min_y, b_min_z,
                         b_max_x, b_max_y, b_max_z):
         return False
    # --- Placement is geometrically valid, update standard Python structures ---
    cube_id = len(placed_cubes)
    placed_cubes.append((x, y, z, S, color))
    # Update the Python set AND the Numba Dict
    blocks_to_add = get_cube_blocks_list(x, y, z, S) # Use JITted list version
    occupied_blocks.update(blocks_to_add) # Update standard Python set
    # Update Numba Dict
    for block in blocks_to_add:
         occupied_blocks_numba_dict[block] = True # Add key to typed dict
    # Update vertex counts and vertex_info (standard Python dicts)
    new_vertices_list = get_cube_vertices_tuple_list(x, y, z, S) # Use JITted list version
    for vertex in new_vertices_list: # Iterate through list
        vertex_counts[vertex] += 1
        current_count = vertex_counts[vertex]
        if current_count == 1:
            vertex_info[vertex] = {
                'parent_id': cube_id, 'parent_size': S, 'parent_color': color
            }
        elif current_count > 1 and vertex in vertex_info:
            del vertex_info[vertex]
    return True
# --- NEW: Helper Function to Generate Mesh Data ---
# (generate_cube_mesh_data function remains unchanged)
def generate_cube_mesh_data(cubes_list):
    """Generates vertices, faces, and colors for a Mesh visual from a list of cubes."""
    num_cubes = len(cubes_list)
    if num_cubes == 0: return None, None, None
    unit_v = np.array([[-0.5,-0.5,-0.5],[+0.5,-0.5,-0.5],[+0.5,+0.5,-0.5],[-0.5,+0.5,-0.5],
                       [-0.5,-0.5,+0.5],[+0.5,-0.5,+0.5],[+0.5,+0.5,+0.5],[-0.5,+0.5,+0.5]])
    unit_f = np.array([[0,1,2],[0,2,3],[4,7,6],[4,6,5],[0,4,5],[0,5,1],
                       [3,2,6],[3,6,7],[0,3,7],[0,7,4],[1,5,6],[1,6,2]], dtype=np.uint32)
    all_vertices = np.zeros((num_cubes * 8, 3), dtype=np.float32)
    all_faces = np.zeros((num_cubes * 12, 3), dtype=np.uint32)
    all_colors = np.zeros((num_cubes * 8, 4), dtype=np.float32)
    print(f"Generating mesh data for {num_cubes} cubes...")
    start_mesh_gen = time.time()
    for i, (x, y, z, S, color_name) in enumerate(cubes_list):
        center = np.array([x + S/2.0, y + S/2.0, z + S/2.0])
        scaled_vertices = unit_v * S + center
        v_offset = i * 8; f_offset = i * 12
        all_vertices[v_offset : v_offset + 8] = scaled_vertices
        all_faces[f_offset : f_offset + 12] = unit_f + v_offset
        hex_color = COLOR_MAP_PLOT.get(color_name, COLOR_MAP_PLOT['default'])
        try: rgba = color.Color(hex_color).rgba
        except Exception: rgba = color.Color(COLOR_MAP_PLOT['default']).rgba
        all_colors[v_offset : v_offset + 8] = rgba
    gen_duration = time.time() - start_mesh_gen
    print(f"Mesh data generation complete ({gen_duration:.2f}s).")
    return all_vertices, all_faces, all_colors
# ===============================
# --- Main Program Loop ---
# ===============================
while True:
    # --- Reset State Variables ---
    placed_cubes = []
    occupied_blocks = set() # Standard Python set for reference if needed
    vertex_counts = defaultdict(int)
    vertex_info = {}
    bounds_min_int = None
    bounds_max_int = None
    # --- Create Numba Typed Dict for occupied blocks ---
    # Define the type of the elements (key: 3-tuple of int64, value: boolean)
    block_type = numba.types.UniTuple(numba.int64, 3)
    value_type = numba.boolean
    # Create the empty typed dict using the imported Dict
    occupied_blocks_nb_dict = Dict.empty(key_type=block_type, value_type=value_type)
    # --- Get User Input for N_FRAMEWORK or Quit ---
    print("\n" + "=" * 69)
    print("Welcome to Marlowe's 3D Fractal Art Generator (VisPy + Numba Version)")
    print("This fractal is initially defined as a framework of size N.")
    print("(N = number of layers added after the initial 1x1x1 cube)")
    print("Higher N values increase complexity exponentially.")
    print("Recommended: N=3 to N=5. N=6+ requires significant time/memory.")
    print("Enter 'q' to quit.")
    print("=" * 69)
    N_FRAMEWORK = None
    while True: # Inner loop for input validation
        framework_input = input("\nEnter the desired value for N (e.g., 3) or 'q' to quit: ").strip().lower()
        if framework_input == 'q': break
        try:
            N_FRAMEWORK_temp = int(framework_input)
            if N_FRAMEWORK_temp > 0: N_FRAMEWORK = N_FRAMEWORK_temp; print(f"Using N_FRAMEWORK = {N_FRAMEWORK}"); break
            else: print("Error: Please enter a positive whole number (greater than 0).")
        except ValueError: print("Error: Invalid input. Please enter a whole number or 'q'.")
    # --- Check if user chose to quit ---
    if N_FRAMEWORK is None: print("Exiting program. Goodbye!"); break # Exit main loop
    # --- Phase 1: Framework Generation ---
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    # Place initiator - Pass the Numba dict to place_cube
    if not place_cube(0, 0, 0, 1, COLOR_SEQUENCE[0], occupied_blocks_nb_dict):  # Pass Numba dict
        print("CRITICAL ERROR: Initiator failed.");
        continue
    # Framework setup...
    framework_paths = {};
    origin = np.array([0.5, 0.5, 0.5])
    initiator_vertices_list = get_cube_vertices_tuple_list(0, 0, 0, 1)
    for v in initiator_vertices_list:
        direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
        if all(c != 0 for c in direction_vector): framework_paths[direction_vector] = 0
    # Framework iterations...
    for n in range(1, N_FRAMEWORK + 1):
        target_size = n + 1;
        layer_color = COLOR_SEQUENCE[n % len(COLOR_SEQUENCE)]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color}")
        new_framework_paths = {};
        placed_count_iter = 0;
        current_paths = framework_paths.copy()
        for direction_vec, last_cube_id in current_paths.items():
            if not (0 <= last_cube_id < len(placed_cubes)): continue
            lx, ly, lz, lS, _ = placed_cubes[last_cube_id]
            parent_size_phase1 = lS
            last_cube_verts_list = get_cube_vertices_tuple_list(lx, ly, lz, lS)
            if not last_cube_verts_list: continue
            cube_center = np.array([lx + lS / 2, ly + lS / 2, lz + lS / 2])
            try:
                attachment_vertex = max(last_cube_verts_list,
                                        key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError:
                continue
            vx, vy, vz = attachment_vertex
            placement_result_tuple = get_placement_coords_at_vertex(
                vx, vy, vz, target_size, lx, ly, lz, parent_size_phase1
            )
            if placement_result_tuple == (-9999, -9999, -9999): continue
            px, py, pz = placement_result_tuple
            # Attempt to place the new cube - Pass Numba dict
            if place_cube(px, py, pz, target_size, layer_color, occupied_blocks_nb_dict):  # Pass Numba dict
                new_cube_id = len(placed_cubes) - 1
                new_framework_paths[direction_vec] = new_cube_id
                placed_count_iter += 1
            # else: pass
        framework_paths.update(new_framework_paths);
        print(f"  Placed {placed_count_iter} framework cubes.")
        if placed_count_iter != 8 and n > 0: print(
            f"Warning: Placed {placed_count_iter}/8 framework cubes in iteration {n}")
    duration_phase1 = time.time() - start_time_phase1;
    print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---");
    print(f"Total cubes after Phase 1: {len(placed_cubes)}")
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
    start_time_phase2 = time.time();
    max_color_cycles = 100;
    cycles_without_placement = 0
    current_target_color_index = 1;
    total_placed_phase2 = 0;
    cycle = 0
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles:  # Phase 2 Loop
        cycle += 1;
        target_color_to_place = COLOR_SEQUENCE[current_target_color_index % len(COLOR_SEQUENCE)]
        required_parent_color = get_required_parent_color(target_color_to_place)
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_to_place} onto {required_parent_color}")
        # 1. Find candidates
        parent_vertices_candidates = [];
        current_vertex_info_snapshot = vertex_info.copy()
        for V, info in current_vertex_info_snapshot.items():
            if info['parent_color'] == required_parent_color and 0 <= info['parent_id'] < len(placed_cubes):
                parent_vertices_candidates.append(
                    {'vertex': V, 'parent_id': info['parent_id'], 'parent_size': info['parent_size']})
        # 1b. Pre-filter
        parent_vertices_found = []
        for candidate in parent_vertices_candidates:
            V = candidate['vertex'];
            parent_id = candidate['parent_id'];
            parent_size = candidate['parent_size']
            px_chk, py_chk, pz_chk, _, _ = placed_cubes[parent_id]
            vx, vy, vz = V
            placement_s1_tuple = get_placement_coords_at_vertex(
                vx, vy, vz, 1, px_chk, py_chk, pz_chk, parent_size
            )
            if placement_s1_tuple == (-9999, -9999, -9999): continue
            px1, py1, pz1 = placement_s1_tuple
            if not (px1 < bounds_min_int[0] or (px1 + 1) > bounds_max_int[0] or py1 < bounds_min_int[1] or (py1 + 1) >
                    bounds_max_int[1] or pz1 < bounds_min_int[2] or (pz1 + 1) > bounds_max_int[2]):
                dist = distance_to_origin(vx, vy, vz)  # Call JITted version
                parent_vertices_found.append(
                    {'vertex': V, 'distance': dist, 'parent_size': parent_size, 'parent_id': parent_id})
        if not parent_vertices_found: print(
            f"  No valid exposed vertices found for {required_parent_color} (after bounds filter)."); cycles_without_placement += 1; current_target_color_index += 1; continue
        print(f"  Found {len(parent_vertices_found)} potential parent vertices ({required_parent_color}).")
        # 2. Sort
        parent_vertices_found.sort(key=lambda x: x['distance'])
        # 3. Attempt placement
        placed_this_color_cycle = 0;
        processed_vertices_this_cycle = set()
        for parent_data in parent_vertices_found:
            V = parent_data['vertex']
            if V in processed_vertices_this_cycle or V not in vertex_info: continue
            S_parent = parent_data['parent_size'];
            parent_id = parent_data['parent_id']
            if not (0 <= parent_id < len(placed_cubes)): continue
            parent_x, parent_y, parent_z, pS_check, _ = placed_cubes[parent_id]
            if pS_check != S_parent: print(f"Warning: Mismatch parent size ID {parent_id}")
            current_parent_size = S_parent
            target_size_start = S_parent + 1;
            best_S_found = 0;
            placement_coords = None
            # Find largest size loop
            for S_try in range(target_size_start, 0, -1):
                vx, vy, vz = V
                px_py_pz_tuple = get_placement_coords_at_vertex(
                    vx, vy, vz, S_try, parent_x, parent_y, parent_z, current_parent_size
                )
                if px_py_pz_tuple == (-9999, -9999, -9999): continue
                px, py, pz = px_py_pz_tuple
                b_min_x = bounds_min_int[0];
                b_min_y = bounds_min_int[1];
                b_min_z = bounds_min_int[2]
                b_max_x = bounds_max_int[0];
                b_max_y = bounds_max_int[1];
                b_max_z = bounds_max_int[2]
                # Call JITted check_placement, passing the TYPED occupied_blocks_nb_dict
                if check_placement(px, py, pz, S_try, occupied_blocks_nb_dict,  # Pass TYPED Dict
                                   b_min_x, b_min_y, b_min_z,
                                   b_max_x, b_max_y, b_max_z):
                    best_S_found = S_try;
                    placement_coords = (px, py, pz);
                    break
                # else: # Debugging check_placement failures
            # Place cube if found - Pass Numba dict
            if best_S_found > 0 and placement_coords:
                if place_cube(placement_coords[0], placement_coords[1], placement_coords[2],
                              best_S_found, target_color_to_place, occupied_blocks_nb_dict):  # Pass Numba dict
                    placed_this_color_cycle += 1;
                    processed_vertices_this_cycle.add(V)
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_to_place}.")
        if placed_this_color_cycle > 0:
            cycles_without_placement = 0
        else:
            cycles_without_placement += 1
        current_target_color_index += 1
    # --- End Phase 2 Loop ---
    duration_phase2 = time.time() - start_time_phase2;
    print(f"--- Phase 2 Complete ({duration_phase2:.2f}s in {cycle} color cycles) ---")
    print(f"Total cubes placed in Phase 2: {total_placed_phase2}");
    print(f"Total cubes after Phase 2: {len(placed_cubes)}")
    print(f"Total occupied 1x1x1 blocks: {len(occupied_blocks)}");
    print(f"Total unique vertices created: {len(vertex_counts)}")
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
print("--- Script Finished ---")
