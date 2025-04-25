# -*- coding: utf-8 -*-
import numpy as np
import numba
from numba import cuda # Import CUDA features
# from numba.typed import Dict # No longer needed
from vispy import scene, app, color
from vispy.scene import visuals
from collections import defaultdict
import math
import time
import gc # Garbage collector interface for memory info (optional)
# --- Configuration ---
# GRID_CELL_SIZE will be set by user input
BATCH_SIZE = 16110 # Number of S>=2 candidates to check per GPU batch (tuneable)
# --- Color Definitions ---
COLOR_SEQUENCE = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
COLOR_MAP_PLOT = {
    'red': '#FF0000', 'orange': '#FFA500', 'yellow': '#FFFF00',
    'green': '#00FF00', 'blue': '#0000FF', 'purple': '#800080',
    'default': '#555555' # Fallback color
}
# Map color names to indices for storage in vertex_info
COLOR_INDEX_MAP = {name: i for i, name in enumerate(COLOR_SEQUENCE)}
DEFAULT_COLOR_INDEX = -1 # Index for default/unknown color
# --- Data Structures ---
placed_cubes = [] # Back to Python list: [(x, y, z, S, color_name), ...]
spatial_grid = defaultdict(list) # Keys: (cx, cy, cz), Values: list of cube_ids (indices)
vertex_counts = defaultdict(int) # Stores count of cubes sharing a vertex
vertex_info = {} # Stores info about exposed vertices: {vertex_tuple: {'parent_id', 'parent_size', 'parent_color_idx'}}
bounds_min_int = None # Tuple (min_x, min_y, min_z)
bounds_max_int = None # Tuple (max_x, max_y, max_z)
# --- Helper Functions ---
def get_next_color_index(current_color_idx):
    if 0 <= current_color_idx < len(COLOR_SEQUENCE):
        return (current_color_idx + 1) % len(COLOR_SEQUENCE)
    return 0
def get_required_parent_color_index(target_placement_color_idx):
    if 0 <= target_placement_color_idx < len(COLOR_SEQUENCE):
        parent_idx = (target_placement_color_idx - 1 + len(COLOR_SEQUENCE)) % len(COLOR_SEQUENCE)
        return parent_idx
    return None
# --- Numba Accelerated Helper Functions (CPU) ---
@numba.jit(nopython=True)
def get_cube_vertices_tuple_list(x, y, z, S):
    vertices_list = []
    xf, yf, zf = float(x), float(y), float(z)
    Sf = float(S)
    for dx in [0.0, Sf]:
        for dy in [0.0, Sf]:
            for dz in [0.0, Sf]:
                vertices_list.append((int(xf + dx), int(yf + dy), int(zf + dz)))
    return vertices_list
@numba.jit(nopython=True)
def distance_to_origin(vx, vy, vz):
    ox, oy, oz = 0.5, 0.5, 0.5
    return math.sqrt((float(vx) - ox)**2 + (float(vy) - oy)**2 + (float(vz) - oz)**2)
@numba.jit(nopython=True)
def get_contact_type_cpu(x1, y1, z1, S1, x2, y2, z2, S2):
    x1_max, y1_max, z1_max = x1 + S1, y1 + S1, z1 + S1
    x2_max, y2_max, z2_max = x2 + S2, y2 + S2, z2 + S2
    ix_min, ix_max = max(x1, x2), min(x1_max, x2_max)
    iy_min, iy_max = max(y1, y2), min(y1_max, y2_max)
    iz_min, iz_max = max(z1, z2), min(z1_max, z2_max)
    len_x = ix_max - ix_min
    len_y = iy_max - iy_min
    len_z = iz_max - iz_min
    if len_x < 0 or len_y < 0 or len_z < 0: return 0
    zero_dims = 0
    if len_x == 0: zero_dims += 1
    if len_y == 0: zero_dims += 1
    if len_z == 0: zero_dims += 1
    if len_x > 0 and len_y > 0 and len_z > 0: return 4
    if zero_dims == 1: return 3
    if zero_dims == 2: return 2
    if zero_dims == 3: return 1
    return 0
@numba.jit(nopython=True)
def get_placement_coords_at_vertex(vertex_x, vertex_y, vertex_z,
                                 cube_size_S,
                                 parent_coords_x, parent_coords_y, parent_coords_z,
                                 parent_size):
    if parent_size < 1 or cube_size_S < 1: return (-9999, -9999, -9999)
    px_max, py_max, pz_max = parent_coords_x + parent_size, parent_coords_y + parent_size, parent_coords_z + parent_size
    valid_x = (vertex_x == parent_coords_x or vertex_x == px_max)
    valid_y = (vertex_y == parent_coords_y or vertex_y == py_max)
    valid_z = (vertex_z == parent_coords_z or vertex_z == pz_max)
    if not (valid_x and valid_y and valid_z): return (-9999, -9999, -9999)
    nx = vertex_x - cube_size_S if vertex_x == parent_coords_x else vertex_x
    ny = vertex_y - cube_size_S if vertex_y == parent_coords_y else vertex_y
    nz = vertex_z - cube_size_S if vertex_z == parent_coords_z else vertex_z
    return (nx, ny, nz)
# --- Helper function for Spatial Grid ---
def get_cube_cell_bounds(x, y, z, S, cell_size):
    S_eff = max(1, S)
    min_cx = math.floor(x / cell_size)
    max_cx = math.floor((x + S_eff - 1) / cell_size)
    min_cy = math.floor(y / cell_size)
    max_cy = math.floor((y + S_eff - 1) / cell_size)
    min_cz = math.floor(z / cell_size)
    max_cz = math.floor((z + S_eff - 1) / cell_size)
    return min_cx, max_cx, min_cy, max_cy, min_cz, max_cz
# --- Numba CUDA Device Function (Runs on GPU) ---
@cuda.jit(device=True)
def get_contact_type_gpu_device(x1, y1, z1, S1, x2, y2, z2, S2):
    """GPU device function to determine contact type."""
    x1_max, y1_max, z1_max = x1 + S1, y1 + S1, z1 + S1
    x2_max, y2_max, z2_max = x2 + S2, y2 + S2, z2 + S2
    ix_min = max(x1, x2); ix_max = min(x1_max, x2_max)
    iy_min = max(y1, y2); iy_max = min(y1_max, y2_max)
    iz_min = max(z1, z2); iz_max = min(z1_max, z2_max)
    len_x = ix_max - ix_min; len_y = iy_max - iy_min; len_z = iz_max - iz_min
    if len_x < 0 or len_y < 0 or len_z < 0: return 0
    zero_dims = 0
    if len_x == 0: zero_dims += 1
    if len_y == 0: zero_dims += 1
    if len_z == 0: zero_dims += 1
    if len_x > 0 and len_y > 0 and len_z > 0: return 4
    if zero_dims == 1: return 3
    if zero_dims == 2: return 2
    if zero_dims == 3: return 1
    return 0
# --- Numba CUDA Kernel for Batch Validation ---
@cuda.jit
def batch_validation_kernel(candidates_data, colliders_data, num_colliders, batch_results):
    """
    CUDA Kernel to check a batch of candidate cubes against a set of potential colliders.
    Writes 1 to batch_results[i] if candidate i has invalid contact, 0 otherwise.
    """
    cand_idx = cuda.grid(1)
    batch_size = candidates_data.shape[0]
    if cand_idx < batch_size:
        cand_x, cand_y, cand_z, cand_S = candidates_data[cand_idx, 0], candidates_data[cand_idx, 1], candidates_data[cand_idx, 2], candidates_data[cand_idx, 3]
        is_invalid = 0
        for coll_idx in range(num_colliders):
            exist_x, exist_y, exist_z, exist_S = colliders_data[coll_idx, 0], colliders_data[coll_idx, 1], colliders_data[coll_idx, 2], colliders_data[coll_idx, 3]
            contact = get_contact_type_gpu_device(cand_x, cand_y, cand_z, cand_S, exist_x, exist_y, exist_z, exist_S)
            if contact >= 2:
                is_invalid = 1
                break
        batch_results[cand_idx] = is_invalid
# --- CPU Placement Validation (Fallback/S=1 Check/Re-check) ---
def is_placement_valid_cpu(cand_x, cand_y, cand_z, cand_S,
                           bounds_min_x, bounds_min_y, bounds_min_z,
                           bounds_max_x, bounds_max_y, bounds_max_z,
                           current_spatial_grid,
                           current_placed_cubes_list):
    """CPU-only validation check."""
    if cand_S < 1: return False
    # Bounds Check
    has_bounds = bounds_min_x != -9999
    if has_bounds:
        if (cand_x < bounds_min_x or (cand_x + cand_S) > bounds_max_x or
            cand_y < bounds_min_y or (cand_y + cand_S) > bounds_max_y or
            cand_z < bounds_min_z or (cand_z + cand_S) > bounds_max_z):
            return False
    # Grid Query
    potential_colliders_ids = set()
    min_cx, max_cx, min_cy, max_cy, min_cz, max_cz = get_cube_cell_bounds(
        cand_x, cand_y, cand_z, cand_S, GRID_CELL_SIZE
    )
    for cx in range(min_cx - 1, max_cx + 2):
        for cy in range(min_cy - 1, max_cy + 2):
            for cz in range(min_cz - 1, max_cz + 2):
                cell_coords = (cx, cy, cz)
                if cell_coords in current_spatial_grid:
                    potential_colliders_ids.update(current_spatial_grid[cell_coords])
    # Precise Check
    if not potential_colliders_ids: return True
    num_placed = len(current_placed_cubes_list)
    for exist_id in potential_colliders_ids:
        if exist_id < 0 or exist_id >= num_placed: continue
        exist_x, exist_y, exist_z, exist_S, _ = current_placed_cubes_list[exist_id]
        contact = get_contact_type_cpu(cand_x, cand_y, cand_z, cand_S,
                                       exist_x, exist_y, exist_z, exist_S)
        if contact >= 2: return False
    return True
# --- place_cube (Appends to Python list) ---
def place_cube(x, y, z, S, color_name):
    """Adds cube to list, updates grid and vertex info."""
    global placed_cubes, vertex_counts, vertex_info, spatial_grid
    new_cube_id = len(placed_cubes)
    placed_cubes.append((x, y, z, S, color_name))
    min_cx, max_cx, min_cy, max_cy, min_cz, max_cz = get_cube_cell_bounds(x, y, z, S, GRID_CELL_SIZE)
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                spatial_grid[(cx, cy, cz)].append(new_cube_id)
    new_vertices_list = get_cube_vertices_tuple_list(x, y, z, S)
    color_idx = COLOR_INDEX_MAP.get(color_name, DEFAULT_COLOR_INDEX)
    for vertex in new_vertices_list:
        vertex_counts[vertex] += 1
        current_count = vertex_counts[vertex]
        if current_count == 1:
            vertex_info[vertex] = {'parent_id': new_cube_id, 'parent_size': S, 'parent_color_idx': color_idx}
        elif current_count > 1 and vertex in vertex_info:
            try: del vertex_info[vertex]
            except KeyError: pass
    return True # Assume placement itself doesn't fail here
# --- Generate Mesh Data (Uses Python list) ---
def generate_cube_mesh_data(cubes_list):
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
        center = np.array([x + S/2.0, y + S/2.0, z + S/2.0], dtype=np.float32)
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
# --- Time Formatting Helper ---
def format_time(seconds):
    if seconds < 60: return f"{seconds:.2f} seconds"
    else: minutes, secs = divmod(seconds, 60); return f"{int(minutes)} minutes {secs:.2f} seconds"
# ===============================
# --- Main Program Loop ---
# ===============================
# Check for CUDA availability once at the start
try:
    if not cuda.is_available(): raise RuntimeError("CUDA not available")
    cuda.device_array(1) # Test context creation
    print("--- CUDA GPU Detected and Initialized. Validation will use GPU Batching. ---")
    GPU_ENABLED = True
except Exception as e:
    print(f"--- WARNING: CUDA Initialization Failed ({e}). ---")
    print("---          Placement validation will run on CPU only. ---")
    GPU_ENABLED = False
while True:
    # --- Reset State Variables ---
    placed_cubes = []
    spatial_grid = defaultdict(list)
    vertex_counts = defaultdict(int)
    vertex_info = {}
    bounds_min_int = None
    bounds_max_int = None
    GRID_CELL_SIZE = 10 # Default value
    # --- Get User Input (Yaksi's version) ---
    print("\n" + "=" * 84)
    print("Welcome to Marlowe's 3D Fractal Art Generator (VisPy + Numba + Grid + Batched GPU v2)") # Title
    print("This fractal is initially defined as a framework of size N.")
    print("(N = number of layers added after the initial 1x1x1 cube)")
    print("Higher N values increase complexity exponentially.")
    print("Recommended: N=2-9. N=10+ may take significant time.")
    if GPU_ENABLED:
        print(f"Using Spatial Grid (CPU) + S=1 Pre-Filter (CPU) + Batched CUDA Validation (GPU, Batch Size={BATCH_SIZE}) + CPU Re-check")
    else:
        print("Using Spatial Grid (CPU) + S=1 Pre-Filter (CPU) + CPU Validation (No GPU)")
    print("Recommended: GRID_CELL_SIZE=1 or 2 for N=14+ (maybe higher for high N values)")
    print("Enter 'q' to quit.")
    print("=" * 84)
    N_FRAMEWORK = None
    framework_input = ""
    while True:
        framework_input = input("\nEnter value for N or 'q' to quit:").strip().lower()
        if framework_input == 'q': break
        try:
            N_FRAMEWORK_temp = int(framework_input)
            if N_FRAMEWORK_temp >= 0: N_FRAMEWORK = N_FRAMEWORK_temp; print(f"\nUsing N_FRAMEWORK = {N_FRAMEWORK}"); break
            else: print("Error: Please enter a non-negative whole number (0 or greater).")
        except ValueError: print("Error: Invalid input. Please enter a whole number or 'q'.")
    if framework_input == 'q': N_FRAMEWORK = None
    if N_FRAMEWORK is not None:
        while True:
            GRID_input = input("\nEnter value for GRID_CELL_SIZE (e.g. 1 or 2 for N=14+, or 'q' to quit):").strip().lower()
            if GRID_input == 'q': N_FRAMEWORK = None; break
            try:
                GRID_temp = int(GRID_input)
                if GRID_temp > 0: GRID_CELL_SIZE = GRID_temp; print(f"\nUsing GRID_CELL_SIZE = {GRID_CELL_SIZE}."); break
                else: print("Error: Please enter a whole number greater than 0 or 'q'.")
            except ValueError: print("Error: Invalid input. Please enter a whole number greater than 0 or 'q'.")
    if N_FRAMEWORK is not None:
        while True:
            BATCH_SIZE_input = input("\nEnter value for BATCH_SIZE (8055+ for N=13+ or 'q' to quit):").strip().lower()
            if BATCH_SIZE_input == 'q':
                N_FRAMEWORK = None
                break
            try:
                BATCH_SIZE_temp = int(BATCH_SIZE_input)
                if BATCH_SIZE_temp > 0: BATCH_SIZE = BATCH_SIZE_temp; print(f"\nUsing BATCH_SIZE = {BATCH_SIZE}."); break
                else: print("Error: Please enter a whole number greater than 0 or 'q'.")
            except ValueError: print("Error: Invalid input. Please enter a whole number greater than 0 or 'q'.")
    if N_FRAMEWORK is None:
        print("Exiting program. Goodbye!")
        break
    # *** Start Overall Timer ***
    overall_start_time = time.time()
    calculation_end_time = None
    vispy_setup_duration = 0
    # --- Phase 1: Framework Generation ---
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    if not place_cube(0, 0, 0, 1, COLOR_SEQUENCE[0]): print("CRITICAL ERROR: Initiator cube placement failed."); continue
    framework_paths = {}
    if N_FRAMEWORK > 0:
        origin = np.array([0.5, 0.5, 0.5])
        initiator_vertices_list = get_cube_vertices_tuple_list(0, 0, 0, 1)
        for v in initiator_vertices_list:
            direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
            if all(c != 0 for c in direction_vector): framework_paths[direction_vector] = 0
    for n in range(1, N_FRAMEWORK + 1):
        target_size = n + 1
        layer_color_idx = n % len(COLOR_SEQUENCE)
        layer_color_name = COLOR_SEQUENCE[layer_color_idx]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color_name}")
        new_framework_paths = {}
        placed_count_iter = 0
        current_paths = framework_paths.copy()
        for direction_vec, last_cube_id in current_paths.items():
            if not (0 <= last_cube_id < len(placed_cubes)): continue
            lx, ly, lz, lS, _ = placed_cubes[last_cube_id]
            parent_size_phase1 = lS
            last_cube_verts_list = get_cube_vertices_tuple_list(lx, ly, lz, lS)
            if not last_cube_verts_list: continue
            cube_center = np.array([lx + lS / 2.0, ly + lS / 2.0, lz + lS / 2.0])
            try: attachment_vertex = max(last_cube_verts_list, key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError: continue
            vx, vy, vz = attachment_vertex
            placement_result_tuple = get_placement_coords_at_vertex(vx, vy, vz, target_size, lx, ly, lz, parent_size_phase1)
            if placement_result_tuple == (-9999, -9999, -9999): continue
            px, py, pz = placement_result_tuple
            # Use CPU check for framework placement
            if is_placement_valid_cpu(px, py, pz, target_size, -9999, -9999, -9999, -9999, -9999, -9999, spatial_grid, placed_cubes):
                 if place_cube(px, py, pz, target_size, layer_color_name):
                     new_cube_id = len(placed_cubes) - 1
                     new_framework_paths[direction_vec] = new_cube_id
                     placed_count_iter += 1
        framework_paths.update(new_framework_paths)
        print(f"  Placed {placed_count_iter} framework cubes.")
        if placed_count_iter != 8 and n > 0 and len(current_paths) == 8: print(f"WARNING: Placed {placed_count_iter}/8 framework cubes in iteration {n}.")
        elif placed_count_iter == 0 and n > 0: print(f"WARNING: No framework cubes placed in iteration {n}. Stopping framework generation."); break
    duration_phase1 = time.time() - start_time_phase1
    print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---")
    print(f"Total cubes after Phase 1: {len(placed_cubes)}")
    if len(placed_cubes) <= 1 and N_FRAMEWORK > 0: print("Warning: Phase 1 framework generation failed or was incomplete.")
    # --- Calculate Final Bounding Box ---
    print("\n--- Calculating Final Bounding Box ---")
    min_coord_f = np.array([float('inf')] * 3); max_coord_f = np.array([float('-inf')] * 3)
    if not placed_cubes: print("Error: No cubes placed."); continue
    for x, y, z, S, _ in placed_cubes:
        min_coord_f = np.minimum(min_coord_f, [x, y, z])
        max_coord_f = np.maximum(max_coord_f, [x + S, y + S, z + S])
    bounds_min_int = np.floor(min_coord_f).astype(int)
    bounds_max_int = np.ceil(max_coord_f).astype(int)
    print(f"Bounding Box Min (int, BBL inclusive): {bounds_min_int}")
    print(f"Bounding Box Max (int, exclusive): {bounds_max_int}")
    if np.any(bounds_max_int <= bounds_min_int): print("ERROR: Invalid bounding box."); continue
    # --- Phase 2: Color-Based Generational Layering ---
    print(f"\n--- Starting Phase 2: Color-Based Generational Layering ---")
    start_time_phase2 = time.time()
    max_color_cycles = 100
    cycles_without_placement = 0
    current_target_color_idx = 2 # Start with Yellow index
    total_placed_phase2 = 0
    cycle = 0
    b_min_x = bounds_min_int[0]; b_min_y = bounds_min_int[1]; b_min_z = bounds_min_int[2]
    b_max_x = bounds_max_int[0]; b_max_y = bounds_max_int[1]; b_max_z = bounds_max_int[2]
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles:
        cycle += 1
        target_color_idx = current_target_color_idx % len(COLOR_SEQUENCE)
        required_parent_idx = get_required_parent_color_index(target_color_idx)
        target_color_name = COLOR_SEQUENCE[target_color_idx]
        required_parent_name = COLOR_SEQUENCE[required_parent_idx] if required_parent_idx is not None else "None"
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_name} onto {required_parent_name}")
        # 1. Find initial candidates
        parent_vertices_candidates_initial = []
        current_vertex_info_snapshot = vertex_info.copy()
        for V, info in current_vertex_info_snapshot.items():
            if V not in vertex_info: continue
            if info['parent_color_idx'] == required_parent_idx:
                 parent_id = info['parent_id']
                 if 0 <= parent_id < len(placed_cubes):
                     parent_vertices_candidates_initial.append({
                         'vertex': V, 'parent_id': parent_id, 'parent_size': info['parent_size']
                     })
        if not parent_vertices_candidates_initial:
            print(f"  No exposed vertices found for parent color {required_parent_name}.")
            cycles_without_placement += 1; current_target_color_idx += 1; continue
        print(f"  Found {len(parent_vertices_candidates_initial)} initial potential parent vertices ({required_parent_name}).")
        # 1b. Pre-filter by bounds
        parent_vertices_filtered = []
        filter_start_time = time.time()
        vertices_removed_by_filter = 0
        for candidate in parent_vertices_candidates_initial:
            V = candidate['vertex']
            parent_id = candidate['parent_id']
            if not (0 <= parent_id < len(placed_cubes)): continue
            parent_x, parent_y, parent_z, S_parent, _ = placed_cubes[parent_id]
            px1, py1, pz1 = get_placement_coords_at_vertex(V[0], V[1], V[2], 1, parent_x, parent_y, parent_z, S_parent)
            if px1 != -9999:
                if not (px1 < b_min_x or (px1 + 1) > b_max_x or py1 < b_min_y or (py1 + 1) > b_max_y or pz1 < b_min_z or (pz1 + 1) > b_max_z):
                    dist = distance_to_origin(V[0], V[1], V[2])
                    parent_vertices_filtered.append({
                         'vertex': V, 'distance': dist, 'parent_size': candidate['parent_size'],
                         'parent_id': parent_id, 'parent_color_idx': required_parent_idx
                     })
                else: vertices_removed_by_filter += 1
            else: vertices_removed_by_filter += 1
        filter_duration = time.time() - filter_start_time
        print(f"  Filtered down to {len(parent_vertices_filtered)} candidates ({vertices_removed_by_filter} removed by bounds, {filter_duration:.2f}s).")
        if not parent_vertices_filtered:
            print(f"  No candidates remaining after bounds filtering.")
            cycles_without_placement += 1; current_target_color_idx += 1; continue
        # 2. Sort filtered candidates
        parent_vertices_filtered.sort(key=lambda x: x['distance'])
        # 3. Perform S=1 CPU Check
        s1_valid_vertices = {} # Store V -> (px, py, pz) for valid S=1 placements
        vertices_to_batch_or_cpu = [] # List of parent_data dicts for vertices needing S>=2 check
        vertices_removed_permanently = 0
        s1_check_start_time = time.time()
        for parent_data in parent_vertices_filtered:
            V = parent_data['vertex']
            if V not in vertex_info: continue
            parent_id = parent_data['parent_id']
            if not (0 <= parent_id < len(placed_cubes)): continue
            parent_x, parent_y, parent_z, S_parent, parent_color_name = placed_cubes[parent_id]
            if S_parent != parent_data['parent_size'] or COLOR_INDEX_MAP.get(parent_color_name, -1) != parent_data['parent_color_idx']:
                 if V in vertex_info:
                     try: del vertex_info[V]
                     except KeyError: pass
                 continue
            px1, py1, pz1 = get_placement_coords_at_vertex(V[0], V[1], V[2], 1, parent_x, parent_y, parent_z, S_parent)
            if px1 != -9999:
                if is_placement_valid_cpu(px1, py1, pz1, 1, b_min_x, b_min_y, b_min_z, b_max_x, b_max_y, b_max_z, spatial_grid, placed_cubes):
                    s1_valid_vertices[V] = (px1, py1, pz1)
                    if S_parent + 1 >= 2:
                        vertices_to_batch_or_cpu.append(parent_data)
                else: # S=1 failed validation
                    if V in vertex_info:
                        try: del vertex_info[V]; vertices_removed_permanently += 1
                        except KeyError: pass
            else: # Coord calculation failed
                if V in vertex_info:
                    try: del vertex_info[V]; vertices_removed_permanently += 1
                    except KeyError: pass
        s1_check_duration = time.time() - s1_check_start_time
        print(f"  CPU S=1 Pre-check complete ({len(s1_valid_vertices)} pass, {vertices_removed_permanently} removed, {s1_check_duration:.2f}s).")
        # 4. Determine Best S>=2 (GPU Batch or CPU)
        best_s_found_s2_plus = defaultdict(lambda: {'S': 0, 'coords': None}) # Store V -> {S: best_S, coords: (px,py,pz)}
        if GPU_ENABLED and vertices_to_batch_or_cpu:
            # --- GPU Batching Path ---
            print(f"  Accumulating S>=2 candidates for GPU batching...")
            accum_start_time = time.time()
            candidates_for_batching = [] # List of dicts: {'coords', 'S', 'V_tuple'}
            for parent_data in vertices_to_batch_or_cpu:
                V = parent_data['vertex']
                if V not in vertex_info: continue
                parent_id = parent_data['parent_id']
                if not (0 <= parent_id < len(placed_cubes)): continue
                parent_x, parent_y, parent_z, S_parent, _ = placed_cubes[parent_id]
                for S_try in range(S_parent + 1, 1, -1): # S_parent+1 down to 2
                    px_py_pz_tuple = get_placement_coords_at_vertex(V[0], V[1], V[2], S_try, parent_x, parent_y, parent_z, S_parent)
                    if px_py_pz_tuple != (-9999, -9999, -9999):
                        candidates_for_batching.append({
                            'coords': px_py_pz_tuple, 'S': S_try, 'V_tuple': V
                        })
            accum_duration = time.time() - accum_start_time
            print(f"    Accumulated {len(candidates_for_batching)} S>=2 candidates ({accum_duration:.2f}s).")
            num_candidates_total = len(candidates_for_batching)
            processed_count = 0
            batch_num = 0
            while processed_count < num_candidates_total:
                batch_num += 1
                batch_start_idx = processed_count
                batch_end_idx = min(processed_count + BATCH_SIZE, num_candidates_total)
                current_batch = candidates_for_batching[batch_start_idx:batch_end_idx]
                current_batch_size = len(current_batch)
                processed_count += current_batch_size
                print(f"    Processing Batch {batch_num} ({current_batch_size} candidates)...")
                batch_proc_start = time.time()
                batch_collider_ids = set()
                for cand_info in current_batch:
                    cx, cy, cz, cS = cand_info['coords'][0], cand_info['coords'][1], cand_info['coords'][2], cand_info['S']
                    min_cx, max_cx, min_cy, max_cy, min_cz, max_cz = get_cube_cell_bounds(cx, cy, cz, cS, GRID_CELL_SIZE)
                    for cx_g in range(min_cx - 1, max_cx + 2):
                        for cy_g in range(min_cy - 1, max_cy + 2):
                            for cz_g in range(min_cz - 1, max_cz + 2):
                                if (cx_g, cy_g, cz_g) in spatial_grid:
                                    batch_collider_ids.update(spatial_grid[(cx_g, cy_g, cz_g)])
                num_colliders = len(batch_collider_ids)
                if num_colliders == 0:
                    batch_results_np = np.zeros(current_batch_size, dtype=np.int32)
                else:
                    try:
                        candidates_data_np = np.array([[c['coords'][0], c['coords'][1], c['coords'][2], c['S']] for c in current_batch], dtype=np.int64)
                        colliders_data_np = np.empty((num_colliders, 4), dtype=np.int64)
                        valid_coll_count = 0
                        current_list_len = len(placed_cubes)
                        for exist_id in batch_collider_ids:
                            if 0 <= exist_id < current_list_len:
                                exist_x, exist_y, exist_z, exist_S, _ = placed_cubes[exist_id]
                                colliders_data_np[valid_coll_count] = [exist_x, exist_y, exist_z, exist_S]
                                valid_coll_count += 1
                        if valid_coll_count == 0:
                             batch_results_np = np.zeros(current_batch_size, dtype=np.int32)
                        else:
                            if valid_coll_count < num_colliders:
                                colliders_data_np = colliders_data_np[:valid_coll_count]
                                num_colliders = valid_coll_count
                            batch_results_np = np.zeros(current_batch_size, dtype=np.int32)
                            d_candidates_data = cuda.to_device(candidates_data_np)
                            d_colliders_data = cuda.to_device(colliders_data_np)
                            d_batch_results = cuda.to_device(batch_results_np)
                            threads_per_block = 128
                            blocks_per_grid = (current_batch_size + (threads_per_block - 1)) // threads_per_block
                            batch_validation_kernel[blocks_per_grid, threads_per_block](
                                d_candidates_data, d_colliders_data, num_colliders, d_batch_results
                            )
                            batch_results_np = d_batch_results.copy_to_host()
                            del d_candidates_data, d_colliders_data, d_batch_results
                            cuda.current_context().synchronize()
                    except Exception as e:
                        print(f"\n!!! CUDA Error during batch {batch_num}: {e}")
                        print("!!! Aborting GPU processing for this cycle, using CPU fallback.")
                        GPU_ENABLED = False # Fallback for the rest of the run
                        best_s_found_s2_plus.clear() # Clear GPU results
                        # Re-run S>=2 checks on CPU immediately
                        for parent_data_cpu in vertices_to_batch_or_cpu:
                             V_cpu = parent_data_cpu['vertex']
                             if V_cpu not in vertex_info: continue
                             parent_id_cpu = parent_data_cpu['parent_id']
                             if not (0 <= parent_id_cpu < len(placed_cubes)): continue
                             parent_x_cpu, parent_y_cpu, parent_z_cpu, S_parent_cpu, _ = placed_cubes[parent_id_cpu]
                             for S_try_cpu in range(S_parent_cpu + 1, 1, -1):
                                 px_py_pz_tuple_cpu = get_placement_coords_at_vertex(V_cpu[0], V_cpu[1], V_cpu[2], S_try_cpu, parent_x_cpu, parent_y_cpu, parent_z_cpu, S_parent_cpu)
                                 if px_py_pz_tuple_cpu != (-9999, -9999, -9999):
                                     if is_placement_valid_cpu(px_py_pz_tuple_cpu[0], px_py_pz_tuple_cpu[1], px_py_pz_tuple_cpu[2], S_try_cpu, b_min_x, b_min_y, b_min_z, b_max_x, b_max_y, b_max_z, spatial_grid, placed_cubes):
                                         if S_try_cpu > best_s_found_s2_plus[V_cpu]['S']:
                                             best_s_found_s2_plus[V_cpu]['S'] = S_try_cpu
                                             best_s_found_s2_plus[V_cpu]['coords'] = px_py_pz_tuple_cpu
                                         break # Found largest for this vertex
                        break # Exit batch processing loop as we've done CPU checks now
                # Process results for this batch (if GPU didn't fail)
                if GPU_ENABLED:
                    for i in range(current_batch_size):
                        if batch_results_np[i] == 0: # Candidate i is valid
                            cand_info = current_batch[i]
                            V = cand_info['V_tuple']
                            S_try = cand_info['S']
                            # Store the largest valid S found so far for this vertex
                            if S_try > best_s_found_s2_plus[V]['S']:
                                 best_s_found_s2_plus[V]['S'] = S_try
                                 best_s_found_s2_plus[V]['coords'] = cand_info['coords']
                    batch_proc_duration = time.time() - batch_proc_start
                    print(f"      Batch {batch_num} processed ({num_colliders} colliders checked, {batch_proc_duration:.2f}s).")
            # End of batch processing loop
            if vertices_to_batch_or_cpu and GPU_ENABLED: print(f"  GPU Batching complete.")
        elif vertices_to_batch_or_cpu:
             # --- CPU Path for S >= 2 ---
             print(f"  Performing S>=2 checks on CPU...")
             cpu_s2_start = time.time()
             for parent_data in vertices_to_batch_or_cpu:
                 V = parent_data['vertex']
                 if V not in vertex_info: continue
                 parent_id = parent_data['parent_id']
                 if not (0 <= parent_id < len(placed_cubes)): continue
                 parent_x, parent_y, parent_z, S_parent, _ = placed_cubes[parent_id]
                 for S_try in range(S_parent + 1, 1, -1): # S_parent+1 down to 2
                     px_py_pz_tuple = get_placement_coords_at_vertex(V[0], V[1], V[2], S_try, parent_x, parent_y, parent_z, S_parent)
                     if px_py_pz_tuple != (-9999, -9999, -9999):
                         if is_placement_valid_cpu(px_py_pz_tuple[0], px_py_pz_tuple[1], px_py_pz_tuple[2], S_try, b_min_x, b_min_y, b_min_z, b_max_x, b_max_y, b_max_z, spatial_grid, placed_cubes):
                             # Found largest valid S>=2 for this vertex
                             best_s_found_s2_plus[V]['S'] = S_try
                             best_s_found_s2_plus[V]['coords'] = px_py_pz_tuple
                             break # Move to next vertex
             cpu_s2_duration = time.time() - cpu_s2_start
             print(f"    CPU S>=2 checks complete ({cpu_s2_duration:.2f}s).")
        # 5. Final Sequential Placement with Re-validation
        placement_start_time = time.time()
        placed_this_color_cycle = 0
        processed_vertices_this_cycle = set() # Track vertices processed in this final loop
        # Iterate through the original sorted list to maintain placement order
        for parent_data in parent_vertices_filtered:
            V = parent_data['vertex']
            # Skip if already processed or removed
            if V in processed_vertices_this_cycle or V not in vertex_info: continue
            # Skip if it didn't even pass S=1
            if V not in s1_valid_vertices: continue
            # Determine best potential placement based on S=1 and S>=2 checks
            s1_coords = s1_valid_vertices[V]
            s2_plus_result = best_s_found_s2_plus.get(V, {'S': 0, 'coords': None})
            best_S_final = 0
            coords_final = None
            if s2_plus_result['S'] > 0: # Found a valid S>=2 placement
                best_S_final = s2_plus_result['S']
                coords_final = s2_plus_result['coords']
            else: # Only S=1 was potentially valid
                best_S_final = 1
                coords_final = s1_coords
            # *** Perform final CPU re-validation ***
            if coords_final and is_placement_valid_cpu(coords_final[0], coords_final[1], coords_final[2], best_S_final, b_min_x, b_min_y, b_min_z, b_max_x, b_max_y, b_max_z, spatial_grid, placed_cubes):
                # Place the cube
                if place_cube(coords_final[0], coords_final[1], coords_final[2], best_S_final, target_color_name):
                    placed_this_color_cycle += 1
                    total_placed_phase2 += 1
            # Mark vertex as processed in this final stage, regardless of placement success after re-check
            processed_vertices_this_cycle.add(V)
        placement_duration = time.time() - placement_start_time
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_name} after re-validation ({placement_duration:.2f}s).")
        # Update termination logic
        if placed_this_color_cycle > 0: cycles_without_placement = 0
        else: cycles_without_placement += 1
        current_target_color_idx += 1 # Move to next color index
    # --- End Phase 2 Loop ---
    duration_phase2 = time.time() - start_time_phase2
    print(f"--- Phase 2 Complete ({duration_phase2:.2f}s in {cycle} color cycles) ---")
    print(f"Total cubes placed in Phase 2: {total_placed_phase2}")
    print(f"Total cubes after Phase 2: {len(placed_cubes)}")
    print(f"Total unique vertices created (ever): {len(vertex_counts)}")
    print(f"Final exposed vertices map size: {len(vertex_info)}")
    # *** Capture Calculation End Time ***
    calculation_end_time = time.time()
    calculation_duration = calculation_end_time - overall_start_time
    print(f"\n--- Calculation Complete ({format_time(calculation_duration)}) ---")
    # --- VisPy Visualization ---
    print("\n--- Preparing VisPy Mesh Visualization ---")
    vispy_start_time = time.time()
    vispy_setup_duration = 0
    if not placed_cubes:
        print("No cubes were placed.")
    else:
        vertices, faces, vertex_colors = generate_cube_mesh_data(placed_cubes)
        if vertices is None or faces is None or vertex_colors is None:
            print("Mesh data generation failed.")
        else:
            print("Creating VisPy scene...")
            canvas = scene.SceneCanvas(keys='interactive', show=True, title=f"3D Fractal N={N_FRAMEWORK} (Mesh + Grid + Batched GPU v2)")
            view = canvas.central_widget.add_view(); view.bgcolor = '#222222'
            print("Adding mesh visual...")
            mesh = visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, shading='flat')
            view.add(mesh)
            view.camera = scene.TurntableCamera(fov=60, elevation=30, azimuth=-45)
            if bounds_min_int is not None and bounds_max_int is not None and np.any(bounds_max_int > bounds_min_int):
                center_point = (bounds_min_int + bounds_max_int) / 2.0
                diag_vector = bounds_max_int - bounds_min_int
                diag_vector[diag_vector <= 0] = 1.0
                diag_length = np.linalg.norm(diag_vector)
                view.camera.distance = max(diag_length * 1.5, 10)
                view.camera.center = tuple(center_point)
            else:
                 first_cube = placed_cubes[0]
                 view.camera.distance = 10 * first_cube[3]
                 view.camera.center = (first_cube[0]+first_cube[3]/2, first_cube[1]+first_cube[3]/2, first_cube[2]+first_cube[3]/2)
            axis = visuals.XYZAxis(parent=view.scene)
            vispy_setup_end_time = time.time()
            vispy_setup_duration = vispy_setup_end_time - vispy_start_time
            print(f"VisPy setup complete ({vispy_setup_duration:.2f}s).")
            print("\nStarting VisPy visualization...")
            print("Rotate: Left Mouse Button + Drag | Zoom: Scroll Wheel | Pan: Shift + Left Button + Drag")
            print("Close the VisPy window to generate another fractal or enter 'q'.")
            app.run()
            print("VisPy window closed.")
    # *** Report Total Processing Time (including VisPy setup) ***
    if calculation_end_time:
        total_processing_duration = calculation_duration + vispy_setup_duration
        print(f"\n--- Total Processing Time for N={N_FRAMEWORK}, GRID_CELL_SIZE={GRID_CELL_SIZE}, BATCH_SIZE={BATCH_SIZE} : {format_time(total_processing_duration)} ---")
        print(f"    (Calculation Time: {format_time(calculation_duration)})")
        print(f"    (VisPy Setup Time: {format_time(vispy_setup_duration)})")
    else:
        partial_duration = time.time() - overall_start_time
        print(f"\n--- Partial Time Elapsed: {format_time(partial_duration)} ---")
print("--- Script Finished ---")
