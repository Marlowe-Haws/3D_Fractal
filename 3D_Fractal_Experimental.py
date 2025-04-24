# -*- coding: utf-8 -*-
import numpy as np
import numba
# from numba.typed import Dict # No longer needed
from vispy import scene, app, color
from vispy.scene import visuals
from collections import defaultdict
import math
import time
# import itertools # No longer needed
# --- Configuration ---
GRID_CELL_SIZE = 1 # Tuneable
# --- Color Definitions ---
COLOR_SEQUENCE = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
COLOR_MAP_PLOT = {
    'red': '#FF0000', 'orange': '#FFA500', 'yellow': '#FFFF00',
    'green': '#00FF00', 'blue': '#0000FF', 'purple': '#800080',
    'default': '#555555'
}
# --- Data Structures ---
placed_cubes = []
spatial_grid = defaultdict(list)
vertex_counts = defaultdict(int)
vertex_info = {}
bounds_min_int = None
bounds_max_int = None
# --- Helper Functions ---
def get_next_color(current_color):
    try: idx = COLOR_SEQUENCE.index(current_color); return COLOR_SEQUENCE[(idx + 1) % len(COLOR_SEQUENCE)]
    except ValueError: return COLOR_SEQUENCE[0]
def get_required_parent_color(target_placement_color):
    try: idx = COLOR_SEQUENCE.index(target_placement_color); parent_idx = (idx - 1 + len(COLOR_SEQUENCE)) % len(COLOR_SEQUENCE); return COLOR_SEQUENCE[parent_idx]
    except ValueError: return None
# --- Numba Accelerated Helper Functions ---
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
def get_contact_type(x1, y1, z1, S1, x2, y2, z2, S2):
    # (Implementation remains the same)
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
    # (Implementation remains the same)
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
    # (Implementation remains the same)
    S_eff = max(1, S)
    min_cx = math.floor(x / cell_size)
    max_cx = math.floor((x + S_eff - 1) / cell_size)
    min_cy = math.floor(y / cell_size)
    max_cy = math.floor((y + S_eff - 1) / cell_size)
    min_cz = math.floor(z / cell_size)
    max_cz = math.floor((z + S_eff - 1) / cell_size)
    return min_cx, max_cx, min_cy, max_cy, min_cz, max_cz
# --- Placement Validation (No AABB Check) ---
def is_placement_valid(cand_x, cand_y, cand_z, cand_S,
                       bounds_min_x, bounds_min_y, bounds_min_z,
                       bounds_max_x, bounds_max_y, bounds_max_z,
                       current_spatial_grid, # Pass the grid
                       placed_cubes_list):   # Pass the list
    """
    Checks if placing a candidate cube is valid using optimized spatial grid search
    and ONLY the precise contact check (get_contact_type).
    """
    if cand_S < 1: return False
    # 1. Bounds Check
    has_bounds = bounds_min_x != -9999
    if has_bounds:
        if (cand_x < bounds_min_x or (cand_x + cand_S) > bounds_max_x or
            cand_y < bounds_min_y or (cand_y + cand_S) > bounds_max_y or
            cand_z < bounds_min_z or (cand_z + cand_S) > bounds_max_z):
            return False
    # 2. Check against nearby cubes using the spatial grid
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
    # 3. Perform checks only against potential colliders
    if not potential_colliders_ids:
        return True
    for exist_id in potential_colliders_ids:
        if exist_id < 0 or exist_id >= len(placed_cubes_list): continue
        exist_x, exist_y, exist_z, exist_S, _ = placed_cubes_list[exist_id]
        # Always perform the precise check now
        contact = get_contact_type(cand_x, cand_y, cand_z, cand_S,
                                   exist_x, exist_y, exist_z, exist_S)
        # Check for forbidden contact types (Edge, Face, Volume Overlap)
        if contact >= 2:
            return False # Invalid placement
    # If loop completes without finding invalid contact, placement is valid
    return True
# --- place_cube (Handles vertex_info updates) ---
def place_cube(x, y, z, S, color):
    global placed_cubes, vertex_counts, vertex_info, spatial_grid
    global bounds_min_int, bounds_max_int
    if S < 1: return False
    b_min_x = bounds_min_int[0] if bounds_min_int is not None else -9999
    b_min_y = bounds_min_int[1] if bounds_min_int is not None else -9999
    b_min_z = bounds_min_int[2] if bounds_min_int is not None else -9999
    b_max_x = bounds_max_int[0] if bounds_max_int is not None else -9999
    b_max_y = bounds_max_int[1] if bounds_max_int is not None else -9999
    b_max_z = bounds_max_int[2] if bounds_max_int is not None else -9999
    if not is_placement_valid(x, y, z, S,
                              b_min_x, b_min_y, b_min_z,
                              b_max_x, b_max_y, b_max_z,
                              spatial_grid, placed_cubes):
        return False
    new_cube_id = len(placed_cubes)
    placed_cubes.append((x, y, z, S, color))
    min_cx, max_cx, min_cy, max_cy, min_cz, max_cz = get_cube_cell_bounds(
        x, y, z, S, GRID_CELL_SIZE
    )
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                spatial_grid[(cx, cy, cz)].append(new_cube_id)
    new_vertices_list = get_cube_vertices_tuple_list(x, y, z, S)
    for vertex in new_vertices_list:
        vertex_counts[vertex] += 1
        current_count = vertex_counts[vertex]
        if current_count == 1:
            # This vertex is newly created by this cube
            vertex_info[vertex] = {
                'parent_id': new_cube_id,
                'parent_size': S,
                'parent_color': color
            }
        elif current_count > 1 and vertex in vertex_info:
            # This vertex was previously exposed (count was 1) and is now covered
            try:
                del vertex_info[vertex]
            except KeyError:
                pass # Should not happen if logic is correct, but safe to include
    return True
# --- Generate Mesh Data (Unchanged) ---
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
# ===============================
# --- Main Program Loop ---
# ===============================
while True:
    # --- Reset State Variables ---
    placed_cubes = []
    spatial_grid = defaultdict(list)
    vertex_counts = defaultdict(int)
    vertex_info = {}
    bounds_min_int = None
    bounds_max_int = None
    # --- Get User Input ---
    print("\n" + "=" * 84)
    print("Welcome to Marlowe's 3D Fractal Art Generator (VisPy + Numba + Grid + Pre-Filtering)") # Title
    print("This fractal is initially defined as a framework of size N.")
    print("(N = number of layers added after the initial 1x1x1 cube)")
    print("Higher N values increase complexity exponentially.")
    print("Recommended: N=2-9. N=10+ may take significant time.")
    print("Using Spatial Grid + S=1 Pre-Filter Optimization") # Indicate optimizations
    print("Recommended: GRID_CELL_SIZE=1 or 2 for N=14+ (maybe higher for high N values)")
    print("Enter 'q' to quit.")
    print("=" * 84)
    N_FRAMEWORK = None
    while True:
        framework_input = input("\nEnter value for N or 'q' to quit:").strip().lower()
        if framework_input == 'q': break
        try:
            N_FRAMEWORK_temp = int(framework_input)
            if N_FRAMEWORK_temp >= 0: N_FRAMEWORK = N_FRAMEWORK_temp; print(f"\nUsing N_FRAMEWORK = {N_FRAMEWORK}"); break
            else: print("Error: Please enter a non-negative whole number (0 or greater).")
        except ValueError: print("Error: Invalid input. Please enter a whole number or 'q'.")
    while True:
        if framework_input == 'q': break
        GRID_input = input("\nEnter value for GRID_CELL_SIZE (e.g. 1 or 2 for N=14+, or 'q' to quit):").strip().lower()
        if GRID_input == 'q':
            N_FRAMEWORK = None
            break
        try:
            GRID_temp = int(GRID_input)
            if GRID_temp > 0: GRID_CELL_SIZE = GRID_temp; print(f"\nUsing GRID_CELL_SIZE = {GRID_CELL_SIZE}."); break
            else: print("Error: Please enter a whole number greater than 0 or 'q'.")
        except ValueError: print("Error: Invalid input. Please enter a whole number greater than 0 or 'q'.")
    if N_FRAMEWORK is None: print("Exiting program. Goodbye!"); break
    # *** Start Overall Timer ***
    overall_start_time = time.time()
    calculation_end_time = None # Initialize
    # --- Phase 1: Framework Generation ---
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    if not place_cube(0, 0, 0, 1, COLOR_SEQUENCE[0]):
        print("CRITICAL ERROR: Initiator cube placement failed."); continue
    framework_paths = {}
    if N_FRAMEWORK > 0:
        origin = np.array([0.5, 0.5, 0.5])
        initiator_vertices_list = get_cube_vertices_tuple_list(0, 0, 0, 1)
        for v in initiator_vertices_list:
            direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
            if all(c != 0 for c in direction_vector): framework_paths[direction_vector] = 0
    for n in range(1, N_FRAMEWORK + 1):
        target_size = n + 1
        layer_color = COLOR_SEQUENCE[n % len(COLOR_SEQUENCE)]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color}")
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
            try:
                attachment_vertex = max(last_cube_verts_list,
                                        key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError: continue
            vx, vy, vz = attachment_vertex
            placement_result_tuple = get_placement_coords_at_vertex(
                vx, vy, vz, target_size, lx, ly, lz, parent_size_phase1
            )
            if placement_result_tuple == (-9999, -9999, -9999): continue
            px, py, pz = placement_result_tuple
            if place_cube(px, py, pz, target_size, layer_color):
                new_cube_id = len(placed_cubes) - 1
                new_framework_paths[direction_vec] = new_cube_id
                placed_count_iter += 1
        framework_paths.update(new_framework_paths)
        print(f"  Placed {placed_count_iter} framework cubes.")
        if placed_count_iter != 8 and n > 0 and len(current_paths) == 8:
             print(f"WARNING: Placed {placed_count_iter}/8 framework cubes in iteration {n}.")
        elif placed_count_iter == 0 and n > 0:
             print(f"WARNING: No framework cubes placed in iteration {n}. Stopping framework generation.")
             break
    duration_phase1 = time.time() - start_time_phase1
    print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---")
    print(f"Total cubes after Phase 1: {len(placed_cubes)}")
    if len(placed_cubes) <= 1 and N_FRAMEWORK > 0:
        print("Warning: Phase 1 framework generation failed or was incomplete.")
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
    current_target_color_index = 2 # Start with Yellow
    total_placed_phase2 = 0
    cycle = 0
    # Pre-calculate bounds components once
    b_min_x = bounds_min_int[0]; b_min_y = bounds_min_int[1]; b_min_z = bounds_min_int[2]
    b_max_x = bounds_max_int[0]; b_max_y = bounds_max_int[1]; b_max_z = bounds_max_int[2]
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles:
        cycle += 1
        target_color_to_place = COLOR_SEQUENCE[current_target_color_index % len(COLOR_SEQUENCE)]
        required_parent_color = get_required_parent_color(target_color_to_place)
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_to_place} onto {required_parent_color}")
        # 1. Find initial candidates
        parent_vertices_candidates_initial = []
        current_vertex_info_snapshot = vertex_info.copy()
        for V, info in current_vertex_info_snapshot.items():
            if V not in vertex_info: continue # Check if removed during this cycle
            if info['parent_color'] == required_parent_color:
                 if 0 <= info['parent_id'] < len(placed_cubes):
                     if placed_cubes[info['parent_id']][4] == required_parent_color:
                         parent_vertices_candidates_initial.append({
                             'vertex': V, 'parent_id': info['parent_id'], 'parent_size': info['parent_size']
                         })
        if not parent_vertices_candidates_initial:
            print(f"  No exposed vertices found for parent color {required_parent_color}.")
            cycles_without_placement += 1; current_target_color_index += 1; continue
        print(f"  Found {len(parent_vertices_candidates_initial)} initial potential parent vertices ({required_parent_color}).")
        # 1b. Pre-filter by bounds (using 1x1x1 check)
        parent_vertices_filtered = []
        filter_start_time = time.time()
        vertices_removed_by_filter = 0
        for candidate in parent_vertices_candidates_initial:
            V = candidate['vertex']
            parent_id = candidate['parent_id']
            if not (0 <= parent_id < len(placed_cubes)): continue
            parent_x, parent_y, parent_z, S_parent, _ = placed_cubes[parent_id]
            px1, py1, pz1 = get_placement_coords_at_vertex(
                V[0], V[1], V[2], 1, parent_x, parent_y, parent_z, S_parent
            )
            if px1 != -9999:
                if not (px1 < b_min_x or (px1 + 1) > b_max_x or
                        py1 < b_min_y or (py1 + 1) > b_max_y or
                        pz1 < b_min_z or (pz1 + 1) > b_max_z):
                    dist = distance_to_origin(V[0], V[1], V[2])
                    parent_vertices_filtered.append({
                         'vertex': V, 'distance': dist,
                         'parent_size': candidate['parent_size'], 'parent_id': candidate['parent_id']
                     })
                else:
                    vertices_removed_by_filter += 1
            else:
                 vertices_removed_by_filter += 1
        filter_duration = time.time() - filter_start_time
        print(f"  Filtered down to {len(parent_vertices_filtered)} candidates ({vertices_removed_by_filter} removed by bounds, {filter_duration:.2f}s).")
        if not parent_vertices_filtered:
            print(f"  No candidates remaining after bounds filtering.")
            cycles_without_placement += 1; current_target_color_index += 1; continue
        # 2. Sort filtered candidates
        parent_vertices_filtered.sort(key=lambda x: x['distance'])
        # 3. Attempt placement with S=1 optimization
        placed_this_color_cycle = 0
        processed_vertices_this_cycle = set()
        vertices_removed_permanently = 0
        for parent_data in parent_vertices_filtered:
            V = parent_data['vertex']
            # Skip if processed in this cycle OR if it was permanently removed earlier in this loop
            if V in processed_vertices_this_cycle or V not in vertex_info: continue
            parent_id = parent_data['parent_id']
            if not (0 <= parent_id < len(placed_cubes)): continue # Stale ID check
            parent_x, parent_y, parent_z, S_parent, p_color = placed_cubes[parent_id]
            # Data consistency check
            if S_parent != parent_data['parent_size'] or p_color != required_parent_color:
                 if V in vertex_info:
                     try: del vertex_info[V]
                     except KeyError: pass
                 continue
            # *** OPTIMIZATION: Check S=1 First ***
            best_S_found = 0
            placement_coords = None
            can_place_s1 = False
            # Calculate coords for S=1
            px1, py1, pz1 = get_placement_coords_at_vertex(
                V[0], V[1], V[2], 1, parent_x, parent_y, parent_z, S_parent
            )
            if px1 != -9999: # Check if coord calculation is valid
                # Check validity for S=1
                if is_placement_valid(px1, py1, pz1, 1,
                                      b_min_x, b_min_y, b_min_z,
                                      b_max_x, b_max_y, b_max_z,
                                      spatial_grid, placed_cubes):
                    # S=1 is possible! Store it as the fallback.
                    can_place_s1 = True
                    best_S_found = 1
                    placement_coords = (px1, py1, pz1)
                else:
                    # S=1 is NOT possible. This vertex is permanently blocked.
                    if V in vertex_info:
                        try:
                            del vertex_info[V]
                            vertices_removed_permanently += 1
                        except KeyError:
                            pass # Already removed
                    # Skip to the next vertex candidate
                    continue
            else:
                # Coordinate calculation failed for S=1, treat as blocked
                if V in vertex_info:
                    try:
                        del vertex_info[V]
                        vertices_removed_permanently += 1
                    except KeyError:
                        pass
                continue
            # *** If S=1 was possible, check larger sizes downwards ***
            if can_place_s1:
                target_size_start = S_parent + 1
                # Check S_parent+1 down to 2
                for S_try in range(target_size_start, 1, -1):
                    px_py_pz_tuple = get_placement_coords_at_vertex(
                        V[0], V[1], V[2], S_try, parent_x, parent_y, parent_z, S_parent
                    )
                    if px_py_pz_tuple == (-9999, -9999, -9999): continue
                    px, py, pz = px_py_pz_tuple
                    if is_placement_valid(px, py, pz, S_try,
                                          b_min_x, b_min_y, b_min_z,
                                          b_max_x, b_max_y, b_max_z,
                                          spatial_grid, placed_cubes):
                        # Found a larger valid size! Update and break.
                        best_S_found = S_try
                        placement_coords = (px, py, pz)
                        break # Stop checking smaller sizes
            # *** Place the cube using best_S_found (which is >= 1 if we got here) ***
            # Need to ensure placement_coords is not None (should always be set if best_S_found > 0)
            if best_S_found > 0 and placement_coords is not None:
                if place_cube(placement_coords[0], placement_coords[1], placement_coords[2],
                              best_S_found, target_color_to_place):
                    placed_this_color_cycle += 1
                    total_placed_phase2 += 1
                    # Mark vertex as processed for *this cycle*
                    processed_vertices_this_cycle.add(V)
                # else: # Should not fail if is_placement_valid passed
                    # print(f"Error: place_cube failed unexpectedly after check V={V}, S={best_S_found}")
            # else: # Should not happen if S=1 check passed
                # print(f"Warning: No valid placement found for V={V} despite S=1 passing.")
                # pass
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_to_place}.")
        if vertices_removed_permanently > 0:
            print(f"  ({vertices_removed_permanently} vertices permanently removed as unusable for this color.)")
        # Update termination logic
        if placed_this_color_cycle > 0: cycles_without_placement = 0
        else: cycles_without_placement += 1
        current_target_color_index += 1 # Move to next color
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
    print(f"\n--- Calculation Complete ({calculation_duration:.2f}s) ---")
    # --- VisPy Visualization ---
    print("\n--- Preparing VisPy Mesh Visualization ---")
    vispy_start_time = time.time()
    if not placed_cubes:
        print("No cubes were placed.")
    else:
        vertices, faces, vertex_colors = generate_cube_mesh_data(placed_cubes)
        if vertices is None or faces is None or vertex_colors is None:
            print("Mesh data generation failed.")
        else:
            print("Creating VisPy scene...")
            canvas = scene.SceneCanvas(keys='interactive', show=True, title=f"3D Fractal N={N_FRAMEWORK} (Mesh + Grid + S1 Opt)")
            view = canvas.central_widget.add_view()
            view.bgcolor = '#222222'
            print("Adding mesh visual...")
            mesh = visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, shading='flat')
            view.add(mesh)
            view.camera = scene.TurntableCamera(fov=60, elevation=30, azimuth=-45)
            if bounds_min_int is not None and bounds_max_int is not None and np.any(bounds_max_int > bounds_min_int):
                center_point = (bounds_min_int + bounds_max_int) / 2.0
                diag_vector = bounds_max_int - bounds_min_int
                diag_vector[diag_vector <= 0] = 1.0 # Avoid zero length
                diag_length = np.linalg.norm(diag_vector)
                view.camera.distance = max(diag_length * 1.5, 10) # Ensure minimum distance
                view.camera.center = tuple(center_point)
            else:
                 # Fallback if bounds are invalid or only one cube exists
                 first_cube = placed_cubes[0]
                 view.camera.distance = 10 * first_cube[3] # Scale distance by initial cube size
                 view.camera.center = (first_cube[0]+first_cube[3]/2, first_cube[1]+first_cube[3]/2, first_cube[2]+first_cube[3]/2)
            axis = visuals.XYZAxis(parent=view.scene)
            # *** Capture End Time BEFORE showing window ***
            vispy_setup_end_time = time.time()
            vispy_setup_duration = vispy_setup_end_time - vispy_start_time
            print(f"VisPy setup complete ({vispy_setup_duration:.2f}s).")
            print("\nStarting VisPy visualization...")
            print("Rotate: Left Mouse Button + Drag | Zoom: Scroll Wheel | Pan: Shift + Left Button + Drag")
            print("Close the VisPy window to generate another fractal or enter 'q'.")
            app.run() # This blocks until the window is closed
            print("VisPy window closed.")
    # *** Report Total Calculation Time ***
    calculation_duration += vispy_setup_duration
    if calculation_end_time: # Check if calculation actually finished
        print(f"\n--- Total Calculation Time for N_Framework:{N_FRAMEWORK}, GRID_CELL_SIZE:{GRID_CELL_SIZE} : {calculation_duration:.2f} seconds ---")
    else:
        # If calculation didn't finish (e.g., error, quit early)
        partial_duration = time.time() - overall_start_time
        print(f"\n--- Partial Time Elapsed: {partial_duration:.2f} seconds ---")
print("--- Script Finished ---")
