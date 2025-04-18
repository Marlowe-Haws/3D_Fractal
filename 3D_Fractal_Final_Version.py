import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import itertools
import math
import time
# --- Color Definitions ---
COLOR_SEQUENCE = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
COLOR_MAP_PLOT = {
    'red': '#FF0000',
    'orange': '#FFA500',
    'yellow': '#FFFF00',
    'green': '#00FF00', # Brighter green
    'blue': '#0000FF',
    'purple': '#800080',
    'default': '#555555' # For errors or uncolored
}
# --- Data Structures ---
placed_cubes = []  # List to store (x, y, z, S, color) tuples
occupied_blocks = set() # Set of (ix, iy, iz) coordinates of occupied 1x1x1 blocks
vertex_counts = defaultdict(int) # Map of (vx, vy, vz) vertex coordinate -> count
vertex_info = {} # Maps vertex_coord -> {'parent_id': int, 'parent_size': int, 'parent_color': str}
# Bounding Box Variables (will be set after Phase 1)
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
def check_placement(x, y, z, S, current_occupied_blocks, current_vertex_counts,
                  bounds_min=None, bounds_max=None):
    """
    Checks if placing a cube is valid GEOMETRICALLY (bounds, NO face or edge contact
    with ANY cube, volume collision, vertex rule). Allows only corner contact.
    """
    if S < 1:
        return False
    # 1. Check against Bounds (ONLY if bounds are provided for Phase 2)
    if bounds_min is not None and bounds_max is not None:
        if (x < bounds_min[0] or (x + S) > bounds_max[0] or
            y < bounds_min[1] or (y + S) > bounds_max[1] or
            z < bounds_min[2] or (z + S) > bounds_max[2]):
            return False
    # --- Check for FACE contact ---
    for face_dim in range(3): # 0:X, 1:Y, 2:Z
        for face_dir in [-1, 1]: # -1: Neg face, +1: Pos face
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
                if neighbor_block in current_occupied_blocks:
                    # print(f"Face Collision: Neighbor {neighbor_block} for cube ({x},{y},{z}, S={S})")
                    return False # Invalid placement (Face Contact)
    # --- Check for EDGE contact ---
    # Iterate through the 12 edge directions relative to the cube's BBL corner (x,y,z)
    # Example: Edge along Z axis at (x-1, y-1) -> check blocks (x-1, y-1, z+k)
    for edge_dim1 in range(3): # Dimension fixed for the edge offset (e.g., X fixed at x-1)
        for edge_dir1 in [-1, S]: # Offset direction (-1 or +S)
            coord1 = (x, y, z)[edge_dim1] + edge_dir1 # e.g., x-1 or x+S
            for edge_dim2 in range(edge_dim1 + 1, 3): # Second dimension fixed (e.g., Y fixed at y-1)
                 for edge_dir2 in [-1, S]: # Offset direction (-1 or +S)
                    coord2 = (x, y, z)[edge_dim2] + edge_dir2 # e.g., y-1 or y+S
                    # The third dimension iterates along the edge length
                    edge_iter_dim = 3 - edge_dim1 - edge_dim2 # The remaining dimension (0, 1, or 2)
                    base_coord3 = (x, y, z)[edge_iter_dim]
                    for k in range(S): # Iterate along the edge length
                        iter_coord3 = base_coord3 + k
                        neighbor_block = [0, 0, 0] # Placeholder
                        neighbor_block[edge_dim1] = coord1
                        neighbor_block[edge_dim2] = coord2
                        neighbor_block[edge_iter_dim] = iter_coord3
                        neighbor_block_tuple = tuple(neighbor_block)
                        if neighbor_block_tuple in current_occupied_blocks:
                             # print(f"Edge Collision: Neighbor {neighbor_block_tuple} for cube ({x},{y},{z}, S={S})")
                             return False # Invalid placement (Edge Contact)
    # --- End Edge/Face Checks ---
    # 3. Check for block overlaps (volume check) - Safeguard, should be covered by above checks
    # This check is technically redundant if face/edge checks are correct, but harmless.
    for block in get_cube_blocks(x, y, z, S):
        if block in current_occupied_blocks:
             # print(f"Volume Collision: Block {block} occupied (trying {S}x{S}x{S} at {(x,y,z)})")
             return False
    # 4. Check vertex sharing limits (ensure NO vertex is shared by > 2 cubes)
    vertices = get_cube_vertices(x, y, z, S)
    for v in vertices:
        if current_vertex_counts.get(v, 0) >= 2:
            # print(f"Vertex Limit Collision: Vertex {v} rule fail (count >= 2) (trying {S}x{S}x{S} at {(x,y,z)})")
            return False
    return True # All checks passed
# --- get_placement_coords_at_vertex (Requires Parent Info for Corner Attachment) ---
def get_placement_coords_at_vertex(vertex, cube_size_S, parent_coords, parent_size):
    """
    Determines the bottom-back-left (BBL) coords to place a cube of size
    cube_size_S attaching AT the vertex of the parent cube, ensuring only
    corner contact. Assumes integer coordinates.
    Requires parent's BBL coords (parent_coords) and size (parent_size).
    Returns None if parent info is invalid or placement calculation fails.
    """
    if parent_coords is None or parent_size is None or parent_size < 1 or cube_size_S < 1:
        # print(f"Warning: Invalid input in get_placement_coords_at_vertex: V={vertex}, S={cube_size_S}, ParentCoords={parent_coords}, ParentSize={parent_size}")
        return None # Cannot calculate without valid info
    vx, vy, vz = vertex
    px, py, pz = parent_coords
    pS = parent_size
    # Check if vertex is actually a vertex of the parent (basic sanity check)
    is_valid_vertex = False
    for dx, dy, dz in itertools.product([0, pS], repeat=3):
        if (px + dx, py + dy, pz + dz) == vertex:
            is_valid_vertex = True
            break
    if not is_valid_vertex:
        # This shouldn't happen if vertex_info is correct, but good check
        # print(f"Warning: Vertex {vertex} is not a valid vertex of parent at {parent_coords} with size {pS}.")
        return None # Return None to prevent bad placements
    # Determine BBL for the new cube based on vertex position relative to parent
    # If the vertex coord matches the parent's min coord (e.g., vx == px),
    # the new cube must end there, so its BBL is (coord - new_size).
    # Otherwise (if it matches parent's max coord: vx == px + pS),
    # the new cube must start there, so its BBL is the coord itself.
    nx = vx - cube_size_S if vx == px else vx
    ny = vy - cube_size_S if vy == py else vy
    nz = vz - cube_size_S if vz == pz else vz
    return nx, ny, nz
# --- place_cube (Simplified - No Parent Info Needed for Checks) ---
def place_cube(x, y, z, S, color):
    # Adds a cube with color, updates structures including vertex_info.
    global placed_cubes, occupied_blocks, vertex_counts, vertex_info
    global bounds_min_int, bounds_max_int # Needed for check_placement call
    if S < 1: return False
    # Geometric check before committing - Uses strict check_placement
    is_phase2 = bounds_min_int is not None
    if not check_placement(x, y, z, S, occupied_blocks, vertex_counts,
                         bounds_min=bounds_min_int if is_phase2 else None,
                         bounds_max=bounds_max_int if is_phase2 else None):
         # print(f"Warning: place_cube check failed S={S} at ({x},{y},{z})")
         return False
    # --- Placement is geometrically valid, update structures ---
    cube_id = len(placed_cubes) # ID is the index *before* appending
    placed_cubes.append((x, y, z, S, color))
    # Update occupied blocks
    occupied_blocks.update(get_cube_blocks(x, y, z, S))
    # Update vertex counts and the EXPOSED vertex info map
    new_vertices = get_cube_vertices(x, y, z, S)
    for vertex in new_vertices:
        vertex_counts[vertex] += 1
        current_count = vertex_counts[vertex]
        # Update vertex_info based on the new count
        if current_count == 1:
            vertex_info[vertex] = {
                'parent_id': cube_id,
                'parent_size': S,
                'parent_color': color
            }
        elif current_count > 1 and vertex in vertex_info:
            # Vertex was exposed but is now shared or internal
            del vertex_info[vertex]
    # print(f"Placed {S}x{S}x{S} cube ({color}) at ({x},{y},{z}) - ID: {cube_id}")
    return True
# =========================
# --- Main Program Loop ---
# =========================
while True:
    # --- Reset State Variables for New Fractal ---
    placed_cubes = []
    occupied_blocks = set()
    vertex_counts = defaultdict(int)
    vertex_info = {}
    bounds_min_int = None
    bounds_max_int = None
        # --- Get User Input for N_FRAMEWORK or Quit ---
    print("\n" + "="*40) # Add separators for clarity
    print("\nWelcome to Marlowe's 3D Fractal Art Generator.")
    print("This fractal is initially defined as a framework of size N.")
    print("(N = number of layers added after the initial 1x1x1 cube.)")
    print("Higher N values increase complexity exponentially (mainly the image rendering).")
    print("Recommended: N=(2-4). N=5+ requires significant time/memory.")
    print("Enter 'q' to quit.")
    print("\n" + "="*40)
    N_FRAMEWORK = None # Initialize N_FRAMEWORK for this loop iteration
    while True: # Inner loop for input validation
        framework_input = input("\nEnter a value for N: ").strip().lower() # Get input, remove whitespace, make lowercase
        if framework_input == 'q':
            # User wants to quit, break inner loop, N_FRAMEWORK remains None
            break
        # If not 'q', try converting to integer
        try:
            N_FRAMEWORK_temp = int(framework_input)
        except ValueError:
            # Input could not be converted to an integer
            print("Error: Invalid input. Please enter a whole number or 'q'.")
            # Loop continues automatically
        else:
            # Try succeeded, check positivity
            if N_FRAMEWORK_temp > 0:
                N_FRAMEWORK = N_FRAMEWORK_temp # Assign valid N
                print(f"Using N_FRAMEWORK = {N_FRAMEWORK}")
                break # Exit the inner input loop, input is valid
            else:
                # Input was a valid integer, but not positive
                print("Error: Please enter a positive whole number (greater than 0).")
                # Loop continues automatically
    # --- Check if user chose to quit ---
    if N_FRAMEWORK is None:
        print("Exiting program. Goodbye!")
        break # Exit the main outer loop
    # =====================================
    # --- Phase 1: Framework Generation ---
    # =====================================
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    # Place initiator (Red)
    initiator_color = COLOR_SEQUENCE[0] # Red
    if not place_cube(0, 0, 0, 1, initiator_color):
         print("CRITICAL ERROR: Could not place initiator cube.")
         exit()
    framework_paths = {} # Store {direction_vector: last_cube_id}
    origin = np.array([0.5, 0.5, 0.5])
    initiator_vertices = list(get_cube_vertices(0, 0, 0, 1))
    for v in initiator_vertices:
        direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
        is_diagonal = all(c != 0 for c in direction_vector)
        if is_diagonal:
            framework_paths[direction_vector] = 0 # Store initiator cube ID (index 0)
    # Framework iterations
    for n in range(1, N_FRAMEWORK + 1):
        target_size = n + 1
        layer_color = COLOR_SEQUENCE[n % len(COLOR_SEQUENCE)]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color}")
        new_framework_paths = {}
        placed_count_iter = 0
        current_paths = framework_paths.copy()
        for direction_vec, last_cube_id in current_paths.items():
            # --- Get Parent Info for Placement Calculation ---
            if not (0 <= last_cube_id < len(placed_cubes)):
                print(f"Error: Invalid last_cube_id {last_cube_id} in framework path {direction_vec}. Skipping.")
                continue
            lx, ly, lz, lS, _ = placed_cubes[last_cube_id]
            parent_coords_phase1 = (lx, ly, lz)
            parent_size_phase1 = lS
            # Find the vertex on the last cube furthest along the growth direction
            last_cube_verts = get_cube_vertices(lx, ly, lz, lS)
            cube_center = np.array([lx + lS/2, ly + lS/2, lz + lS/2])
            # Ensure attachment_vertex is calculated correctly
            try:
                attachment_vertex = max(last_cube_verts, key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError: # Handle empty last_cube_verts (shouldn't happen)
                 print(f"Error: Could not find attachment vertex for framework cube D={direction_vec}, ParentID={last_cube_id}. Skipping.")
                 continue
            # --- Calculate placement coords using function requiring parent info ---
            placement_result = get_placement_coords_at_vertex(attachment_vertex, target_size, parent_coords_phase1, parent_size_phase1)
            if placement_result is None:
                print(f"Error: Failed to calculate placement coords for framework cube {layer_color} S={target_size} for D={direction_vec}. Vertex={attachment_vertex}, Parent={parent_coords_phase1}, Size={parent_size_phase1}")
                continue
            px, py, pz = placement_result
            # Attempt to place the new cube (uses simple place_cube/check_placement)
            if place_cube(px, py, pz, target_size, layer_color):
                new_cube_id = len(placed_cubes) - 1
                new_framework_paths[direction_vec] = new_cube_id
                placed_count_iter += 1
            else:
                 # Check_placement likely failed due to strict adjacency rule
                 # print(f"Info: Failed geometric check (likely adjacency) for framework cube {layer_color} S={target_size} for D={direction_vec} at Coords=({px},{py},{pz})")
                 pass # Reduce noise, failure is expected sometimes with strict rules
        framework_paths.update(new_framework_paths)
        print(f"  Placed {placed_count_iter} framework cubes.")
        if placed_count_iter != 8 and n > 0:
            print(f"Warning: Expected 8 framework cubes, placed {placed_count_iter} in iteration {n}")
    duration_phase1 = time.time() - start_time_phase1
    print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---")
    print(f"Total cubes after Phase 1: {len(placed_cubes)}")
    # ====================================
    # --- Calculate Final Bounding Box ---
    # ====================================
    print("\n--- Calculating Final Bounding Box ---")
    min_coord_f = np.array([float('inf')] * 3)
    max_coord_f = np.array([float('-inf')] * 3)
    if not placed_cubes:
        print("Warning: No cubes placed, cannot determine bounds.")
        bounds_min_int = np.array([0, 0, 0])
        bounds_max_int = np.array([1, 1, 1])
    else:
        for x, y, z, S, _ in placed_cubes:
             min_coord_f = np.minimum(min_coord_f, [x, y, z])
             max_coord_f = np.maximum(max_coord_f, [x + S, y + S, z + S])
        bounds_min_int = np.floor(min_coord_f).astype(int)
        bounds_max_int = np.ceil(max_coord_f).astype(int)
    print(f"Bounding Box Min (int, BBL inclusive): {bounds_min_int}")
    print(f"Bounding Box Max (int, exclusive): {bounds_max_int}")
    if np.any(bounds_max_int <= bounds_min_int):
        print("ERROR: Invalid bounding box calculated (max <= min). Exiting.")
        exit()
    # ==================================================
    # --- Phase 2: Color-Based Generational Layering ---
    # ==================================================
    print(f"\n--- Starting Phase 2: Color-Based Generational Layering ---")
    start_time_phase2 = time.time()
    max_color_cycles = 100
    cycles_without_placement = 0
    current_target_color_index = 1
    total_placed_phase2 = 0
    cycle = 0
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles:
        cycle += 1
        target_color_to_place = COLOR_SEQUENCE[current_target_color_index % len(COLOR_SEQUENCE)]
        required_parent_color = get_required_parent_color(target_color_to_place)
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_to_place} onto {required_parent_color}")
        # 1. Find exposed vertices of the required PARENT color
        parent_vertices_candidates = [] # Store {'vertex': V, 'parent_id': ID, 'parent_size': S}
        current_vertex_info_snapshot = vertex_info.copy() # Work on snapshot
        for V, info in current_vertex_info_snapshot.items():
            if info['parent_color'] == required_parent_color:
                # Basic check: ensure parent_id is valid before adding
                if 0 <= info['parent_id'] < len(placed_cubes):
                     parent_vertices_candidates.append({
                         'vertex': V,
                         'parent_id': info['parent_id'],
                         'parent_size': info['parent_size']
                     })
                # else: # Debugging invalid parent IDs in vertex_info
                #    print(f"Debug: Found vertex {V} with invalid parent_id {info['parent_id']} in vertex_info.")
        # 1b. Pre-filter based on bounds check for S=1 placement
        parent_vertices_found = []
        for candidate in parent_vertices_candidates:
            V = candidate['vertex']
            parent_id = candidate['parent_id']
            parent_size = candidate['parent_size']
            # --- Get Parent Coords for Bounds Check ---
            # We already checked parent_id validity when creating candidates
            px_chk, py_chk, pz_chk, _, _ = placed_cubes[parent_id]
            parent_coords_check = (px_chk, py_chk, pz_chk)
            # --- Calculate placement for S=1 using correct function ---
            placement_s1 = get_placement_coords_at_vertex(V, 1, parent_coords_check, parent_size)
            if placement_s1 is None:
                # print(f"Warning: Failed to calculate S=1 placement for bounds check V={V}. Skipping.")
                continue # Skip if placement calculation fails
            px1, py1, pz1 = placement_s1
            # Check if S=1 placement is within bounds
            if not (px1 < bounds_min_int[0] or (px1 + 1) > bounds_max_int[0] or
                    py1 < bounds_min_int[1] or (py1 + 1) > bounds_max_int[1] or
                    pz1 < bounds_min_int[2] or (pz1 + 1) > bounds_max_int[2]):
                # Bounds check passed, add to the list for sorting
                parent_vertices_found.append({
                    'vertex': V,
                    'distance': distance_to_origin(V), # Calculate distance for sorting
                    'parent_size': parent_size, # Pass size along
                    'parent_id': parent_id # Pass ID along
                })
        if not parent_vertices_found:
            print(f"  No valid exposed vertices found for parent color {required_parent_color} (after bounds filter).")
            cycles_without_placement += 1
            current_target_color_index += 1
            continue
        print(f"  Found {len(parent_vertices_found)} potential parent vertices ({required_parent_color}).")
        # 2. Sort parent vertices by distance
        parent_vertices_found.sort(key=lambda x: x['distance'])
        # 3. Attempt placement sequentially
        placed_this_color_cycle = 0
        processed_vertices_this_cycle = set()
        for parent_data in parent_vertices_found:
            V = parent_data['vertex']
            # Check if vertex still exists and hasn't been processed this cycle
            if V in processed_vertices_this_cycle or V not in vertex_info:
                continue
            # --- Get Parent Info (Needed for get_placement_coords_at_vertex) ---
            S_parent = parent_data['parent_size']
            parent_id = parent_data['parent_id']
            current_parent_coords = None
            current_parent_size = S_parent # Use size from parent_data
            # Retrieve parent coordinates (already checked ID validity earlier)
            parent_x, parent_y, parent_z, pS_check, _ = placed_cubes[parent_id]
            if pS_check != S_parent: # Sanity check
                 print(f"Warning: Mismatch parent size for ID {parent_id}. Data:{S_parent}, List:{pS_check}")
            current_parent_coords = (parent_x, parent_y, parent_z)
            # --- Find largest size that fits geometrically ---
            target_size_start = S_parent + 1
            best_S_found = 0
            placement_coords = None
            for S_try in range(target_size_start, 0, -1):
                # Calculate placement coords using the function requiring parent info
                px_py_pz = get_placement_coords_at_vertex(V, S_try, current_parent_coords, current_parent_size)
                if px_py_pz is None: # Placement calculation failed
                    continue
                px, py, pz = px_py_pz
                # Call the STRICT check_placement (which doesn't need parent info itself)
                if check_placement(px, py, pz, S_try, occupied_blocks, vertex_counts,
                                   bounds_min=bounds_min_int, bounds_max=bounds_max_int):
                    best_S_found = S_try
                    placement_coords = (px, py, pz)
                    break # Found largest suitable size
                # else: # Debugging check_placement failures
                #    print(f"DEBUG: check_placement failed for S={S_try} at ({px},{py},{pz}) from V={V}, Parent={current_parent_coords},S={current_parent_size}")
            # --- End Size Finding Loop ---
            # --- If a valid size was found, attempt to place the cube ---
            if best_S_found > 0 and placement_coords:
                # Call the SIMPLE place_cube (which doesn't need parent info)
                if place_cube(placement_coords[0], placement_coords[1], placement_coords[2],
                              best_S_found, target_color_to_place):
                    placed_this_color_cycle += 1
                    # Mark parent vertex V as processed for this cycle.
                    # place_cube handles removing V from vertex_info if it gets covered.
                    processed_vertices_this_cycle.add(V)
                # else: # Optional debug if place_cube fails after check_placement passed
                #    print(f"Debug: place_cube failed for S={best_S_found} at {placement_coords} after check_placement passed.")
            # --- End Placement Attempt ---
        # *** END OF CORRECTED INDENTATION BLOCK ***
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_to_place}.")
        # Update total and track consecutive non-placement cycles
        if placed_this_color_cycle > 0:
            total_placed_phase2 += placed_this_color_cycle
            cycles_without_placement = 0
        else:
            cycles_without_placement += 1
        # Move to the next target color for the next cycle
        current_target_color_index += 1
    # --- End Phase 2 Loop ---
    duration_phase2 = time.time() - start_time_phase2
    print(f"--- Phase 2 Complete ({duration_phase2:.2f}s in {cycle} color cycles) ---")
    print(f"Total cubes placed in Phase 2: {total_placed_phase2}")
    print(f"Total cubes after Phase 2: {len(placed_cubes)}")
    print(f"Total occupied 1x1x1 blocks: {len(occupied_blocks)}")
    print(f"Total unique vertices created: {len(vertex_counts)}")
    print(f"Final exposed vertices map size: {len(vertex_info)}")
    # ===================================
    # --- Final Voxel Grid Generation ---
    # ===================================
    print("\n--- Generating Final Voxel Grid ---")
    start_time_voxel = time.time()
    grid_min = bounds_min_int
    grid_max = bounds_max_int
    grid_size = grid_max - grid_min
    print(f"Grid Dimensions (X,Y,Z): {grid_size}")
    if np.any(grid_size <= 0):
        print("Error: Invalid grid size calculated. Cannot create voxel grid.")
        voxel_grid = None
        color_grid = None
    else:
        # Create boolean grid for filled status and object grid for colors
        voxel_grid = np.zeros(grid_size, dtype=bool)
        color_grid = np.empty(grid_size, dtype=object) # To store color strings/tuples
        cubes_processed = 0
        skipped_cubes = 0
        print(f"Processing {len(placed_cubes)} cubes into grid...")
        # Iterate through all placed cubes (from phase 1 and phase 2)
        for i, (x, y, z, S, color) in enumerate(placed_cubes):
            # Convert logical coords to grid indices (relative to grid_min)
            # Use floor for start index to handle potential negative coords correctly
            ix_start = int(np.floor(x - grid_min[0]))
            iy_start = int(np.floor(y - grid_min[1]))
            iz_start = int(np.floor(z - grid_min[2]))
            # Calculate end indices (exclusive for slicing)
            ix_end = ix_start + S
            iy_end = iy_start + S
            iz_end = iz_start + S
            # Check if the cube's grid indices fit within the numpy array bounds
            if (0 <= ix_start < grid_size[0] and 0 <= iy_start < grid_size[1] and 0 <= iz_start < grid_size[2] and
                ix_end <= grid_size[0] and iy_end <= grid_size[1] and iz_end <= grid_size[2]):
                # Assign boolean value (True) and the plot color to the slice
                voxel_grid[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] = True
                plot_color = COLOR_MAP_PLOT.get(color, COLOR_MAP_PLOT['default']) # Get hex color
                color_grid[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] = plot_color
                cubes_processed += 1
            else:
                # This might happen if rounding/floor causes issues right at the boundary
                print(f"Warning: Cube {i} ({x},{y},{z}, S={S}, {color}) out of calculated bounds during voxelization. Indices ({ix_start}:{ix_end}, {iy_start}:{iy_end}, {iz_start}:{iz_end}). Grid Size {grid_size}. Skipping.")
                skipped_cubes += 1
        print(f"Processed {cubes_processed} cubes into voxel grid. Skipped {skipped_cubes} cubes.")
        duration_voxel = time.time() - start_time_voxel
        print(f"Voxel grid generation complete ({duration_voxel:.2f}s).")
    # =====================
    # --- Visualization ---
    # =====================
    print("\n--- Visualizing ---")
    if voxel_grid is None or not np.any(voxel_grid):
         print("Voxel grid is empty or invalid. Nothing to visualize.")
    else:
        # Get the grid dimensions
        nx, ny, nz = voxel_grid.shape
        # Create an RGBA array with the same dimensions as voxel_grid, plus alpha channel
        rgba_grid = np.zeros((nx, ny, nz, 4)) # Default is transparent black [0,0,0,0]
        # Populate the RGBA grid based on the color_grid for True voxels
        #print("DEBUG: Populating RGBA grid...")
        true_indices = np.argwhere(voxel_grid) # Get indices where voxel_grid is True
        if len(true_indices) > 0:
            hex_colors_for_true_voxels = color_grid[voxel_grid] # Get the corresponding hex colors
            # Check if lengths match (sanity check)
            if len(true_indices) != len(hex_colors_for_true_voxels):
                print(f"ERROR: Mismatch between true voxel count ({len(true_indices)}) and extracted colors ({len(hex_colors_for_true_voxels)})!")
                rgba_grid = None # Indicate failure
            else:
                # Convert hex colors to RGBA and assign them to the correct locations in rgba_grid
                try:
                    # Ensure all elements are strings before conversion
                    hex_colors_for_true_voxels = hex_colors_for_true_voxels.astype(str)
                    rgba_colors = mcolors.to_rgba_array(hex_colors_for_true_voxels)
                    # Use advanced indexing to place the RGBA values into the grid
                    rgba_grid[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2]] = rgba_colors
                    #print(f"DEBUG: Successfully populated RGBA grid. Shape: {rgba_grid.shape}")
                except Exception as e:
                    print(f"ERROR: Failed during RGBA grid population: {e}")
                    rgba_grid = None # Indicate failure
        else:
            print("DEBUG: No True voxels found to populate RGBA grid.")
        # --- Plotting ---
        print("Processing image... (This can take a long time for high values of N, monitor resource use carefully.)")
        if rgba_grid is not None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            # Pass the boolean grid and the full 4D RGBA grid
            ax.voxels(voxel_grid, facecolors=rgba_grid, edgecolor=None, shade=False)
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            ax.set_zlabel("Z index")
            ax.set_title(f"3D Fractal (Framework N={N_FRAMEWORK}, Color Layers)")
            # Set axis limits based on the numpy grid dimensions (0 to size-1)
            ax.set_xlim(0, grid_size[0])
            ax.set_ylim(0, grid_size[1])
            ax.set_zlim(0, grid_size[2])
            # Ensure axes have equal visual scaling
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout() # Adjust plot to prevent labels overlapping
            plt.show()
        else:
            print("Visualization skipped due to RGBA grid population error or empty grid.")
    print("--- Script Finished ---")
