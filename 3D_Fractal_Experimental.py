import numpy as np
import numba
# Removed: from numba.typed import Dict # No longer needed
from vispy import scene, app, color
from vispy.scene import visuals
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
placed_cubes = [] # List to store tuples: (x, y, z, S, color)
# Removed: occupied_blocks = set() # No longer needed
vertex_counts = defaultdict(int) # Stores count of cubes sharing a vertex
vertex_info = {} # Stores info about exposed vertices: {vertex_tuple: {'parent_id', 'parent_size', 'parent_color'}}
bounds_min_int = None # Tuple (min_x, min_y, min_z)
bounds_max_int = None # Tuple (max_x, max_y, max_z)
# --- Helper Functions ---
# (get_next_color, get_required_parent_color remain the same)
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
# --- Numba Accelerated Helper Functions ---
@numba.jit(nopython=True)
def get_cube_vertices_tuple_list(x, y, z, S):
    """Numba-friendly version returning a list of tuples for vertices."""
    # Note: Numba doesn't directly support list comprehensions with multiple loops easily
    # in nopython mode sometimes, so an explicit loop is safer.
    vertices_list = []
    # Using float/int directly should be fine as inputs are likely numeric
    xf, yf, zf = float(x), float(y), float(z)
    Sf = float(S)
    for dx in [0.0, Sf]:
        for dy in [0.0, Sf]:
            for dz in [0.0, Sf]:
                # Ensure output is tuple of integers if inputs were integers
                vertices_list.append((int(xf + dx), int(yf + dy), int(zf + dz)))
    return vertices_list
# Removed: get_cube_blocks_list # No longer needed
@numba.jit(nopython=True)
def distance_to_origin(vx, vy, vz):
    """Numba-friendly distance calculation."""
    ox, oy, oz = 0.5, 0.5, 0.5 # Center of the initial 1x1x1 cube
    return math.sqrt((float(vx) - ox)**2 + (float(vy) - oy)**2 + (float(vz) - oz)**2)
# Removed: Old check_placement function
# --- NEW: Numba function for precise cube-pair contact check ---
@numba.jit(nopython=True)
def get_contact_type(x1, y1, z1, S1, x2, y2, z2, S2):
    """
    Determines the type of contact between two axis-aligned cubes.
    Assumes integer coordinates and sizes.
    Args:
        x1, y1, z1, S1: Coordinates (BBL) and size of cube 1.
        x2, y2, z2, S2: Coordinates (BBL) and size of cube 2.
    Returns:
        int: Contact type code:
            0: No Contact
            1: Vertex Contact (touching at a single corner)
            2: Edge Contact (sharing an edge segment, not a face)
            3: Face Contact (sharing a face area)
            4: Volume Overlap (cubes intersect)
    """
    # Define intervals (inclusive)
    x1_max, y1_max, z1_max = x1 + S1, y1 + S1, z1 + S1
    x2_max, y2_max, z2_max = x2 + S2, y2 + S2, z2 + S2
    # Calculate intersection intervals
    ix_min, ix_max = max(x1, x2), min(x1_max, x2_max)
    iy_min, iy_max = max(y1, y2), min(y1_max, y2_max)
    iz_min, iz_max = max(z1, z2), min(z1_max, z2_max)
    # Calculate lengths of intersection intervals
    len_x = ix_max - ix_min
    len_y = iy_max - iy_min
    len_z = iz_max - iz_min
    # Check for no contact (negative length means no overlap on that axis)
    if len_x < 0 or len_y < 0 or len_z < 0:
        return 0 # No Contact
    # Count dimensions with zero length (touching) vs positive length (overlap)
    zero_dims = 0
    if len_x == 0: zero_dims += 1
    if len_y == 0: zero_dims += 1
    if len_z == 0: zero_dims += 1
    # Classify based on intersection lengths
    if len_x > 0 and len_y > 0 and len_z > 0:
        return 4 # Volume Overlap
    elif zero_dims == 1: # Exactly one dimension is touching, others overlap
        return 3 # Face Contact
    elif zero_dims == 2: # Exactly two dimensions are touching, one overlaps
        return 2 # Edge Contact
    elif zero_dims == 3: # All three dimensions are touching at a single point
        return 1 # Vertex Contact
    else:
        # This case should theoretically not be reached if logic above is sound
        # but acts as a safeguard. It implies some overlap exists but doesn't fit
        # the standard categories cleanly, which might indicate an issue.
        # However, given integer coords/sizes, it likely means Volume Overlap
        # if any len > 0. Let's assume Volume Overlap if not Face/Edge/Vertex.
        # Re-evaluating: If len_x>=0, len_y>=0, len_z>=0, and it's not 1,2,3,
        # it must have at least one positive length, and not fit the face/edge patterns.
        # This implies volume overlap.
        # Let's refine the conditions slightly for clarity:
        if len_x > 0 and len_y > 0 and len_z > 0: return 4
        if (len_x == 0 and len_y > 0 and len_z > 0) or \
           (len_x > 0 and len_y == 0 and len_z > 0) or \
           (len_x > 0 and len_y > 0 and len_z == 0): return 3
        if (len_x == 0 and len_y == 0 and len_z > 0) or \
           (len_x == 0 and len_y > 0 and len_z == 0) or \
           (len_x > 0 and len_y == 0 and len_z == 0): return 2
        if len_x == 0 and len_y == 0 and len_z == 0: return 1
        # Fallback/Error case (should not happen with integer inputs)
        return 0 # Or perhaps raise an error? For now, return No Contact.
# --- NEW: Numba function to validate placement against existing cubes ---
@numba.jit(nopython=True)
def is_placement_valid(cand_x, cand_y, cand_z, cand_S,
                       placed_cubes_arr, # Numpy array (N, 4) of x,y,z,S
                       bounds_min_x, bounds_min_y, bounds_min_z,
                       bounds_max_x, bounds_max_y, bounds_max_z):
    """
    Checks if placing a candidate cube is valid according to rules:
    1. Within bounds.
    2. No volume overlap with any existing cube.
    3. No face contact with any existing cube.
    4. No edge contact with any existing cube.
    (Vertex contact is allowed).
    Args:
        cand_x, y, z, S: Candidate cube parameters.
        placed_cubes_arr: NumPy array of existing cubes' (x, y, z, S).
        bounds_min/max_x,y,z: Bounding box limits.
    Returns:
        bool: True if placement is valid, False otherwise.
    """
    if cand_S < 1: return False # Ensure valid size
    # 1. Bounds Check
    has_bounds = bounds_min_x != -9999 # Check if bounds are set
    if has_bounds:
        if (cand_x < bounds_min_x or (cand_x + cand_S) > bounds_max_x or
            cand_y < bounds_min_y or (cand_y + cand_S) > bounds_max_y or
            cand_z < bounds_min_z or (cand_z + cand_S) > bounds_max_z):
            return False # Outside bounds
    # 2. Check against each existing cube
    num_placed = placed_cubes_arr.shape[0]
    for i in range(num_placed):
        exist_x, exist_y, exist_z, exist_S = placed_cubes_arr[i, 0], placed_cubes_arr[i, 1], placed_cubes_arr[i, 2], placed_cubes_arr[i, 3]
        contact = get_contact_type(cand_x, cand_y, cand_z, cand_S,
                                   exist_x, exist_y, exist_z, exist_S)
        # Check for forbidden contact types
        if contact == 4: # Volume Overlap
            return False
        if contact == 3: # Face Contact
            return False
        if contact == 2: # Edge Contact
            return False
        # contact == 1 (Vertex Contact) is allowed
        # contact == 0 (No Contact) is allowed
    # If loop completes without finding invalid contact, placement is valid
    return True
@numba.jit(nopython=True)
def get_placement_coords_at_vertex(vertex_x, vertex_y, vertex_z,
                                 cube_size_S,
                                 parent_coords_x, parent_coords_y, parent_coords_z,
                                 parent_size):
    """Numba-friendly version. Determines BBL coords for a new cube of size
       cube_size_S attaching at vertex (vx,vy,vz) of parent cube."""
    # Ensure inputs are valid
    if parent_size < 1 or cube_size_S < 1:
        return (-9999, -9999, -9999) # Indicate failure
    # Check if (vertex_x, vertex_y, vertex_z) is actually a vertex of the parent
    is_valid_vertex = False
    px_max, py_max, pz_max = parent_coords_x + parent_size, parent_coords_y + parent_size, parent_coords_z + parent_size
    # Check if vertex components match parent boundaries
    valid_x = (vertex_x == parent_coords_x or vertex_x == px_max)
    valid_y = (vertex_y == parent_coords_y or vertex_y == py_max)
    valid_z = (vertex_z == parent_coords_z or vertex_z == pz_max)
    if not (valid_x and valid_y and valid_z):
         # The provided point is not one of the 8 corners of the parent cube
         return (-9999, -9999, -9999) # Indicate failure
    # Determine the Bottom-Back-Left (BBL) corner of the new cube
    # If the vertex coordinate is the minimum for that axis on the parent,
    # the new cube's BBL coordinate for that axis must be shifted back by its size.
    # Otherwise, the new cube's BBL coordinate matches the vertex coordinate.
    nx = vertex_x - cube_size_S if vertex_x == parent_coords_x else vertex_x
    ny = vertex_y - cube_size_S if vertex_y == parent_coords_y else vertex_y
    nz = vertex_z - cube_size_S if vertex_z == parent_coords_z else vertex_z
    return (nx, ny, nz)
# --- Modified place_cube (Uses new validation, no dictionary) ---
def place_cube(x, y, z, S, color): # Removed occupied_blocks_numba_dict argument
    """
    Attempts to place a cube. Checks validity using is_placement_valid.
    If valid, adds cube to placed_cubes list and updates vertex info.
    """
    global placed_cubes, vertex_counts, vertex_info # Removed occupied_blocks
    global bounds_min_int, bounds_max_int
    if S < 1: return False
    # Prepare bounds components for is_placement_valid
    # Use -9999 as a sentinel value if bounds are not yet calculated
    b_min_x = bounds_min_int[0] if bounds_min_int is not None else -9999
    b_min_y = bounds_min_int[1] if bounds_min_int is not None else -9999
    b_min_z = bounds_min_int[2] if bounds_min_int is not None else -9999
    b_max_x = bounds_max_int[0] if bounds_max_int is not None else -9999
    b_max_y = bounds_max_int[1] if bounds_max_int is not None else -9999
    b_max_z = bounds_max_int[2] if bounds_max_int is not None else -9999
    # --- Convert placed_cubes list to NumPy array for Numba function ---
    # Optimization Note: This conversion happens on every call. If it becomes
    # a bottleneck for very large numbers of cubes, we might maintain the
    # array more directly, but start with this simpler approach.
    if not placed_cubes:
        # Provide an empty array with the correct shape (0 rows, 4 columns)
        # Use int64 to match typical Numba integer types
        placed_array = np.empty((0, 4), dtype=np.int64)
    else:
        # Extract only x, y, z, S for the collision check array
        data_for_array = [(c[0], c[1], c[2], c[3]) for c in placed_cubes]
        placed_array = np.array(data_for_array, dtype=np.int64) # Ensure consistent type
    # --- Call the NEW JITted validation function ---
    if not is_placement_valid(x, y, z, S, placed_array,
                              b_min_x, b_min_y, b_min_z,
                              b_max_x, b_max_y, b_max_z):
        # print(f"DEBUG: Placement invalid for cube S={S} at ({x},{y},{z})") # Optional debug
        return False
    # --- Placement is geometrically valid, update structures ---
    # print(f"DEBUG: Placing cube S={S} at ({x},{y},{z}), Color: {color}") # Optional debug
    cube_id = len(placed_cubes)
    placed_cubes.append((x, y, z, S, color)) # Add to Python list
    # Removed: occupied_blocks update logic
    # Update vertex counts and vertex_info (standard Python dicts)
    # Use the JITted function to get vertices
    new_vertices_list = get_cube_vertices_tuple_list(x, y, z, S)
    for vertex in new_vertices_list:
        vertex_counts[vertex] += 1
        current_count = vertex_counts[vertex]
        if current_count == 1:
            # First time this vertex is created, mark it as exposed
            vertex_info[vertex] = {
                'parent_id': cube_id,
                'parent_size': S,
                'parent_color': color
            }
        elif current_count > 1 and vertex in vertex_info:
            # This vertex is now shared by multiple cubes, it's no longer exposed
            # Remove it from the exposed vertex info dictionary
            del vertex_info[vertex]
        # If current_count > 1 and vertex NOT in vertex_info, it means it was
        # already covered by previous cubes, so do nothing.
    return True
# --- Helper Function to Generate Mesh Data ---
# (generate_cube_mesh_data function remains unchanged)
def generate_cube_mesh_data(cubes_list):
    """Generates vertices, faces, and colors for a Mesh visual from a list of cubes."""
    num_cubes = len(cubes_list)
    if num_cubes == 0: return None, None, None
    # Unit cube vertices centered at 0,0,0 (size 1x1x1)
    unit_v = np.array([[-0.5,-0.5,-0.5],[+0.5,-0.5,-0.5],[+0.5,+0.5,-0.5],[-0.5,+0.5,-0.5],
                       [-0.5,-0.5,+0.5],[+0.5,-0.5,+0.5],[+0.5,+0.5,+0.5],[-0.5,+0.5,+0.5]])
    # Unit cube faces (triangles)
    unit_f = np.array([[0,1,2],[0,2,3],[4,7,6],[4,6,5],[0,4,5],[0,5,1],
                       [3,2,6],[3,6,7],[0,3,7],[0,7,4],[1,5,6],[1,6,2]], dtype=np.uint32)
    all_vertices = np.zeros((num_cubes * 8, 3), dtype=np.float32)
    all_faces = np.zeros((num_cubes * 12, 3), dtype=np.uint32)
    all_colors = np.zeros((num_cubes * 8, 4), dtype=np.float32) # RGBA per vertex
    print(f"Generating mesh data for {num_cubes} cubes...")
    start_mesh_gen = time.time()
    for i, (x, y, z, S, color_name) in enumerate(cubes_list):
        # Calculate center based on BBL (x,y,z) and size S
        center = np.array([x + S/2.0, y + S/2.0, z + S/2.0], dtype=np.float32)
        # Scale unit vertices by S and translate to center
        scaled_vertices = unit_v * S + center
        # Calculate offsets for this cube's data in the large arrays
        v_offset = i * 8
        f_offset = i * 12
        # Add vertices
        all_vertices[v_offset : v_offset + 8] = scaled_vertices
        # Add faces, adjusting indices by vertex offset
        all_faces[f_offset : f_offset + 12] = unit_f + v_offset
        # Get color and apply to all 8 vertices of this cube
        hex_color = COLOR_MAP_PLOT.get(color_name, COLOR_MAP_PLOT['default'])
        try:
            # VisPy color conversion
            rgba = color.Color(hex_color).rgba
        except Exception:
            # Fallback to default color if parsing fails
            rgba = color.Color(COLOR_MAP_PLOT['default']).rgba
        all_colors[v_offset : v_offset + 8] = rgba # Assign same color to all 8 vertices
    gen_duration = time.time() - start_mesh_gen
    print(f"Mesh data generation complete ({gen_duration:.2f}s).")
    return all_vertices, all_faces, all_colors
# ===============================
# --- Main Program Loop ---
# ===============================
while True:
    # --- Reset State Variables ---
    placed_cubes = []
    # Removed: occupied_blocks = set()
    vertex_counts = defaultdict(int)
    vertex_info = {}
    bounds_min_int = None
    bounds_max_int = None
    # Removed: occupied_blocks_nb_dict creation
    # --- Get User Input for N_FRAMEWORK or Quit ---
    print("\n" + "=" * 69)
    print("Welcome to Marlowe's 3D Fractal Art Generator (VisPy + Numba - List Check)") # Updated title
    print("This fractal is initially defined as a framework of size N.")
    print("(N = number of layers added after the initial 1x1x1 cube)")
    print("Higher N values increase complexity exponentially.")
    print("Recommended: N=1 to N=4. N=5+ may take significant time.") # Adjusted recommendation
    print("Enter 'q' to quit.")
    print("=" * 69)
    N_FRAMEWORK = None
    while True: # Inner loop for input validation
        framework_input = input("\nEnter the desired value for N (e.g., 3) or 'q' to quit: ").strip().lower()
        if framework_input == 'q':
            break
        try:
            N_FRAMEWORK_temp = int(framework_input)
            if N_FRAMEWORK_temp >= 0: # Allow N=0 (just the initial cube)
                N_FRAMEWORK = N_FRAMEWORK_temp
                print(f"Using N_FRAMEWORK = {N_FRAMEWORK}")
                break
            else:
                print("Error: Please enter a non-negative whole number (0 or greater).")
        except ValueError:
            print("Error: Invalid input. Please enter a whole number or 'q'.")
    # --- Check if user chose to quit ---
    if N_FRAMEWORK is None:
        print("Exiting program. Goodbye!")
        break # Exit main loop
    # --- Phase 1: Framework Generation ---
    print(f"\n--- Starting Phase 1: Framework (N={N_FRAMEWORK}) ---")
    start_time_phase1 = time.time()
    # Place initiator cube (size 1x1x1 at origin)
    # Call place_cube WITHOUT the dictionary argument
    if not place_cube(0, 0, 0, 1, COLOR_SEQUENCE[0]):
        print("CRITICAL ERROR: Initiator cube placement failed (unexpected).")
        continue # Skip to next main loop iteration
    # Framework setup for N > 0
    framework_paths = {} # Stores {direction_vector: last_cube_id_in_path}
    if N_FRAMEWORK > 0:
        origin = np.array([0.5, 0.5, 0.5]) # Center of the initiator cube
        # Get vertices of the initiator cube
        initiator_vertices_list = get_cube_vertices_tuple_list(0, 0, 0, 1)
        # Initialize paths for the 8 diagonal directions from the origin
        for v in initiator_vertices_list:
            direction_vector = tuple(np.sign(np.array(v) - origin).astype(int))
            # Ensure it's a diagonal direction (no zero components)
            if all(c != 0 for c in direction_vector):
                framework_paths[direction_vector] = 0 # Initial cube (ID 0) is start of all paths
    # Framework iterations...
    for n in range(1, N_FRAMEWORK + 1):
        target_size = n + 1 # Size of cubes in this layer
        layer_color = COLOR_SEQUENCE[n % len(COLOR_SEQUENCE)]
        print(f"Framework Iteration {n}, Target Size: {target_size}x{target_size}x{target_size}, Color: {layer_color}")
        new_framework_paths = {} # Track newly placed cubes for this layer
        placed_count_iter = 0
        current_paths = framework_paths.copy() # Iterate over paths from previous layer
        for direction_vec, last_cube_id in current_paths.items():
            # Get details of the last cube in this path
            if not (0 <= last_cube_id < len(placed_cubes)):
                print(f"Warning: Invalid last_cube_id {last_cube_id} for direction {direction_vec}. Skipping.")
                continue
            lx, ly, lz, lS, _ = placed_cubes[last_cube_id]
            parent_size_phase1 = lS # Size of the cube we attach to
            # Find the vertex of the parent cube furthest along the direction vector
            last_cube_verts_list = get_cube_vertices_tuple_list(lx, ly, lz, lS)
            if not last_cube_verts_list:
                print(f"Warning: Could not get vertices for cube {last_cube_id}. Skipping.")
                continue
            cube_center = np.array([lx + lS / 2.0, ly + lS / 2.0, lz + lS / 2.0])
            try:
                # Find vertex maximizing dot product with direction vector (relative to center)
                attachment_vertex = max(last_cube_verts_list,
                                        key=lambda vert: np.dot(np.array(vert) - cube_center, direction_vec))
            except ValueError:
                 print(f"Warning: Could not find attachment vertex for cube {last_cube_id}. Skipping.")
                 continue # Should not happen if list is not empty
            # Calculate placement coordinates for the new cube
            vx, vy, vz = attachment_vertex
            placement_result_tuple = get_placement_coords_at_vertex(
                vx, vy, vz, target_size, lx, ly, lz, parent_size_phase1
            )
            # Check if placement calculation was successful
            if placement_result_tuple == (-9999, -9999, -9999):
                # This might happen if get_placement_coords_at_vertex fails its internal checks
                # print(f"DEBUG: Failed get_placement_coords_at_vertex for dir {direction_vec}")
                continue
            px, py, pz = placement_result_tuple
            # Attempt to place the new cube - Call place_cube WITHOUT dictionary
            if place_cube(px, py, pz, target_size, layer_color):
                new_cube_id = len(placed_cubes) - 1
                new_framework_paths[direction_vec] = new_cube_id # Update path end
                placed_count_iter += 1
            # else: # Debugging framework placement failures
                # print(f"DEBUG: Framework placement failed for dir {direction_vec} at iter {n}")
                # print(f"  Parent ID {last_cube_id}: ({lx},{ly},{lz}), S={lS}")
                # print(f"  Attach Vertex: {attachment_vertex}")
                # print(f"  Target Coords: ({px},{py},{pz}), S={target_size}")
                # # Optionally, run is_placement_valid again here with debug prints inside it
                # pass
        # Update framework paths with the newly placed cubes for the next iteration
        framework_paths.update(new_framework_paths)
        print(f"  Placed {placed_count_iter} framework cubes.")
        # Check if all 8 directions succeeded (only relevant after first iteration)
        if placed_count_iter != 8 and n > 0 and len(current_paths) == 8:
             print(f"WARNING: Placed {placed_count_iter}/8 framework cubes in iteration {n}. Geometry might be constrained.")
        elif placed_count_iter == 0 and n > 0:
             print(f"WARNING: No framework cubes placed in iteration {n}. Stopping framework generation.")
             break # Stop framework if no progress
    duration_phase1 = time.time() - start_time_phase1
    print(f"--- Phase 1 Complete ({duration_phase1:.2f}s) ---")
    print(f"Total cubes after Phase 1: {len(placed_cubes)}")
    # Check if Phase 1 actually placed anything beyond the initiator
    if len(placed_cubes) <= 1 and N_FRAMEWORK > 0:
        print("Warning: Phase 1 framework generation failed or was incomplete. Phase 2 might be empty.")
        # Decide whether to continue to Phase 2 or stop
        # continue # Go to next N input loop
        # For now, let's allow Phase 2 to run even if framework is minimal
    # --- Calculate Final Bounding Box ---
    print("\n--- Calculating Final Bounding Box ---")
    min_coord_f = np.array([float('inf')] * 3)
    max_coord_f = np.array([float('-inf')] * 3)
    if not placed_cubes:
        print("Error: No cubes placed. Skipping BBox calculation and Phase 2.")
        continue # Skip to next N input loop
    for x, y, z, S, _ in placed_cubes:
        min_coord_f = np.minimum(min_coord_f, [x, y, z])
        max_coord_f = np.maximum(max_coord_f, [x + S, y + S, z + S])
    # Convert to integer bounds (floor for min, ceil for max)
    bounds_min_int = np.floor(min_coord_f).astype(int)
    bounds_max_int = np.ceil(max_coord_f).astype(int)
    print(f"Bounding Box Min (int, BBL inclusive): {bounds_min_int}")
    print(f"Bounding Box Max (int, exclusive): {bounds_max_int}")
    # Validate bounds
    if np.any(bounds_max_int <= bounds_min_int):
        print("ERROR: Invalid bounding box calculated (max <= min). Skipping Phase 2.")
        continue # Skip to next N input loop
    # --- Phase 2: Color-Based Generational Layering ---
    print(f"\n--- Starting Phase 2: Color-Based Generational Layering ---")
    start_time_phase2 = time.time()
    max_color_cycles = 100 # Limit cycles to prevent infinite loops
    cycles_without_placement = 0
    # Start placing the color *after* the last framework color
    last_framework_color_index = N_FRAMEWORK % len(COLOR_SEQUENCE)
    current_target_color_index = (last_framework_color_index + 1) % len(COLOR_SEQUENCE)
    total_placed_phase2 = 0
    cycle = 0
    while cycles_without_placement < len(COLOR_SEQUENCE) and cycle < max_color_cycles:
        cycle += 1
        target_color_to_place = COLOR_SEQUENCE[current_target_color_index % len(COLOR_SEQUENCE)]
        required_parent_color = get_required_parent_color(target_color_to_place)
        print(f"\nColor Cycle {cycle}, Target: Place {target_color_to_place} onto {required_parent_color}")
        # 1. Find candidate exposed vertices of the required parent color
        parent_vertices_candidates = []
        # Use a copy in case vertex_info is modified during placement attempts within the loop
        current_vertex_info_snapshot = vertex_info.copy()
        for V, info in current_vertex_info_snapshot.items():
            # Check if vertex is still exposed and matches required parent color
            if info['parent_color'] == required_parent_color:
                 # Check parent_id is valid (important if cubes get removed, though not currently implemented)
                 if 0 <= info['parent_id'] < len(placed_cubes):
                     # Verify the color in placed_cubes matches (sanity check)
                     if placed_cubes[info['parent_id']][4] == required_parent_color:
                         parent_vertices_candidates.append({
                             'vertex': V,
                             'parent_id': info['parent_id'],
                             'parent_size': info['parent_size']
                         })
                     # else: # Mismatch color - data inconsistency?
                     #    print(f"Warning: Vertex info color mismatch for vertex {V}")
                 # else: # Invalid parent_id
                 #    print(f"Warning: Invalid parent_id {info['parent_id']} in vertex_info for {V}")
        if not parent_vertices_candidates:
            print(f"  No exposed vertices found for parent color {required_parent_color}.")
            cycles_without_placement += 1
            current_target_color_index += 1 # Move to next color
            continue # Skip to next color cycle
        print(f"  Found {len(parent_vertices_candidates)} potential parent vertices ({required_parent_color}).")
        # 1b. Pre-filter candidates by distance and basic bounds check (optional but can speed up)
        #     We can skip this for now and rely on the main placement check
        parent_vertices_found = []
        for candidate in parent_vertices_candidates:
             V = candidate['vertex']
             # Calculate distance for sorting
             dist = distance_to_origin(V[0], V[1], V[2]) # Use JITted version
             parent_vertices_found.append({
                 'vertex': V,
                 'distance': dist,
                 'parent_size': candidate['parent_size'],
                 'parent_id': candidate['parent_id']
             })
        # 2. Sort candidate vertices by distance from origin (ascending)
        parent_vertices_found.sort(key=lambda x: x['distance'])
        # 3. Attempt placement for each candidate vertex
        placed_this_color_cycle = 0
        processed_vertices_this_cycle = set() # Track vertices used in this cycle
        for parent_data in parent_vertices_found:
            V = parent_data['vertex']
            # Skip if vertex was already used in this cycle or is no longer exposed
            if V in processed_vertices_this_cycle or V not in vertex_info:
                continue
            parent_id = parent_data['parent_id']
            # Double-check parent_id validity before accessing placed_cubes
            if not (0 <= parent_id < len(placed_cubes)):
                print(f"Warning: Stale parent_id {parent_id} encountered for vertex {V}. Skipping.")
                continue
            parent_x, parent_y, parent_z, S_parent, p_color = placed_cubes[parent_id]
            # Sanity check: ensure parent size and color match vertex_info
            if S_parent != parent_data['parent_size'] or p_color != required_parent_color:
                 print(f"Warning: Data mismatch for parent {parent_id} at vertex {V}. Skipping.")
                 # Clean up potentially stale vertex_info entry?
                 if V in vertex_info: del vertex_info[V]
                 continue
            # Try placing cubes of decreasing size, starting from S_parent + 1
            # down to 1x1x1, taking the largest valid size.
            target_size_start = S_parent + 1
            best_S_found = 0
            placement_coords = None
            for S_try in range(target_size_start, 0, -1):
                vx, vy, vz = V # Unpack vertex coordinates
                # Calculate potential BBL coordinates for this size
                px_py_pz_tuple = get_placement_coords_at_vertex(
                    vx, vy, vz, S_try, parent_x, parent_y, parent_z, S_parent
                )
                if px_py_pz_tuple == (-9999, -9999, -9999):
                    continue # Calculation failed for this size
                px, py, pz = px_py_pz_tuple
                # Check if this placement is valid (bounds, no overlap/face/edge contact)
                # We need the current state of placed_cubes as an array
                # Note: Re-converting array inside this inner loop is inefficient.
                # Consider converting once before the parent_vertices_found loop.
                # For now, keep the conversion inside place_cube for simplicity.
                # Temporarily call place_cube to check validity (it includes the check)
                # This is slightly wasteful as it might update vertex_info prematurely
                # A better way: call is_placement_valid directly here.
                # --- Direct call to is_placement_valid ---
                if not placed_cubes:
                    temp_placed_array = np.empty((0, 4), dtype=np.int64)
                else:
                    temp_data = [(c[0], c[1], c[2], c[3]) for c in placed_cubes]
                    temp_placed_array = np.array(temp_data, dtype=np.int64)
                b_min_x = bounds_min_int[0]; b_min_y = bounds_min_int[1]; b_min_z = bounds_min_int[2]
                b_max_x = bounds_max_int[0]; b_max_y = bounds_max_int[1]; b_max_z = bounds_max_int[2]
                if is_placement_valid(px, py, pz, S_try, temp_placed_array,
                                      b_min_x, b_min_y, b_min_z,
                                      b_max_x, b_max_y, b_max_z):
                    # Found a valid size, this is the largest possible one
                    best_S_found = S_try
                    placement_coords = (px, py, pz)
                    break # Stop checking smaller sizes for this vertex
            # If a valid placement size was found, perform the actual placement
            if best_S_found > 0 and placement_coords:
                # Call place_cube WITHOUT dictionary argument to finalize
                if place_cube(placement_coords[0], placement_coords[1], placement_coords[2],
                              best_S_found, target_color_to_place):
                    placed_this_color_cycle += 1
                    total_placed_phase2 += 1 # Increment total count for Phase 2
                    # Mark this vertex as used so we don't try placing on it again this cycle
                    processed_vertices_this_cycle.add(V)
                    # Note: place_cube already updated vertex_info, potentially removing V
                # else: # Should not fail here if is_placement_valid passed, but good to check
                    # print(f"Error: place_cube failed unexpectedly after is_placement_valid passed for V={V}, S={best_S_found}")
        print(f"  Placed {placed_this_color_cycle} cubes of color {target_color_to_place}.")
        # Update logic for stopping Phase 2
        if placed_this_color_cycle > 0:
            cycles_without_placement = 0 # Reset counter if placement occurred
        else:
            cycles_without_placement += 1 # Increment counter if no cubes of this color were placed
        current_target_color_index += 1 # Move to the next color in the sequence
    # --- End Phase 2 Loop ---
    duration_phase2 = time.time() - start_time_phase2
    print(f"--- Phase 2 Complete ({duration_phase2:.2f}s in {cycle} color cycles) ---")
    print(f"Total cubes placed in Phase 2: {total_placed_phase2}") # Corrected variable name
    print(f"Total cubes after Phase 2: {len(placed_cubes)}")
    # Removed: Total occupied blocks printout
    print(f"Total unique vertices created (ever): {len(vertex_counts)}")
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
        if vertices is None or faces is None or vertex_colors is None:
            print("Mesh data generation failed or resulted in empty data.")
        else:
            # Create VisPy Canvas and View
            print("Creating VisPy scene...")
            canvas = scene.SceneCanvas(keys='interactive', show=True, title=f"3D Fractal N={N_FRAMEWORK} (Mesh - List Check)")
            view = canvas.central_widget.add_view()
            view.bgcolor = '#222222' # Dark background
            # Create Mesh visual
            print("Adding mesh visual...")
            mesh = visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, shading='flat')
            # mesh.transform = scene.transforms.STTransform(translate=[0, 0, 0], scale=[1, 1, 1]) # Optional transform
            view.add(mesh)
            # Set up the camera based on calculated bounds
            view.camera = scene.TurntableCamera(fov=60, elevation=30, azimuth=-45)
            if bounds_min_int is not None and bounds_max_int is not None and np.any(bounds_max_int > bounds_min_int):
                center_point = (bounds_min_int + bounds_max_int) / 2.0
                diag_vector = bounds_max_int - bounds_min_int
                # Handle cases where a dimension might be flat (e.g., 2D structure)
                diag_vector[diag_vector <= 0] = 1.0 # Avoid zero length for norm/distance calc
                diag_length = np.linalg.norm(diag_vector)
                view.camera.distance = max(diag_length * 1.5, 10) # Ensure minimum distance
                view.camera.center = tuple(center_point)
            else:
                 # Fallback if bounds are invalid or only one cube exists
                 view.camera.distance = 10 * (placed_cubes[0][3] if placed_cubes else 1) # Scale distance by initial cube size
                 view.camera.center = (placed_cubes[0][0]+placed_cubes[0][3]/2, placed_cubes[0][1]+placed_cubes[0][3]/2, placed_cubes[0][2]+placed_cubes[0][3]/2) if placed_cubes else (0,0,0)
            # Add XYZ Axis for orientation
            axis = visuals.XYZAxis(parent=view.scene)
            # Run the VisPy application
            print("\nStarting VisPy visualization...")
            print("Rotate: Left Mouse Button + Drag | Zoom: Scroll Wheel | Pan: Shift + Left Button + Drag")
            print("Close the VisPy window to generate another fractal or enter 'q'.")
            app.run()
            print("VisPy window closed.")
print("--- Script Finished ---")
