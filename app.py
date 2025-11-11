"""
Pathfinding Comparison Dashboard (Streamlit)

- Edit grid in the table (0 = free, 1 = wall)
- Set Start / End coordinates (x,y)
- Generate random maze
- Run A*, Dijkstra, IDA* and compare results side-by-side
- Includes color legend for visualization
"""

import streamlit as st
import numpy as np
import heapq
import time
from collections import deque, namedtuple
from PIL import Image, ImageDraw
import pandas as pd
import random

Point = namedtuple("Point", ["x", "y"])

# -------------------------
# Pathfinding implementations
# -------------------------
def manhattan(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def neighbors(p, w, h):
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = p.x + dx, p.y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield Point(nx, ny)

def astar_grid(start, goal, grid):
    """Return (path_list or None, explored_set, nodes_expanded)"""
    w, h = grid.shape[1], grid.shape[0]
    open_heap = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    explored = set()
    nodes_expanded = 0

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path)), explored, nodes_expanded

        for n in neighbors(current, w, h):
            if grid[n.y, n.x] == 1 and n != goal:
                continue
            tentative_g = g_score[current] + 1
            if n not in g_score or tentative_g < g_score[n]:
                g_score[n] = tentative_g
                f = tentative_g + manhattan(n, goal)
                came_from[n] = current
                heapq.heappush(open_heap, (f, tentative_g, n))
    return None, explored, nodes_expanded

def dijkstra_grid(start, goal, grid):
    """Dijkstra = A* with zero heuristic."""
    w, h = grid.shape[1], grid.shape[0]
    open_heap = []
    heapq.heappush(open_heap, (0, 0, start))
    came_from = {}
    dist = {start: 0}
    explored = set()
    nodes_expanded = 0

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path)), explored, nodes_expanded

        for n in neighbors(current, w, h):
            if grid[n.y, n.x] == 1 and n != goal:
                continue
            nd = g + 1
            if n not in dist or nd < dist[n]:
                dist[n] = nd
                came_from[n] = current
                heapq.heappush(open_heap, (nd, nd, n))
    return None, explored, nodes_expanded

def ida_star_grid(start, goal, grid, max_iters=1000000):
    w, h = grid.shape[1], grid.shape[0]
    threshold = manhattan(start, goal)
    path = [start]
    explored = set()
    nodes_expanded = 0
    iterations = 0

    def search(g, bound):
        nonlocal nodes_expanded, iterations
        node = path[-1]
        iterations += 1
        if iterations > max_iters:
            return float('inf'), False
        f = g + manhattan(node, goal)
        if f > bound:
            return f, False
        if node == goal:
            return True, True
        minimum = float('inf')
        explored.add(node)
        nodes_expanded += 1
        for n in neighbors(node, w, h):
            if grid[n.y, n.x] == 1 and n != goal:
                continue
            if n in path:
                continue
            path.append(n)
            t, found = search(g + 1, bound)
            if found:
                return True, True
            if isinstance(t, (int, float)) and t < minimum:
                minimum = t
            path.pop()
        return minimum, False

    while True:
        t, found = search(0, threshold)
        if found:
            return list(path), explored, nodes_expanded
        if t == float('inf'):
            return None, explored, nodes_expanded
        threshold = t

# -------------------------
# Rendering utilities
# -------------------------
def render_grid(grid, path=None, explored=None, start=None, goal=None,
                cell_px=20, show_grid=True):
    h, w = grid.shape
    img = Image.new("RGB", (w * cell_px, h * cell_px), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Colors
    color_free = (220, 220, 220)
    color_wall = (15, 15, 15)
    color_path = (255, 255, 0)      # Yellow path
    color_explored = (100, 150, 255)  # Blue explored
    color_start = (0, 255, 0)       # Green start
    color_goal = (255, 0, 0)        # Red goal

    for y in range(h):
        for x in range(w):
            x0, y0 = x * cell_px, y * cell_px
            x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1
            if grid[y, x] == 1:
                draw.rectangle([x0, y0, x1, y1], fill=color_wall)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color_free)

    if explored:
        for p in explored:
            if p == start or p == goal:
                continue
            draw.rectangle([p.x * cell_px, p.y * cell_px,
                            p.x * cell_px + cell_px - 1, p.y * cell_px + cell_px - 1],
                           fill=color_explored)

    if path:
        for p in path:
            if p == start or p == goal:
                continue
            draw.rectangle([p.x * cell_px, p.y * cell_px,
                            p.x * cell_px + cell_px - 1, p.y * cell_px + cell_px - 1],
                           fill=color_path)

    # Start / Goal
    if start:
        draw.rectangle([start.x * cell_px, start.y * cell_px,
                        start.x * cell_px + cell_px - 1, start.y * cell_px + cell_px - 1],
                       fill=color_start)
    if goal:
        draw.rectangle([goal.x * cell_px, goal.y * cell_px,
                        goal.x * cell_px + cell_px - 1, goal.y * cell_px + cell_px - 1],
                       fill=color_goal)

    # Grid lines
    if show_grid:
        for gx in range(0, w * cell_px, cell_px):
            draw.line([(gx, 0), (gx, h * cell_px)], fill=(50, 50, 50))
        for gy in range(0, h * cell_px, cell_px):
            draw.line([(0, gy), (w * cell_px, gy)], fill=(50, 50, 50))

    return img

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config("Pathfinding Comparison", layout="wide")
st.title("🔬 Pathfinding Comparison Dashboard — A*, Dijkstra, IDA*")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    cols = st.columns(2)
    with cols[0]:
        grid_w = st.number_input("Grid width", 8, 60, 20)
    with cols[1]:
        grid_h = st.number_input("Grid height", 8, 60, 20)
    cell_px = st.slider("Cell pixel size", 8, 40, 20)
    show_grid_lines = st.checkbox("Show grid lines", True)
    random_fill = st.slider("Random wall density (%)", 0, 60, 25)
    st.divider()
    start_x = st.number_input("Start X", 0, grid_w - 1, 1)
    start_y = st.number_input("Start Y", 0, grid_h - 1, 1)
    end_x = st.number_input("End X", 0, grid_w - 1, grid_w - 2)
    end_y = st.number_input("End Y", 0, grid_h - 1, grid_h - 2)
    st.divider()
    btn_random = st.button("Generate Random Maze")
    btn_clear = st.button("Clear Walls")
    btn_run = st.button("Run Algorithms")

# Initialize session grid
if 'grid_df' not in st.session_state or st.session_state.get('grid_shape') != (grid_h, grid_w):
    arr = np.zeros((grid_h, grid_w), dtype=int)
    st.session_state['grid_df'] = pd.DataFrame(arr)
    st.session_state['grid_shape'] = (grid_h, grid_w)
    st.session_state['results'] = {}

# Actions
if btn_random:
    arr = (np.random.rand(grid_h, grid_w) < (random_fill / 100)).astype(int)
    arr[start_y, start_x] = 0
    arr[end_y, end_x] = 0
    st.session_state['grid_df'] = pd.DataFrame(arr)
    st.session_state['results'] = {}
if btn_clear:
    arr = np.zeros((grid_h, grid_w), dtype=int)
    st.session_state['grid_df'] = pd.DataFrame(arr)
    st.session_state['results'] = {}

st.markdown("### 🧩 Editable Grid (0 = free, 1 = wall)")
grid_df = st.data_editor(st.session_state['grid_df'], num_rows="dynamic", use_container_width=True)
grid_arr = (grid_df.to_numpy(dtype=int) != 0).astype(int)
st.session_state['grid_df'] = pd.DataFrame(grid_arr)

start = Point(int(start_x), int(start_y))
goal = Point(int(end_x), int(end_y))
grid_arr[start.y, start.x] = 0
grid_arr[goal.y, goal.x] = 0

# Run algorithms
if btn_run:
    results = {}
    for name, func in [("A*", astar_grid), ("Dijkstra", dijkstra_grid), ("IDA*", ida_star_grid)]:
        t0 = time.perf_counter()
        path, explored, nodes = func(start, goal, grid_arr)
        t1 = time.perf_counter()
        results[name] = {
            'path': path, 'explored': explored, 'time': t1 - t0,
            'nodes': nodes, 'path_len': len(path) if path else None
        }
    st.session_state['results'] = results

results = st.session_state.get('results', {})

col1, col2, col3 = st.columns(3)
for col, algo in zip((col1, col2, col3), ["A*", "Dijkstra", "IDA*"]):
    with col:
        st.subheader(algo)
        res = results.get(algo)
        img = render_grid(grid_arr, 
                          path=res['path'] if res else None, 
                          explored=res['explored'] if res else None, 
                          start=start, goal=goal, cell_px=cell_px, 
                          show_grid=show_grid_lines)
        st.image(img, use_container_width=True)
        if res:
            st.markdown(f"**Time:** {res['time']*1000:.2f} ms  \n**Nodes:** {res['nodes']}  \n**Path len:** {res['path_len']}")
        else:
            st.caption("_No run yet_")

# Legend
st.markdown("### 🎨 Color Legend")
legend_cols = st.columns(6)
colors = {
    "Free cell": "#DCDCDC",
    "Wall": "#0F0F0F",
    "Explored": "#6496FF",
    "Path": "#FFFF00",
    "Start": "#00FF00",
    "Goal": "#FF0000"
}
for (label, color), c in zip(colors.items(), legend_cols):
    c.markdown(
        f"""
        <div style="display:flex;align-items:center;">
            <div style="width:25px;height:25px;background:{color};border:1px solid #444;margin-right:8px;"></div>
            <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption("🧠 Tip: Edit the grid, generate random mazes, and compare A*, Dijkstra, and IDA* performance visually.")
