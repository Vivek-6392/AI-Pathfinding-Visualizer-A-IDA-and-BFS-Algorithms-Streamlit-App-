# 🧭 AI Pathfinding Visualizer

An interactive **AI Pathfinding Visualizer** built using **Python** and **Streamlit**, showcasing classic AI search algorithms like **A\***, **IDA\***, and **BFS**.  
This mini-project demonstrates how different algorithms explore nodes and find the shortest path in a 2D grid with obstacles.

---

## 🚀 Features

- 🧠 **Multiple Algorithms**
  - A* (A-Star)
  - IDA* (Iterative Deepening A*)
  - BFS (Breadth-First Search)

- 🎨 **Interactive Visualization**
  - Click to add/remove walls (obstacles)
  - Choose start and end nodes
  - Watch the algorithm explore in real time

- 📊 **Statistics Display**
  - Nodes explored
  - Path length
  - Execution time

- ⚙️ **Customizable Grid**
  - Adjustable grid size
  - Variable animation speed

---

## 🖼️ Visual Meaning of Tiles

| **Color** | **Represents** |
|------------|----------------|
| 🟩 Green | Start Node |
| 🟥 Red | End Node |
| ⬛ Black | Wall / Obstacle |
| 🟦 Blue | Explored Nodes |
| 🟨 Yellow | Final Shortest Path |

---

## 🧩 Algorithms Explained

### 🔹 A* (A-Star)
A heuristic-based search algorithm that uses the formula:
f(n) = g(n) + h(n)

yaml
Copy code
- `g(n)` = Cost from start to node `n`
- `h(n)` = Heuristic (estimated cost to goal)
- Fast and optimal for most use cases

---

### 🔹 IDA* (Iterative Deepening A*)
A memory-efficient version of A*, combining **depth-first search** with **iterative cost thresholds**.  
Trades speed for significantly reduced memory use.

---

### 🔹 BFS (Breadth-First Search)
Explores all possible nodes level by level.  
Guarantees shortest path in unweighted grids but can be slow for large maps.

---

## 🧰 Installation & Run Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/AI-Pathfinding-Visualizer.git
cd AI-Pathfinding-Visualizer
2️⃣ Install dependencies
bash
Copy code
pip install streamlit numpy pillow
3️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
(Replace app.py with your main file name, e.g., ai_pathfinder_app.py)

🧮 Example Output
Algorithm: A*

Nodes explored: 130

Path length: 18

Execution time: 0.05s

Each algorithm shows different exploration behavior and efficiency.

🧠 Educational Use
This project is perfect for:

Understanding AI search algorithms

Comparing performance and efficiency

Visual learning of pathfinding concepts

💡 Future Enhancements
Add Dijkstra’s algorithm

Add Greedy Best-First Search

Save and load grid layouts

Add diagonal movement support

Dark/light mode toggle

🧑‍💻 Author
Vivek Yadav
📚 Mini Project – Artificial Intelligence
Built using Python, Streamlit, and NumPy

🪪 License
This project is licensed under the MIT License – you’re free to use, modify, and distribute it for educational purposes.

🌟 Acknowledgements
Inspired by visualizers from:

Pathfinding Visualizer (Clement Mihailescu)

