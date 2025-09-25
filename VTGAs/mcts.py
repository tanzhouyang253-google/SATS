import osmnx as ox
import networkx as nx
import random
import math

# 参数配置
MAX_ITERATIONS = 1000
MAX_SIM_STEPS = 20
TERMINAL_DISTANCE = 30  # 米为单位

def haversine_distance(n1, n2, G):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    return math.hypot(x1 - x2, y1 - y2) * 111000  # approx 米

# MCTS Node
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = []
        self.action_from_parent = None

    def best_child(self, c_param=1.0):
        best_score = -float('inf')
        best_node = None
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits + 1) / child.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, G):
        if not self.untried_actions:
            self.untried_actions = list(G.neighbors(self.state))
        while self.untried_actions:
            new_state = self.untried_actions.pop()
            if new_state not in [c.state for c in self.children]:
                child = Node(new_state, parent=self)
                self.children.append(child)
                return child
        return None

def simulate(state, G, target):
    path = [state]
    visited = {state}
    for _ in range(MAX_SIM_STEPS):
        neighbors = [n for n in G.neighbors(path[-1]) if n not in visited]
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        if haversine_distance(next_node, target, G) < TERMINAL_DISTANCE:
            break
    dist = haversine_distance(path[-1], target, G)
    return -dist

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def extract_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))

def mcts(G, start, target):
    root = Node(start)
    for _ in range(MAX_ITERATIONS):
        node = root
        # 选择
        while node.children:
            node = node.best_child()
        # 扩展
        child = node.expand(G)
        node_to_simulate = child if child else node
        # 模拟
        reward = simulate(node_to_simulate.state, G, target)
        # 回溯
        backpropagate(node_to_simulate, reward)
        # 提前终止
        if -reward < TERMINAL_DISTANCE:
            print("[INFO] 提前找到目标附近路径")
            return extract_path(node_to_simulate)
    print("[WARN] 未找到目标点附近路径")
    return extract_path(root.best_child() or root)

def generate_virtual_trajectory(start_xy, end_xy, mode, radius=2000,):
    """
    输入: 起点和终点坐标 [lng, lat]
    输出: [[lng1, lat1], [lng2, lat2], ...]
    """
    print("[INFO] 加载路网中...")
    G = ox.graph_from_point(start_xy[::-1], dist=radius, network_type=mode)
    start_node = ox.distance.nearest_nodes(G, X=start_xy[0], Y=start_xy[1])
    end_node = ox.distance.nearest_nodes(G, X=end_xy[0], Y=end_xy[1])
    print(f"[INFO] 起点: {start_node}, 终点: {end_node}")
    
    path_nodes = mcts(G, start_node, end_node)
    coords = [[G.nodes[n]['x'], G.nodes[n]['y']] for n in path_nodes]  # 输出为[[lng, lat]]
    
    print("[RESULT] 虚拟轨迹:")
    for pt in coords:
        print(pt)
    return coords


if __name__ == "__main__":
    generate_virtual_trajectory([116.47157,39.89802], [116.46844,39.89926], 'walk')
