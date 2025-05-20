from itertools import islice, product
import xml.etree.ElementTree as ET
import networkx as nx
import sumolib
import numpy as np
import csv
from xml.dom import minidom
from itertools import permutations


# BPR函数的参数
alpha = 0.15
beta = 4
min_percentage = 0.2
max_percentage = 0.8

netfile_name = r"E:\研究生资料\论文\实验数据文件\实验\map.net.xml"
path_file = r"E:\研究生资料\论文\实验数据文件\实验\paths.csv"
edge_capacities_file = r"E:\研究生资料\论文\实验数据文件\实验\edge_capacities.csv"

# 10&10
try_trip = r"E:\研究生资料\论文\实验数据文件\实验\trytrip_10.xml"
try_rou_file = r"E:\研究生资料\论文\实验数据文件\实验\new\try.rou_10.xml"
car = 10

# #10&20
# try_trip = r"D:\新建文件夹\Open_Graph\10_20\trytrip_20.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_20.xml"
# car = 20

# #10&30
# try_trip = r"D:\新建文件夹\Open_Graph\10_30\trytrip_30.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_30.xml"
# car = 30

# #10&40
# try_trip = r"D:\新建文件夹\Open_Graph\10_40\trytrip_40.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_40.xml"
# car = 40

# #10&50
# try_trip = r"D:\新建文件夹\Open_Graph\10_50\trytrip_50.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_50.xml"
# car = 50

# #10&60s
# try_trip = r"D:\新建文件夹\Open_Graph\10_60\trytrip_60.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_60.xml"
# car = 60

# #10&70
# try_trip = r"D:\新建文件夹\Open_Graph\10_70\trytrip_70.xml"
# try_rou_file = r"D:\新建文件夹\new\try.rou_70.xml"
# car = 70

# #10&80
# try_trip = r"E:\研究生资料\论文\实验数据文件\实验\trytrip_80.xml"
# try_rou_file = r"E:\研究生资料\论文\实验数据文件\实验\new\try.rou_80.xml"
# car = 80

# #10&90
# try_trip = r"E:\论文\实验数据文件\实验\trytrip_90.xml"
# try_rou_file = r"E:\论文\实验数据文件\实验\new\try.rou_90.xml"
# car = 90

# #10&100
# try_trip = r"E:\论文\实验数据文件\实验\trytrip_100.xml"
# try_rou_file = r"E:\论文\实验数据文件\实验\new\try.rouall_100.xml"
# car = 100


def generate_network_from_sumo(sumo_net_file):
    net = sumolib.net.readNet(sumo_net_file)
    graph = nx.DiGraph()  # 创建有向图

    edge_id_to_time = {}  # 创建字典来存储 edge_id 到 time 的映射

    for edge in net.getEdges():  # 添加边和权重
        for lane in edge.getLanes():
            lane_attributes = lane._allowed
            if check_disallowed(lane_attributes):
                break
            else:
                edge_length = edge.getLength()
                edge_speed = lane.getSpeed()
                edge_time = edge_length / edge_speed
                edge_id = edge.getID()
                graph.add_edge(edge.getFromNode().getID(), edge.getToNode().getID(), weight=edge.getLength(),
                               id=edge_id, time=edge_time)
                edge_id_to_time[edge_id] = edge_time  # 将 edge_id 和 time 存入字典
                break  # 一旦找到允许车辆的车道，就可以跳出循环

    return graph, edge_id_to_time


def check_disallowed(lane_attributes):
    return "private" not in lane_attributes


def parse_trip_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    car_trip_data = []
    for item in root.findall('trip'):
        start_edge_id = item.get('from')
        end_edge_id = item.get('to')
        car_trip_data.append((start_edge_id, end_edge_id))

    return car_trip_data


def get_edge_ids_on_path(graph, path):
    edge_ids = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        if graph.has_edge(source, target):
            edge_ids.append(graph[source][target]['id'])
    return edge_ids


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / float(union)


# 策略包含（5，0，0）这种
def generate_strategies(total_vehicles, paths):
    return [tuple(strategy) for strategy in product(range(total_vehicles + 1), repeat=paths) if
            sum(strategy) == total_vehicles]


# 生成的策略每条路都有车辆选择
def generate_car_strategies(total_vehicles, paths):
    strategies = []
    for strategy in product(range(total_vehicles + 1), repeat=paths):
        if sum(strategy) == total_vehicles and all(v > 0 for v in strategy):
            strategies.append(tuple(strategy))
    return strategies


def generate_distributions(total_vehicles, num_groups, min_percentage, max_percentage):
    min_vehicles = int(total_vehicles * min_percentage)
    max_vehicles = int(total_vehicles * max_percentage)
    valid_distributions = []

    def backtrack(remaining, group, current_distribution):
        if group == num_groups - 1:
            if min_vehicles <= remaining <= max_vehicles:
                current_distribution.append(remaining)
                # Ensure all permutations of the current distribution are considered
                for perm in permutations(current_distribution):
                    if perm not in valid_distributions:
                        valid_distributions.append(perm)
                current_distribution.pop()
            return

        for i in range(min_vehicles, max_vehicles + 1):
            if i <= remaining - min_vehicles * (num_groups - group - 1):
                current_distribution.append(i)
                backtrack(remaining - i, group + 1, current_distribution)
                current_distribution.pop()

    backtrack(total_vehicles, 0, [])
    return valid_distributions


# def calculate_path_time(graph, edge_id_to_time, paths):
#     total_times = []
#     for path in paths:
#         total_time = 0
#         for edge_id in path:
#             if edge_id in edge_id_to_time:
#                 total_time += edge_id_to_time[edge_id]
#             else:
#                 print(f"Edge {edge_id} not found in edge_id_to_time dictionary.")
#         total_times.append(total_time)
#     return total_times


def create_rou_file(path_edge_ids, file_name):
    with open(file_name, 'w') as f:
        f.write('<routes>\n')
        f.write('<vType id="passenger" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70"/>\n')

        for num, route_content in enumerate(path_edge_ids):
            route_content = ' '.join(route_content)
            f.write(f'    <vehicle id="v{num}" depart="0" color="1,0,0" type="passenger">\n')
            f.write(f'        <route edges="{route_content}"/>\n')
            f.write('    </vehicle>\n')

        f.write('</routes>\n')


def calculate_all_paths_time(graph, edge_id_to_time, edges):
    # 初始化一个二维数组来存储每条路径的行驶时间
    path_times = np.zeros((len(edges), len(edges[0])))

    for i, paths in enumerate(edges):
        for j, path in enumerate(paths):
            total_time = 0
            for edge_id in path:
                if edge_id in edge_id_to_time:
                    total_time += edge_id_to_time[edge_id]
                else:
                    print(f"Edge {edge_id} not found in edge_id_to_time dictionary.")
            path_times[i][j] = total_time

    return path_times


# BPR函数
def bpr_function(t0, v, C, alpha=0.15, beta=4):
    return t0 * (1 + alpha * (v / C) ** beta)


def generate_strategies(total_vehicles, paths):
    return [tuple(strategy) for strategy in product(range(total_vehicles + 1), repeat=paths) if sum(strategy) == total_vehicles]


def save_paths_to_csv(paths, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Path1', 'Path2', 'Path3'])
        writer.writerows(paths)


def read_paths_from_csv(input_file):
    paths = []
    with open(input_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            paths.append(row)
    return paths


# 统计每个路段上的车辆个数
def count_vehicles(paths, choices):
    edge_vehicle_count = {}
    for path, choice in zip(paths, choices):
        for edge in path:
            if edge not in edge_vehicle_count:
                edge_vehicle_count[edge] = 0
            # print(edge, choice, edge_vehicle_count[edge])
            edge_vehicle_count[edge] += choice
    return edge_vehicle_count


# 读取车道容量信息
def read_lane_capacities(filename):
    lane_capacities = {}
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            if row:
                lane_id = row[0].strip()
                lane_capacity = float(row[1].strip())
                lane_capacities[lane_id] = lane_capacity
    return lane_capacities


def calculate_bpr_time(t0, V, C):
    """计算BPR函数得到的通行时长"""
    return t0 * (1 + alpha * (V / C) ** beta)


def calculate_path_time(path, edge_id_to_time, edge_capacities, vehicle_counts):
    """计算路径的总通行时长"""
    total_time = 0
    for edge in path:
        t0 = edge_id_to_time[edge]
        C = edge_capacities[edge]
        V = vehicle_counts[edge]
        # print("t0:", t0)
        # print("C:", C)
        # print("V:", V)
        total_time += calculate_bpr_time(t0, V, C)
        # print(total_time)
    return total_time


def calculate_utility(time):
    return -time


def calculate_total_utility(strategy, base_path_times):
    total_time = 0.0
    times = []

    # 计算加权的行驶总时间和收集各车辆的行驶时间
    for i, num_vehicles in enumerate(strategy):
        actual_time = base_path_times[i]
        total_time += actual_time * num_vehicles
        times.extend([actual_time] * num_vehicles)  # 将每辆车的时间加入到 times 列表

    # 计算行驶时间的标准方差
    stdev = np.std(times)

    # 计算总收益：70% 的行驶总时间 + 30% 的行驶时间标准方差
    total_utility = -(0.7 * total_time + 0.3 * stdev)
    return total_utility


def replicator_dynamics(strategy_prob, payoff_matrix, num_iterations=100, learning_rate=0.01):
    num_strategies = len(strategy_prob)
    for _ in range(num_iterations):
        avg_payoff = np.dot(strategy_prob, payoff_matrix)  # 当前策略组合的平均收益
        for i in range(num_strategies):
            payoff_diff = np.dot(payoff_matrix[i], strategy_prob) - np.dot(avg_payoff, strategy_prob)
            strategy_prob[i] += learning_rate * strategy_prob[i] * payoff_diff
        strategy_prob = np.clip(strategy_prob, 0, 1)  # 确保概率在0到1之间
        total_prob = np.sum(strategy_prob)
        if total_prob == 0:
            # 处理总和为0的情况，例如，分配均匀概率或采取其他措施
            strategy_prob = np.ones_like(strategy_prob) / len(strategy_prob)
        else:
            strategy_prob /= total_prob  # 归一化，使总概率为1

        strategy_prob /= np.sum(strategy_prob)  # 归一化，使总概率为1
    return strategy_prob


def create_vehicle_element(vehicle_id, path, depart_time="0.00"):
    vehicle = ET.Element("vehicle", id=str(vehicle_id), depart=depart_time)
    route = ET.SubElement(vehicle, "route", edges=path)
    return vehicle


def generate_route_file(vehicle_distributions, paths, filename):
    root = ET.Element("routes", attrib={
        'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
        'xsi:noNamespaceSchemaLocation': "http://sumo.dlr.de/xsd/routes_file.xsd"
    })

    vehicle_id = 1
    for dis_index, distribution in enumerate(vehicle_distributions):
        print(distribution)
        for path_index, num_vehicles in enumerate(distribution):
            print(path_index, num_vehicles)
            for _ in range(num_vehicles):
                print(_)
                print(', '.join(paths[dis_index][path_index]))
                vehicle = create_vehicle_element(vehicle_id, ' '.join(paths[dis_index][path_index]))
                root.append(vehicle)
                vehicle_id += 1

    # Convert to a string and prettify
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(xml_str)
    pretty_str = parsed.toprettyxml(indent="    ")

    # Write to file
    with open(filename, 'w') as f:
        f.write(pretty_str)


# 主程序
def main():
    # 生成网络结构
    graph, edge_id_to_time = generate_network_from_sumo(netfile_name)
    # 获取节点和边的数量
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    print(f"节点数: {num_nodes}")
    print(f"边数: {num_edges}")

    # 创建映射(将G图中的edge——from——to与SUMO中的edge_id进行对应)
    edge_to_nodes = {(data['id']): (u, v) for (u, v, data) in graph.edges(data=True)}

    # 读取trip文件，获取到所有的起始点信息
    trip_data = parse_trip_xml(try_trip)

    # 存储所有的路径信息
    path_edges = []
    paths = []
    nodeset_list = []
    for trip in trip_data:
        source_edge_id, target_edge_id = trip  # 解包元组获取起点和终点

        # 获取起点和终点节点的ID
        source_node_id = edge_to_nodes[source_edge_id][0]
        target_node_id = edge_to_nodes[target_edge_id][1]

        # 获取不同算法的最短路径
        dijkstra_path = nx.dijkstra_path(graph, source=source_node_id, target=target_node_id, weight='weight')
        bfs_tree = nx.bfs_tree(graph, source=source_node_id)
        bfs_path = nx.shortest_path(bfs_tree, source=source_node_id, target=target_node_id)
        # dfs_tree = nx.dfs_tree(graph, source=source_node_id)
        # dfs_path = nx.shortest_path(dfs_tree, source=source_node_id, target=target_node_id)
        # 使用 Bellman-Ford 算法计算最短路径
        # bellman_ford_path = nx.bellman_ford_path(graph, source=source_node_id, target=target_node_id, weight='weight')
        # 使用 K 最短路径算法
        k_shortest_paths = list(
            islice(nx.shortest_simple_paths(graph, source=source_node_id, target=target_node_id, weight='weight'), 4))

        # 打印路径信息
        # print(f"Shortest path from {source_edge_id} to {target_edge_id} is: {dijkstra_path}")
        # print(f"Shortest path from {source_edge_id} to {target_edge_id} is: {bfs_path}")
        # print(f"Shortest path from {source_edge_id} to {target_edge_id} is: {k_shortest_paths[3]}")

        # 获取路径上的边ID
        path_edges.append(get_edge_ids_on_path(graph, dijkstra_path))
        path_edges.append(get_edge_ids_on_path(graph, bfs_path))
        path_edges.append(get_edge_ids_on_path(graph, k_shortest_paths[1]))

        paths.append(path_edges)
        path_edges = []
        # 创建一个集合，将这三个路径的节点都加进去
        node_set = set(
            get_edge_ids_on_path(graph, dijkstra_path) + get_edge_ids_on_path(graph, bfs_path) + get_edge_ids_on_path(
                graph, k_shortest_paths[1]))
        # print(len(node_set))
        nodeset_list.append(node_set)

    # print(paths)
    # 调用方法保存路径
    # save_paths_to_csv(paths, path_file)

    # 计算路径时间并存储为二维数组
    path_times = calculate_all_paths_time(graph, edge_id_to_time, paths)

    # 输出所有路径的总行车时间
    # print(f"所有路径的总行车时间为: \n{path_times}")

    # 计算雅卡尔相似性
    similarity_matrix = np.zeros((len(nodeset_list), len(nodeset_list)))
    for i in range(len(nodeset_list)):
        for j in range(len(nodeset_list)):
            similarity_matrix[i][j] = jaccard_similarity(nodeset_list[i], nodeset_list[j])
    # print(similarity_matrix)

    # 找到每行最大值的索引
    max_index_dict = {}
    for i in range(similarity_matrix.shape[0]):
        max_num = 0
        num = 0
        for j in range(len(similarity_matrix[i])):
            if i != j and max_num < similarity_matrix[i][j]:
                max_num = similarity_matrix[i][j]
                num = j
        max_index_dict[i] = num
    # print(max_index_dict)

    # 生成所有可能的策略
    # strategies = generate_distributions(car, 3, min_percentage, max_percentage)
    strategies = generate_car_strategies(car, 3)
    print(f"共有 {len(strategies)} 种策略")
    print(strategies)

    edge_capacities = read_lane_capacities(edge_capacities_file)

    # paths = read_paths_from_csv(r"D:\SUMOProject\dalianMap\Open_Graph\paths.csv")

    # 打印路径信息
    # for path in paths:
    #     for x in path:
    #         print(x)

    # 计算收益函数数组
    array_2d_A = [[0 for _ in range(len(strategies))] for _ in range(len(strategies))]
    array_2d_B = [[0 for _ in range(len(strategies))] for _ in range(len(strategies))]
    game = []
    for A, B in max_index_dict.items():
        for i, GroupA in enumerate(strategies):
            for j, GroupB in enumerate(strategies):
                # 计算车辆组1的车流量
                # print(paths[0])
                group1_vehicle_count = count_vehicles(paths[A], GroupA)
                # 计算车辆组2的车流量
                group2_vehicle_count = count_vehicles(paths[B], GroupB)

                # print(len(group1_vehicle_count), len(group2_vehicle_count))
                # 合并两个车辆组的车流量
                total_vehicle_count = group1_vehicle_count.copy()
                for edge, count in group2_vehicle_count.items():
                    if edge in total_vehicle_count:
                        total_vehicle_count[edge] += count
                    else:
                        total_vehicle_count[edge] = count

                # 计算每条路径的总通行时长
                path_time_strategies_A = [calculate_path_time(path, edge_id_to_time, edge_capacities,
                                                              total_vehicle_count) for path in paths[A]]
                path_time_strategies_B = [calculate_path_time(path, edge_id_to_time, edge_capacities,
                                                              total_vehicle_count) for path in paths[A]]
                # print(path_time_strategies_A, path_time_strategies_B)
                array_2d_A[i][j] = calculate_total_utility(GroupA, path_time_strategies_A)
                array_2d_B[i][j] = calculate_total_utility(GroupB, path_time_strategies_B)

        # 初始策略概率分布
        strategy_prob_A = np.ones(len(strategies)) / len(strategies)
        strategy_prob_B = np.ones(len(strategies)) / len(strategies)
        # 运行演化博弈，更新策略概率
        num_iterations = 100
        learning_rate = 0.01
        strategy_prob_A = replicator_dynamics(strategy_prob_A, array_2d_A, num_iterations, learning_rate)
        strategy_prob_B = replicator_dynamics(strategy_prob_B, array_2d_B, num_iterations, learning_rate)

        # 获取最小值的下标
        max_index_A = np.argmax(strategy_prob_A)
        max_index_B = np.argmax(strategy_prob_B)
        # print(strategies)
        game.append(strategies[max_index_A])
        # print(strategies[max_index_A])
        # print(strategies[max_index_B])
        # print("Updated strategy probabilities for Group A:", strategy_prob_A)
        # print("Updated strategy probabilities for Group B:", strategy_prob_B)

    print(game)
    # 生成SUMO路线文件
    generate_route_file(game, paths, try_rou_file)


# 运行主程序
main()

