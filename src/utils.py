CEFR_Descriptors = {
    1: "Can understand very short, simple texts a single phrase at a time, picking up familiar names, words and basic phrases and rereading as required.",
    2: "Can understand short, simple texts containing the highest frequency vocabulary, including a proportion of shared international vocabulary items.",
    3: "Can read straightforward factual texts on subjects related to their field of interest with a satisfactory level of comprehension.",
    4: "Can read with a large degree of independence, adapting style and speed of reading to different texts and purposes, and using appropriate reference sources selectively. Has a broad active reading vocabulary, but may experience some difficulty with low-frequency idioms.",
    5: "Can understand a wide variety of texts including literary writings, newspaper or magazine articles, and specialised academic or professional publications, provided there are opportunities for rereading and they have access to reference tools. Can understand in detail lengthy, complex texts, whether or not these relate to their own area of speciality, provided they can reread difficult sections.",
    6: "Can understand a wide range of long and complex texts, appreciating subtle distinctions of style and implicit as well as explicit meaning. Can understand virtually all types of texts including abstract, structurally complex, or highly colloquial literary and non-literary writings."
}


def agent_policy(nodes, profit_matrix, start_node, end_node):
    """
    计算从指定起点到指定终点的最大收益路径。

    参数:
        nodes (list): 节点列表，例如 ['a', 'b', 'c', 'd']。
        profit_matrix (list of lists): 收益矩阵，profit_matrix[i][j] 表示从节点 i 到节点 j 的收益。
        start_node (str): 起点节点名称。
        end_node (str): 终点节点名称。

    返回:
        max_profit (float): 最大收益。
        path (list): 最大收益路径。
    """
    # 获取节点索引
    node_index = {node: idx for idx, node in enumerate(nodes)}
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]

    n = len(nodes)
    if n == 0 or not profit_matrix:
        return 0, []

    # 动态规划表，dp[i] 表示从起点到节点 i 的最大收益
    dp = [float('-inf')] * n
    # 路径记录表，记录到达每个节点的前一个节点
    path = [None] * n

    # 起点初始化
    dp[start_idx] = 0.0  # 起点的初始收益为 1.0（可以调整为其他值）

    # 动态规划计算最大收益
    for i in range(n):
        for j in range(i+1, n):  # 只能从前面的节点到后面的节点
            if profit_matrix[i][j] != 0 and dp[i] != float('-inf'):  # 如果有收益
                new_profit = dp[i] + profit_matrix[i][j]
                if new_profit > dp[j]:  # 更新最大收益
                    dp[j] = new_profit
                    path[j] = i

    # 回溯路径
    current = end_idx
    result_path = [nodes[current]]
    while path[current] is not None:
        current = path[current]
        result_path.append(nodes[current])
    result_path.reverse()

    return dp[end_idx], result_path
