CEFR_Descriptors = {
    1: "Can understand very short, simple texts a single phrase at a time, picking up familiar names, words and basic phrases and rereading as required.",
    2: "Can understand short, simple texts containing the highest frequency vocabulary, including a proportion of shared international vocabulary items.",
    3: "Can read straightforward factual texts on subjects related to their field of interest with a satisfactory level of comprehension.",
    4: "Can read with a large degree of independence, adapting style and speed of reading to different texts and purposes, and using appropriate reference sources selectively. Has a broad active reading vocabulary, but may experience some difficulty with low-frequency idioms.",
    5: "Can understand a wide variety of texts including literary writings, newspaper or magazine articles, and specialised academic or professional publications, provided there are opportunities for rereading and they have access to reference tools. Can understand in detail lengthy, complex texts, whether or not these relate to their own area of speciality, provided they can reread difficult sections.",
    6: "Can understand a wide range of long and complex texts, appreciating subtle distinctions of style and implicit as well as explicit meaning. Can understand virtually all types of texts including abstract, structurally complex, or highly colloquial literary and non-literary writings."
}


def agent_policy(nodes, profit_matrix, start_node, end_node):
    node_index = {node: idx for idx, node in enumerate(nodes)}
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]

    # initial reward matrix
    n = len(nodes)
    if n == 0 or not profit_matrix:
        return 0, []
    dp = [float('-inf')] * n
    path = [None] * n
    dp[start_idx] = 0.0

    for i in range(n):  # finding reward_highest path for simplification using DP
        for j in range(i+1, n):
            if profit_matrix[i][j] != 0 and dp[i] != float('-inf'):
                new_profit = dp[i] + profit_matrix[i][j]
                if new_profit > dp[j]:
                    dp[j] = new_profit
                    path[j] = i

    current = end_idx
    result_path = [nodes[current]]
    while path[current] is not None:
        current = path[current]
        result_path.append(nodes[current])
    result_path.reverse()

    return dp[end_idx], result_path  # return highest reward adn responding path



