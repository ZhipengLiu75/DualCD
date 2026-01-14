def average_accuracy(acc_matrix):
    final_accs = acc_matrix[-1]
    return sum(final_accs) / len(final_accs)


def performance_aware_forgetting(acc_matrix):
    T = len(acc_matrix)
    forgetting_list = []
    for j in range(T - 1):
        max_acc = max(acc_matrix[i][j] for i in range(j, T - 1))
        final_acc = acc_matrix[-1][j]
        forgetting = (1 - final_acc) * max((max_acc - final_acc) / max_acc, 0)
        forgetting_list.append(forgetting)
    return sum(forgetting_list) / len(forgetting_list)


def forgetting(acc_matrix):
    T = len(acc_matrix)
    forgetting_list = []
    for j in range(T - 1):
        max_acc = max(acc_matrix[i][j] for i in range(j, T - 1))
        final_acc = acc_matrix[-1][j]
        forgetting = max((max_acc - final_acc) / max_acc, 0)
        forgetting_list.append(forgetting)
    return sum(forgetting_list) / len(forgetting_list)


if __name__ == '__main__':
    x1 = [[0.40],
          [0.32, 0.28]]
    x2 = [[0.40],
          [0.32, 0.28],
          [0.22, 0.17, 0.40]]
    print(performance_aware_forgetting(x1))
    print(performance_aware_forgetting(x2))
