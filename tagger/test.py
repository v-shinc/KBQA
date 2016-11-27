import numpy as np
# def viterbi_decode_top3(score, transition_params):
#     trellis = [np.zeros_like(score) for _ in range(3)]
#     num_tags = transition_params.shape[0]
#     backpointers = [np.zeros_like(score, dtype=np.int32) for _ in range(3)]
#     backkth = [np.zeros_like(score, dtype=np.int32) for _ in range(3)]
#     for j in range(num_tags):
#         trellis[0][0, j] = score[0, j]
#         trellis[1][0, j] = -10
#         trellis[2][0, j] = -100
#
#     for t in range(1, score.shape[0]):
#         for j in range(num_tags):
#             for k in range(num_tags):
#                 for p in range(3):
#                     v = trellis[p][t-1, k] + transition_params[k, j] + score[t, j]
#
#                     if v > trellis[0][t, j]:
#                         trellis[0][t, j], trellis[1][t, j], trellis[2][t, j] = v, trellis[0][t, j], trellis[1][t, j]
#                         backpointers[0][t, j], backpointers[1][t, j], backpointers[2][t, j] = k, backpointers[0][t, j], backpointers[1][t, j]
#                         backkth[0][t, j], backkth[1][t, j], backkth[2][t, j] = p, backkth[0][t, j], backkth[1][t, j]
#                     elif v > trellis[1][t, j]:
#                         trellis[1][t, j], trellis[2][t, j] = v, trellis[1][t, j]
#                         backpointers[1][t, j], backpointers[2][t, j] = k, backpointers[1][t, j]
#                         backkth[1][t, j], backkth[2][t, j] = p, backkth[1][t, j]
#                     elif v > trellis[2][t, j]:
#                         trellis[2][t,j] = v
#                         backpointers[2][t, j] = k
#                         backkth[2][t, j] = p
#     print backpointers
#     print backkth
#     viterbi = [[100] for _ in range(3)]
#     kth = [[4] for _ in range(3)]
#     score0, score1, score2 = 0, 0, 0
#
#     for k in range(3):
#         for j in range(num_tags):
#             if trellis[k][-1, j] > score0:
#                 viterbi[0][-1], viterbi[1][-1], viterbi[2][-1] = j, viterbi[0][-1], viterbi[1][-1]
#                 kth[0][-1], kth[1][-1], kth[2][-1] = k, kth[0][-1], kth[1][-1]
#                 score0 = trellis[k][-1, j]
#             elif trellis[k][-1, j] > score1:
#                 viterbi[1][-1], viterbi[2][-1] = j, viterbi[1][-1]
#                 kth[1][-1], kth[2][-1] = k, kth[1][-1]
#                 score1 = trellis[k][-1, j]
#             elif trellis[k][-1, j] > score2:
#                 viterbi[2][-1] = j
#                 kth[2][-1] = k
#                 score2 = trellis[k][-1, j]
#     for i in range(3):
#         for t in reversed(range(1, score.shape[0])):
#             v = viterbi[i][-1]
#             k = kth[i][-1]
#             viterbi[i].append(backpointers[k][t, v])
#             kth[i].append(backkth[k][t, v])
#         viterbi[i].reverse()
#
#     return viterbi, [score0, score1, score2]

def viterbi_decode_top2(score, transition_params):
    print '[viterbi_decode_top2]', score
    print '[viterbi_decode_top2]', transition_params
    N = 2
    dp = [np.zeros_like(score) for _ in range(N)]   # dp[]
    num_tags = transition_params.shape[0]
    backpointers = [np.zeros_like(score, dtype=np.int32) for _ in range(N)]
    backkth = [np.zeros_like(score, dtype=np.int32) for _ in range(N)]
    for j in range(num_tags):
        dp[0][0, j] = score[0, j]
        dp[1][0, j] = -9999999
        # trellis[2][0, j] = -100

    for t in range(1, score.shape[0]):
        for j in range(num_tags):
            for k in range(num_tags):
                for p in range(N):
                    v = dp[p][t-1, k] + transition_params[k, j] + score[t, j]
                    if v >= dp[0][t, j]:
                        dp[0][t, j], dp[1][t, j] = v, dp[0][t, j]
                        backpointers[0][t, j], backpointers[1][t, j] = k, backpointers[0][t, j]
                        backkth[0][t, j], backkth[1][t, j] = p, backkth[0][t, j]
                    elif v >= dp[1][t, j]:
                        dp[1][t, j] = v
                        backpointers[1][t, j] = k
                        backkth[1][t, j] = p

    viterbi = [[100] for _ in range(N)]  # 100 is randomly chosen
    kth = [[4] for _ in range(N)]
    score0, score1, score2 = 0, 0, 0

    for k in range(N):
        for j in range(num_tags):
            if dp[k][-1, j] >= score0:
                viterbi[0][-1], viterbi[1][-1] = j, viterbi[0][-1]
                kth[0][-1], kth[1][-1] = k, kth[0][-1]
                score0, score1 = dp[k][-1, j], score0
            elif dp[k][-1, j] >= score1:
                viterbi[1][-1] = j
                kth[1][-1] = k
                score1 = dp[k][-1, j]

    for i in range(N):
        for t in reversed(range(1, score.shape[0])):
            v = viterbi[i][-1]
            k = kth[i][-1]
            viterbi[i].append(backpointers[k][t, v])
            kth[i].append(backkth[k][t, v])
        viterbi[i].reverse()

    return viterbi, [score0, score1]

if __name__ == '__main__':
    # score = np.array([[1, 2], [3, 1], [1, 2]])
    # transition_params = np.array([[1, 2], [3, 4]])
    # seq, scores = viterbi_decode_top3(score, transition_params)
    # print seq
    # print scores

    score = np.array([[-26.00880623, -27.84826088, -25.9820137,   -4.25348282, -27.43421173],
 [-27.32139397, -16.58219528, -34.09654999, -36.27558517, -45.00359344],
 [-27.67468452, -18.21633911, -31.78575134, -36.49900818, -47.57540131],
 [-26.5184803,  -19.58366776, -30.72154808, -37.00998688, -50.05875397],
 [-24.13256073, -20.89519691, -30.1657753, -40.31112289, -53.0104332 ],
 [-26.15292549, -18.85009766, -29.76890755, -40.30982971, -50.74594879],
 [-25.00113869, -20.17194176, -29.58408928, -39.31032562, -46.8590126 ],
 [-25.06334496, -26.55460548, -20.68481445, -39.68439484, -38.0427742 ],
 [-26.43078232, -25.17695427, -30.02226257, -28.89418983,  -1.81726646]])
    transition_params = np.array([[2.15751505e+00,  -1.74032950e+00,  -8.41138554e+00,  -1.35139310e+00, -3.86454642e-01],
 [ -1.21634703e+01,   1.10379577e+00,   2.31630421e+00,  -2.32365394e+00,
    2.21989572e-01],
 [  2.04314947e+00,  -1.78048074e+00,  -1.12410221e+01,  -1.56529427e+00,
    5.15773892e-01],
 [ -5.22425318e+00,  -2.51889348e-01,   2.49096543e-01,  -8.37462008e-01,
   -3.26884091e-01],
 [ -1.03669369e+00,  -1.20541072e+00,   7.40161049e-04,   4.77745742e-01,
   -1.76283944e+00]])
    seq, scores = viterbi_decode_top2(score, transition_params)
    print seq
    print scores

