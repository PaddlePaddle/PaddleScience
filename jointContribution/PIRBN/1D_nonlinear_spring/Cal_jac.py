import paddle


def cal_adapt(pirbn, x):
    lamda_g = 0.0
    lamda_b1 = 0.0
    lamda_b2 = 0.0
    n_neu = len(pirbn.get_weights()[1])

    ### in-domain
    n1 = x[0].shape[0]
    for i in range(n1):
        temp_x = [x[0][i, ...].unsqueeze(0), paddle.to_tensor([[0.0]])]
        temp_x[0].stop_gradient = False
        temp_x[1].stop_gradient = False
        y = pirbn(temp_x)
        l1t = paddle.grad(
            y[0], pirbn.parameters(), retain_graph=True, create_graph=True
        )
        for j in l1t:
            lamda_g = lamda_g + paddle.sum(j**2) / n1
        temp = paddle.concat((l1t[0], l1t[1].reshape((1, n_neu))), axis=1)
        if i == 0:
            jac = temp
        else:
            jac = paddle.concat((jac, temp), axis=0)
    ### bound
    n2 = x[1].shape[0]
    for i in range(n2):
        temp_x = [paddle.to_tensor([[0.0]]), x[1][i, ...].unsqueeze(0)]
        temp_x[0].stop_gradient = False
        temp_x[1].stop_gradient = False
        y = pirbn(temp_x)
        l1t = paddle.grad(
            y[1], pirbn.parameters(), retain_graph=True, create_graph=True
        )
        l2t = paddle.grad(
            y[2], pirbn.parameters(), retain_graph=True, create_graph=True
        )
        for j in l1t:
            lamda_b1 = lamda_b1 + paddle.sum(j**2) / n2
        for j in l2t:
            lamda_b2 = lamda_b2 + paddle.sum(j**2) / n2
    ### calculate adapt factors
    temp = lamda_g + lamda_b1 + lamda_b2
    lamda_g = temp / lamda_g
    lamda_b1 = temp / lamda_b1
    lamda_b2 = temp / lamda_b2

    return lamda_g, lamda_b1, lamda_b2, jac
