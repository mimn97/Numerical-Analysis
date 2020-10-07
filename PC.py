from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


# Milne-Simpson PC method

def milnePC(def_fn, xa, xb, ya, N):

    f = def_fn  # intakes function to method to approximate

    h = (xb - xa) / N  # creates step size based on input values of a, b, N

    t = np.arange(xa, xb + h, h)  # array initialized to hold mesh points t

    y = np.zeros((N + 1,))  # array to hold Midpoint Method approximated y values

    y[0] = ya  # initial condition

    # using RK4 to obtain the first 3 points

    for i in range(0, N):
        if i in range(0, 3):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + (h / 2.0), y[i] + (k1 / 2.0))
            k3 = h * f(t[i] + (h / 2.0), y[i] + (k2 / 2.0))
            k4 = h * f(t[i] + h, y[i] + k3)

            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        else:
            y[i + 1] = y[i-3] + (4*h/3)*(2*f(t[i], y[i]) - f(t[i-1], y[i-1])
                                         + 2*f(t[i-2], y[i-2]))

            y[i + 1] = y[i-1] + (h/3)*(f(t[i+1], y[i+1]) + 4*f(t[i], y[i])
                                       + f(t[i-1], y[i-1]))

    return t, y


# Adams Fourth Order PC

def adamsPC(def_fn, xa, xb, ya, h):
    f = def_fn  # intakes function to method to approximate

    N = int((xb - xa) / h)  # creates step size based on input values of a, b, N

    t = np.arange(xa, xb + h, h)  # array intialized to hold mesh points t

    y = np.zeros((N + 1,))  # array to hold Midpoint Method approximated y values

    y[0] = ya  # initial condition

    # using RK4 to obtain the first 3 points
    for i in range(0, N):
        if i in range(0, 3):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + (h / 2.0), y[i] + (k1 / 2.0))
            k3 = h * f(t[i] + (h / 2.0), y[i] + (k2 / 2.0))
            k4 = h * f(t[i] + h, y[i] + k3)

            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        else:

            y[i + 1] = y[i] + (h/24.0) * (55.0 * f(t[i], y[i]) - 59.0 * f(t[i - 1], y[i - 1])
                                   + 37.0 * f(t[i - 2], y[i - 2]) - 9.0 * f(t[i - 3], y[i - 3]))

            y[i + 1] = y[i] + (h/24.0) * (9.0 * f(t[i + 1], y[i + 1])
                                   + 19.0 * f(t[i],y[i]) - 5.0 * f(t[i - 1], y[i - 1]) + f(t[i - 2], y[i - 2]))

    return t, y


if __name__ == "__main__":

    d_f = lambda x, y: (2 - 2*x*y)/(x**2 + 1)
    f = lambda x: (2*x + 1)/(x**2 + 1)
    x_1 = np.arange(0, 1.1, 0.1)
    x_2 = np.arange(0, 1.05, 0.05)

    x_milne_1, result_milne_1 = milnePC(d_f, 0, 1, 1, 10)
    x_milne_2, result_milne_2 = milnePC(d_f, 0, 1, 1, 20)

    x_adam_1, result_adam_1 = adamsPC(d_f, 0, 1, 1, 0.1)
    x_adam_2, result_adam_2 = adamsPC(d_f, 0, 1, 1, 0.05)

    y_exact_1 = f(x_1)
    y_exact_2 = f(x_2)

    print(result_adam_1)

    err_milne_1 = np.abs(y_exact_1 - result_milne_1)
    err_adam_1 = np.abs(y_exact_1 - result_adam_1)
    err_milne_2 = np.abs(y_exact_2 - result_milne_2)
    err_adam_2 = np.abs(y_exact_2 - result_adam_2)

    print(err_adam_1)
    print(err_adam_2)

    for i in range(len(err_adam_1)):
        print(err_adam_1[i] / err_adam_2[i*2])


    print(err_milne_1)
    print(err_milne_2)

    for i in range(len(err_milne_1)):
        print(err_milne_1[i] / err_milne_2[i*2])

    plt.figure(1)

    plt.plot(x_1, err_adam_1, label='ABM4')
    plt.plot(x_1, err_milne_1, label='Milne-Simpson')
    #plt.plot(x_2, err_adam_2, label='h=0.05')

    plt.xlabel('t')
    plt.ylabel('Absolute Error')
    plt.title('Stability Comparison when h = 0.1')
    plt.legend()

    plt.figure(2)

    plt.plot(x_1, err_milne_1, label='h=0.1')
    plt.plot(x_2, err_milne_2, label='h=0.05')
    plt.xlabel('t')
    plt.ylabel('Absolute Error')
    plt.title('Milne-Simpson Predictor-Corrector')
    plt.legend()

    plt.show()





