import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# 1.1
# for single precision
a = np.float32(3.3e38)
b = np.float32(2)
print(b * a)
# for double precision
a = 1.7e308
print(2 * a)

# 1.2
a = float("inf")
print(9 / a)
#print(0/0)
#print(8/0)

# 1.3
a = np.nan
print(a)
print(math.isnan(a))
print(a + 1)

# 1.4
a = float("inf")
b = float("inf")
c = np.nan
d = np.nan
print("comparison between infinity is %d" % (a > b))
print(c > d)
print(a > c)
print(a > 1)
print(c > 1)

# 1.5
a = +0
b = -0
print(np.sqrt(a))
print(np.sqrt(b))


# 2.1
def get_pi(x):
    return (np.sqrt(1 + np.power(x, 2)) - 1) / x


def get_pi_32(x):
    return (np.sqrt(np.float32(1) + np.power(np.float32(x), np.float32(2))) - np.float32(1)) / np.float32(x)


# for double precision
num_pi = []
num_pi_32 = []
y_error_single = []
y_error_double = []
t = 1 / np.sqrt(3)
n = 9
num_pi.append(6 * np.power(2, 0) * t)
for i in range(1, n):
    t = get_pi(t)
    pi = 6 * t * np.power(2, i)
    num_pi.append(pi)
    error = abs(pi-np.pi)/np.pi
    y_error_double.append(error)
pi_64 = num_pi[-1]
print('with double precision, Pi is:', pi_64)

# for single precision
t = 1 / np.sqrt(3)
t = np.float32(t)
num_pi_32.append(np.float32(np.float32(6) * np.power(np.float32(2), np.float32(0)) * t))
for i in range(1, n):
    t = get_pi_32(t)
    pi = np.float32(np.float32(6) * t * np.power(np.float32(2), np.float32(i)))
    num_pi_32.append(pi)
    error = abs(pi-np.float32(np.pi))/np.pi
    y_error_single.append(error)
pi_32 = num_pi_32[-1]
print('with single precision, Pi is:', pi_32)
# plot
x = list(range(0, n))

y1 = num_pi
y2 = num_pi_32
y3 = []

print(len(x))
for i in range(1, n+1):
    y3.append(np.pi)
print(len(y3))
print(len(y_error_double))
plt.plot(x, y1, color="r", linestyle="-", marker="^", linewidth=1, label='double precision')

plt.plot(x, y2, color="b", linestyle="--", marker="s", linewidth=1, label='single precision')

plt.plot(x, y3, color="g", linestyle=":", marker="+", linewidth=1, label='pi')

plt.xlabel(u'x')
plt.ylabel(u'result_pi')
plt.legend()

plt.show()

# 2.1.2
x = list(range(1, n))
plt.plot(x, y_error_double, color="r", linestyle="-", marker="^", linewidth=1, label='double precision')

plt.plot(x, y_error_single, color="b", linestyle="--", marker="s", linewidth=1, label='single precision')
plt.xlabel(u'x')
plt.ylabel(u'error')
plt.legend()
plt.show()

#2.2
def get_pi(x):
    return x/(np.sqrt(1+np.power(x, 2))+1)


def get_pi_32(x):
    return np.float32(x)/((np.sqrt(np.float32(1)+np.power(np.float32(x), np.float32(2)))+np.float32(1)))


# for double precision
num_pi = []
num_pi_32 = []
y_error_single = []
y_error_double = []
t = 1 / np.sqrt(3)
n = 30
num_pi.append(6 * np.power(2, 0) * t)
for i in range(1, n):
    t = get_pi(t)
    pi = 6 * t * np.power(2, i)
    num_pi.append(pi)
    error = abs(pi-np.pi)/np.pi
    y_error_double.append(error)

pi_64 = num_pi[-1]
print('with double precision, Pi is:', pi_64)

# for single precision
t = 1 / np.sqrt(3)
t = np.float32(t)
num_pi_32.append(np.float32(np.float32(6) * np.power(np.float32(2), np.float32(0)) * t))
for i in range(1, n):
    t = get_pi_32(t)
    pi = np.float32(np.float32(6) * t * np.power(np.float32(2), np.float32(i)))
    num_pi_32.append(pi)
    error = abs(pi-np.float32(np.pi))/np.pi
    y_error_single.append(error)
    print(i,"single",pi)
pi_32 = num_pi_32[-1]
print('with single precision, Pi is:', pi_32)
# plot
x = list(range(0, n))

y1 = num_pi
y2 = num_pi_32
y3 = []

print(len(x))
for i in range(1, n+1):
    y3.append(np.pi)
print(len(y3))
print(len(y_error_double))
plt.plot(x, y1, color="r", linestyle="-", marker="^", linewidth=1, label='double precision')

plt.plot(x, y2, color="b", linestyle="--", marker="s", linewidth=1, label='single precision')

plt.plot(x, y3, color="g", linestyle=":", marker="+", linewidth=1, label='pi')

plt.xlabel(u'x')
plt.ylabel(u'result_pi')
plt.legend()

plt.show()
# 2.2.2
x = list(range(1, n))
plt.plot(x, y_error_double, color="r", linestyle="-", marker="^", linewidth=1, label='double precision')

plt.plot(x, y_error_single, color="b", linestyle="--", marker="s", linewidth=1, label='single precision')
plt.xlabel(u'x')
plt.ylabel(u'error')
plt.legend()
plt.show()

# 3.1

x_num = np.logspace(-18,-14,50)
y_1_num = []
y_2_num = []
for i in range(0, len(x_num)):
    x = x_num[i]
    y_1_num.append(np.log(x + 1) / x)
    y_2_num.append(np.log1p(x) / x)


plt.plot(x_num, y_1_num, color="r", linestyle="--", marker="s", linewidth=1, label='log',ms = 0.5)
plt.plot(x_num, y_2_num, color="g", linestyle=":", marker="+", linewidth=1, label='log1p', ms = 0.5)
plt.xlabel('the value of x')
plt.ylabel('the value of ln(1+x)/x')
plt.title('compare ln(1+x) and log1p')
plt.legend()


plt.show()


# 3.2
def taylor_ln(x):
    return x - (1 / 2) * np.power(x, 2)


def new_ln_small(x):
    if x < 1e-15:
        return taylor_ln(x)
    return np.log(1 + x)


x = 1e-4
x_num = []
y_ln = []
y_taylor = []
for i in range(1, 400):
    x_num.append(x)

    y_ln.append(abs(np.log(1 + x)-np.log1p(x))/np.log1p(x))

    y_taylor.append(abs(taylor_ln(x)-np.log1p(x))/np.log1p(1))
    x= x+1e-7


plt.plot(x_num, y_ln, color="r", linestyle="--", marker="s", ms=0.1, linewidth=1, label='ln(1+x)')
plt.plot(x_num, y_taylor, color="g", linestyle=":", marker="+", ms=0.1, linewidth=1, label='taylor')
plt.xlabel('x')
plt.ylabel('relative error')
plt.legend()
plt.title('the error of taylor and ln(1+x)')
plt.show()

x = 1e-10
x_num = []
y_ln = []
y_taylor = []
for i in range(1, 500):
    x_num.append(x)
    y_ln.append(abs(np.log(1 + x)-np.log1p(x))/np.log1p(x))

    y_taylor.append(abs(taylor_ln(x)-np.log1p(x))/np.log1p(1))
    x = x + 1e-10

plt.plot(x_num, y_ln, color="r", linestyle="--", marker="s", ms=0.1, linewidth=1, label='ln(1+x)')
plt.plot(x_num, y_taylor, color="g", linestyle=":", marker="+", ms=0.1, linewidth=1, label='taylor')
plt.xlabel('x')
plt.ylabel('relative error')
plt.legend()
plt.show()



# 3.3
def new(x):
    if x < sys.float_info.epsilon:
        return x
    return (x * np.log(1 + x)) / ((1 + x) - 1)



x_num = np.logspace(-12,0,500)
y_log1p = []
y_new = []
for i in range(0, len(x_num)):
    x = x_num[i]
    y_new.append(abs(new(x)-np.log1p(x))/np.log1p(x))


plt.plot(x_num, y_new, color="r", linestyle="--", marker="s", linewidth=1, ms=0.5, label=' alternative calculation')
plt.xlabel('x')
plt.ylabel('relative error')
plt.legend()
plt.title('error of this calculation')
plt.show()


# 4.1

def second_derivative_64(x, y):
    return (np.sin(x + y) - 2 * np.sin(x) + np.sin(x - y)) / np.power(y, 2)


def second_derivative_32(x, y):
    return np.float32((np.sin(np.float32(x) + np.float32(y)) - 2 * np.sin(np.float32(x)) + np.sin(
        np.float32(x) - np.float32(y))) / (np.float32(y) * np.float32(y)))


x0 = np.pi / 4
exact_sin = -np.sin(x0)
h_list = []
sin_64_list = []
sin_32_list = []
err1 = []
err2 = []

for j in range(7, 15):
    h = np.power(2, j)*np.power(1/2, 16)
    h_list.append(h)
    sin_64 = second_derivative_64(x0, h)
    sin_32 = second_derivative_32(x0, h)
    err1.append(abs(sin_64 - exact_sin) / abs(exact_sin))
    err2.append(abs(sin_32 - exact_sin) / abs(exact_sin))
    sin_64_list.append(sin_64)
    sin_32_list.append(sin_32)

plt.plot(h_list, err1, color="r", linestyle="--", marker="^", linewidth=1, ms=4, label='double')
plt.xlabel('h')
plt.ylabel('relative error')
plt.legend()
plt.show()

plt.plot(h_list, err2, color="r", linestyle=":", marker="+", linewidth=1, ms=4, label='single')
plt.xlabel('h')
plt.ylabel('relative error')
plt.legend()
plt.show()


# 4.2
x0 = np.pi / 4
exact_sin = -np.sin(x0)
h_list = []
sin_64_list = []
sin_32_list = []
err1 = []
err2 = []
error_t = []
error_r = []
epsilon_64 = np.finfo(np.float).eps
epsilon_32 = np.finfo(np.float32).eps
# double plot
for j in range(0, 8):
    h = np.power(2, j)*np.power(1/2, 16)
    h_list.append(h)
    sin_64 = second_derivative_64(x0, h)
    sin_32 = second_derivative_32(x0, h)
    err1.append(abs(sin_64 - exact_sin) / abs(exact_sin))
    err2.append(abs(sin_32 - exact_sin) / abs(exact_sin))
    sin_64_list.append(sin_64)
    sin_32_list.append(sin_32)
    error_t.append(epsilon_64/(np.power(h, 2)))
    error_r.append(np.power(h, 2)/12)

plt.plot(h_list, err1, color="r", linestyle="--", marker="^", linewidth=1, ms=2, label='double')
plt.plot(h_list, error_t, color="b", linestyle="--", marker="+", linewidth=1, ms=2, label='truncation error')
plt.plot(h_list, error_r, color="g", linestyle="--", marker="s", linewidth=1, ms=2, label='roundoff error')
plt.xlabel('h')
plt.ylabel('relative error')
plt.legend()
plt.title('double precision error')
plt.show()

print(np.power(12*epsilon_64,1/4))

# single plot
x0 = np.pi / 4
exact_sin = -np.sin(x0)
h_list = []
sin_64_list = []
sin_32_list = []
err1 = []
err2 = []
error_t = []
error_r = []
epsilon_64 = np.finfo(np.float).eps
epsilon_32 = np.finfo(np.float32).eps
for j in range(7, 15):
    h = np.power(2, j)*np.power(1/2, 16)
    h_list.append(h)
    sin_64 = second_derivative_64(x0, h)
    sin_32 = second_derivative_32(x0, h)
    err1.append(abs(sin_64 - exact_sin) / abs(exact_sin))
    err2.append(abs(sin_32 - exact_sin) / abs(exact_sin))
    sin_64_list.append(sin_64)
    sin_32_list.append(sin_32)
    error_t.append(epsilon_32/(np.power(h, 2)))
    error_r.append(np.power(h, 2)/12)

plt.plot(h_list, err2, color="r", linestyle="--", marker="^", linewidth=1, ms=4, label='single')
plt.plot(h_list, error_t, color="b", linestyle="--", marker="+", linewidth=1, ms=2, label='truncation error')
plt.plot(h_list, error_r, color="g", linestyle="--", marker="s", linewidth=1, ms=2, label='roundoff error')
plt.xlabel('h')
plt.ylabel('relative error')
plt.legend()
plt.title('single precision error')
plt.show()
print(np.power(12*epsilon_32,1/4))
print(np.power(1/2,12))