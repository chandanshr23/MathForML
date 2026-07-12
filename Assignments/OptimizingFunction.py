from jax import grad
import jax.numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Analyse the dataset

df = pd.read_csv(r'C:\Users\2026\Desktop\MathForML\Assignments\data\prices.csv') 
print(df)

#To Load historical prices of the product from A and B

prices_A = df.price_supplier_a_dollars_per_item
prices_B = df.price_supplier_b_dollars_per_item
prices_A = np.array(prices_A).astype('float32')
prices_B = np.array(prices_B).astype('float32')

print("Some prices of supplier A:", prices_A[0:5])
print("Some prices of supplier B:", prices_B[0:5])
print("Average of the prices, supplier A:", np.mean(prices_A))
print("Average of the prices, supplier B:", np.mean(prices_B))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(prices_A, 'g', label="Supplier A")
plt.plot(prices_B, 'b', label="Supplier B")
plt.legend()

#plt.show()

#Construct The function L to optimize and find its min poin

def f_of_omega(omega):
    f = prices_A * omega +prices_B * (1-omega)
    return f

def l_of_omega(omega):
    return 1/len(f_of_omega(omega)) * np.sum((f_of_omega(omega)-np.mean(f_of_omega(omega)))**2)

print("L(omega = 0) =",l_of_omega(0)) #110.72
print("L(omega = 0.2)",l_of_omega(0.2))#61.156796
print("L(omega = 0.8) =",l_of_omega(0.8)) #11.212797
print("L(omega = 1) =",l_of_omega(1)) # 27.48

#Observe that the functions are decreasing but increased at 1
#Therefore 
N =1001
omega_array = np.linspace(0, 1, N, endpoint=True)

def L_of_omega_array(omega_array):
    N = len(omega_array)
    L_array = np.zeros(N)
    for i in range(N):
        L = l_of_omega(omega_array[i])
        L_array = L_array.at[i].set(L)

    return L_array
L_array = L_of_omega_array(omega_array)
print("L(omega = 0) =",L_array[0])
print("L(omega = 1) =",L_array[N-1])

i_opt = L_array.argmin()
omega_opt = omega_array[i_opt]
L_opt = L_array[i_opt]
print(f'omega_min = {omega_opt:.3f}\nL_of_omega_min = {L_opt:.7f}')

def dLdOmega_of_omega_array(omega_array):
    N = len(omega_array)
    dLdOmega_array = np.zeros(N)
    for i in range(N):
        dLdOmega = grad(l_of_omega)(omega_array[i])
        dLdOmega_array = dLdOmega_array.at[i].set(dLdOmega)

    return dLdOmega_array
dLdOmega_array = dLdOmega_of_omega_array(omega_array)
print("dLdOmega(omega = 0) =",dLdOmega_array[0])
print("dLdOmega(omega = 1) =",dLdOmega_array[N-1])

i_opt_2 = np.abs(dLdOmega_array).argmin()
omega_opt_2 = omega_array[i_opt_2]
dLdOmega_opt_2 = dLdOmega_array[i_opt_2]
print(f'omega_min = {omega_opt_2:.3f}\ndLdOmega_min = {dLdOmega_opt_2:.7f}')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Setting the axes at the origin.
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(omega_array,  L_array, "black", label = "$\mathcal{L}\\left(\omega\\right)$")
plt.plot(omega_array,  dLdOmega_array, "orange", label = "$\mathcal{L}\'\\left(\omega\\right)$")
plt.plot([omega_opt, omega_opt_2], [L_opt,dLdOmega_opt_2], 'ro', markersize=3)

plt.legend()

plt.show()