import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# Define the system of equations
# -x1 + 3x2 = 7
# 3x1 + 2x2 = 1
# ---------------------------------------

A = np.array([
    [-1, 3],
    [3, 2]
], dtype=float)

b = np.array([7, 1], dtype=float)

# Create augmented matrix [A | b]
A_system = np.hstack((A, b.reshape(-1, 1)))

print("Augmented Matrix:")
print(A_system)

print("\nSecond Row:")
print(A_system[1])


# ---------------------------------------
# Function to plot the equations
# ---------------------------------------

def plot_lines(M):

    # Generate x values
    x_1 = np.linspace(-10, 10, 200)

    # Equation 1
    x_2_line_1 = (M[0, 2] - M[0, 0] * x_1) / M[0, 1]

    # Equation 2
    x_2_line_2 = (M[1, 2] - M[1, 0] * x_1) / M[1, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot first equation
    ax.plot(
        x_1,
        x_2_line_1,
        linewidth=2,
        color="blue",
        label=f"$x_2={-M[0,0]/M[0,1]:.2f}x_1+{M[0,2]/M[0,1]:.2f}$"
    )

    # Plot second equation
    ax.plot(
        x_1,
        x_2_line_2,
        linewidth=2,
        color="orange",
        label=f"$x_2={-M[1,0]/M[1,1]:.2f}x_1+{M[1,2]/M[1,1]:.2f}$"
    )

    # Extract A and b
    A = M[:, :-1]
    b = M[:, -1]

    # Compute determinant
    det = np.linalg.det(A)

    print("\nDeterminant =", det)

    if det != 0:

        # Solve Ax = b
        solution = np.linalg.solve(A, b)

        print("Solution =", solution)

        # Plot solution point
        ax.plot(
            solution[0],
            solution[1],
            "ro",
            markersize=10,
            label="Solution"
        )

        # Annotate solution
        ax.text(
            solution[0] + 0.2,
            solution[1] + 0.2,
            f"({solution[0]:.0f}, {solution[1]:.0f})",
            fontsize=12,
            color="red"
        )

    else:
        print("No unique solution exists.")

    # Formatting
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_yticks(np.arange(-10, 11, 1))

    ax.grid(True)
    ax.set_aspect("equal")

    ax.legend()

    plt.show()


# ---------------------------------------
# Plot the system
# ---------------------------------------

plot_lines(A_system)



#To Avoid Singularity errror due to 0 or infinite solutions it's better to solve the equations under try/catch block
#Say for this equation : -x1+3x2=7 , 3x1+9x2=1 
A_2=np.array([[-1,3],[3,9]],dtype=np.type(float))
b_2=np.array([7,1],dtype=np.type(float))
d_2=np.linalg.det(A_2) #Gives error

#Therefore
try:
    x_2=np.linalg.det(A_2)
except np.linalg.linalgarror as err:
    print(err)


A2_system = np.hstack((A_2,b_2.reshape(2,1)))
print(A2_system)

#perform elimination
A2_system_res = A2_system.copy()
A2_system_res[1] = 3 * A2_system_res[0]+A2_system_res[1]
print(A2_system_res)

plot_lines(A2_system)


#What if the solutions are infinite

b_3 = np.array([7,-21] , dtype=np.dtype(float))

A_3_system = np.hstack((A_2, b_3.reshape((2,1))))
print(A_3_system)

A_3_system_res = A_3_system.copy()
A_3_system_res[1] = 3 * A_3_system_res[0] + A_3_system_res[1]
print(A_3_system_res)

plot_lines(A_3_system)