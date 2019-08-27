

def scatter(X, ax, y=None):
    ax.scatter(X[:, 0], X[:, 1], s=6.0, c=y, cmap='viridis')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])
