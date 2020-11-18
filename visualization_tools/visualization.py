import matplotlib.pyplot as plt

def visualize_pos_encoding(array):
    """
    Plot heatmap of positional encoding. 
    """
    fig, ax = plt.subplots()
    plt.imshow(array)
    cbar = plt.colorbar()
    plt.show()



