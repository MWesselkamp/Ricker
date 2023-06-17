def split_data(x, t = 180):
    """
    Function for the two-dimensional data returned by the Ricker simulator.
    t: point in time up to which we split
    """
    x_train, x_test = x[:,:t], x[:,t:]
    return x_train, x_test