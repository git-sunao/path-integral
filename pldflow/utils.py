import jax.numpy as jnp
from jax import vmap
import jax.scipy.signal as jsignal
import jax

class LinearInterp:
    """
    Linear interpolater
    
    1. Find index `i` satisfying x[i] <= x_in < x[i+1]
    2. y = a_i * (x_in - x_i) + y_i, where a_i = (y_{i+1}-y_i)/(x_{i+1}-x_i)
    """
    def __init__(self, x, y, axis=0):
        """
        x: 1d array
        y: nd array
        axis: axis to interp
        """
        assert x.ndim == 1, 'ndim of x must be 1.'
        self.axis = axis
        self.x = x
        self.y = jnp.moveaxis(y, axis, 0)
        self.a = (self.y[1:,...]-self.y[:-1,...])/(self.x[1:,jnp.newaxis]-self.x[:-1,jnp.newaxis])
        
    def find_i(self, x):
        """
        Returns index `i` satisfying
        self.x[i,:] <= x < self.x[i+1,:]
        """
        i = jnp.searchsorted(self.x, x, side='right')-1
        return i
    
    def parse_i(self, vals, idxs):
        """
        vals: (n, ...) array
        idxs: (n, ...) array
        Returns vals[idxs]
        """
        assert vals.shape[1:] == idxs.shape[1:], f'shape mismatch: {vals.shape} and {idxs.shape}'
        # shape
        vshape = vals.shape
        ishape = idxs.shape
        # reshape
        vals2d = vals.reshape(vshape[0], -1)
        idxs2d = idxs.reshape(ishape[0], -1)
        # map
        outs2d = vmap(lambda val, idx: val[idx])(vals2d.T, idxs2d.T).T
        # reshape
        out = outs2d.reshape(ishape)
        return out        
    
    def __call__(self, x_in, return_idx=False):
        """
        This function returns y = a_i * (x_in - x_i) + y_i
        where a_i = (y_{i+1}-y_i)/(x_{i+1}-x_i)

        Parameters:
            x_in (n, ...): input x
            return_idx (bool): if True, return index `i`

        Returns:
            y (n, ...): interpolated y

        If x_in.ndim == 1, then the same x_in array is used 
        for all y. If not, then x_in must be the same shape as y.
        and the individual x_in is used for each y.
        """
        i = self.find_i(x_in)
        if x_in.ndim == 1:
            a = self.a[i,:]
            dx= x_in - self.x[i]
            y = a*dx + self.y[i,:]
        else:
            a = self.parse_i(self.a, i)
            dx= x_in - self.x[i]
            y = a*dx + self.parse_i(self.y, i)
        # return
        if return_idx:
            return y, i
        else:
            return y
        
def gaussian_kernel(size=5, sigma=1.0):
    """Create a 1D gaussian kernel"""
    x = jnp.arange(-size // 2 + 1, size // 2 + 1)
    kernel = jnp.exp(-x**2 / (2 * sigma**2))
    return kernel / jnp.sum(kernel)

def extrapolate_linear(data, size, axis=-1):
    """Linear extrapolation along a specified axis."""
    x = jnp.arange(data.shape[axis])
    
    def extrap(x, data, size):
        slope_left = data[1] - data[0]
        left_extrap = data[0] + slope_left * jnp.arange(-size, 0)
        
        slope_right = data[-1] - data[-2]
        right_extrap = data[-1] + slope_right * jnp.arange(1, size + 1)
        
        return jnp.concatenate([left_extrap, data, right_extrap])
    
    return jnp.apply_along_axis(lambda d: extrap(x, d, size), axis, data)

def gaussian_filter_jax(data, sigma=1.0, axis=-1, extrap=False):
    """Apply a Gaussian filter along a specified axis."""
    if extrap:
        size = int(sigma)
        data = extrapolate_linear(data, size, axis=axis)
    
    kernel = gaussian_kernel(size=int(6 * sigma) + 1, sigma=sigma)
    
    # Move the target axis to the last dimension for easier processing
    data = jnp.moveaxis(data, axis, -1)
    smoothed = jnp.apply_along_axis(lambda d: jsignal.convolve(d, kernel, mode='same'), -1, data)
    
    # Move axis back to original position
    smoothed = jnp.moveaxis(smoothed, -1, axis)
    
    if extrap:
        smoothed = jnp.take(smoothed, indices=jnp.arange(size, smoothed.shape[axis] - size), axis=axis)
    
    return smoothed

def unique_rows_via_columns(x):
    idx = jnp.ones(x.shape[0], dtype=bool)
    for col in x.T:
        _, unique_idx = jnp.unique(col, return_index=True)
        idx &= jnp.zeros(x.shape[0], dtype=bool).at[unique_idx].set(True)
    return x[idx]