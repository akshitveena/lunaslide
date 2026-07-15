import numpy as np

def compute_slope(H, grid_spacing):
    dx = np.diff(H, axis=1, append=H[:, -1:])
    dy = np.diff(H, axis=0, append=H[-1:, :])
    return np.sqrt((dx / grid_spacing)**2 + (dy / grid_spacing)**2)

def simulate_mass_wasting(H, grid_spacing=5.0, crit=0.577, relax_factor=0.2, max_iter=500):
    """
    Simulates slope relaxation / landslides using cellular automata.
    crit = 0.577 is roughly tan(30°).
    """
    H_ca = H.copy()
    toppled_mask = np.zeros_like(H, dtype=bool)
    changed = True
    iter_count = 0
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        for shift, axis in [(-1, 1), (1, 1), (-1, 0), (1, 0)]:
            nbr = np.roll(H_ca, shift, axis=axis)
            slope = (H_ca - nbr) / grid_spacing
            
            # Only move material downhill
            mask = slope > crit
            
            # Prevent wrap-around mass movement at the boundaries (Pac-Man effect)
            if axis == 0:
                if shift == 1:
                    mask[0, :] = False
                elif shift == -1:
                    mask[-1, :] = False
            elif axis == 1:
                if shift == 1:
                    mask[:, 0] = False
                elif shift == -1:
                    mask[:, -1] = False
                    
            if mask.any():
                changed = True
                # Move a fraction of the excess height
                excess_height = (slope - crit) * grid_spacing
                delta = relax_factor * excess_height
                
                # Subtract from source
                H_ca[mask] -= delta[mask]
                
                # Add to neighbor (roll the delta array in the opposite direction)
                delta_array = np.zeros_like(H_ca)
                delta_array[mask] = delta[mask]
                deposit_array = np.roll(delta_array, -shift, axis=axis)
                
                H_ca += deposit_array
                toppled_mask |= mask
                
    return H_ca, toppled_mask, iter_count
