import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        # Number of tiles along each dimension for one tiling
        self.tiles_per_dim = (
            np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        )

        # Tiles in one tiling, and all tilings
        self.tiles_per_tiling = int(np.prod(self.tiles_per_dim))
        self.total_tiles = self.tiles_per_tiling * self.num_tilings

        # One weight per tile
        self.weights = np.zeros(self.total_tiles)

        # Pre-compute tiling start offsets
        self.tiling_offsets = []
        for tiling_index in range(self.num_tilings):
            start = state_low - (tiling_index / self.num_tilings) * tile_width
            self.tiling_offsets.append(start)

    def _active_tiles(self, state: np.array):
        """Return indices of active tiles for this state across all tilings."""
        indices = []
        for tiling_index in range(self.num_tilings):
            start = self.tiling_offsets[tiling_index]
            shifted = state - start

            # Tile coordinates in this tiling
            coords = np.floor(shifted / self.tile_width).astype(int)
            coords = np.clip(coords, 0, self.tiles_per_dim - 1)

            # Flatten to one index within this tiling
            tile_id = np.ravel_multi_index(coords, self.tiles_per_dim)

            # Shift by tiling offset
            tile_id += tiling_index * self.tiles_per_tiling
            indices.append(tile_id)
        return indices

    def __call__(self,s):
        active = self._active_tiles(s)
        return float(np.sum(self.weights[active]))

    def update(self,alpha,G,s_tau):
        """
        Semi-gradient TD update for state s_tau.

        alpha: step size
        G:     target (n-step return)
        """
        active = self._active_tiles(s_tau)
        value = self.__call__(s_tau)
        td_error = G - value

        # Spread update over active tiles
        step = alpha * td_error / self.num_tilings
        for idx in active:
            self.weights[idx] += step
