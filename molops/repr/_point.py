from dataclasses import dataclass
from typing import Union

import numpy as np
import plotly.graph_objects as go


@dataclass
class PointCloud:
    pos: np.ndarray
    val: np.ndarray
    
    name: str = None
    
    def __repr__(self):
        if self.name is None:
            return f'PointCloud({self.pos.shape[0]} points)'
        return f'PointCloud({self.name}, {self.pos.shape[0]} points)'
    
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx: Union[int, np.ndarray]):
        if isinstance(idx, int):
            idx = np.array([idx])
        return PointCloud(self.pos[idx], self.val[idx])
    
    def scatter_view(self):
        colobar = dict(title='Values',
                        titleside='right',
                        titlefont=dict(size=16),
                        thickness=10)
        figure = go.Figure(data=[go.Scatter3d(
            x=self.pos[:, 0], 
            y=self.pos[:, 1], 
            z=self.pos[:, 2], 
            mode='markers', 
            marker=dict(
                size=3,
                color=self.val,
                colorscale='viridis',
                colorbar=colobar
            )
        )])
        # Hide the axes
        figure.update_layout(scene=dict(xaxis=dict(visible=False),
                                        yaxis=dict(visible=False),
                                        zaxis=dict(visible=False)))
        # Set figure size
        figure.update_layout(width=800, height=800)
        figure.show()