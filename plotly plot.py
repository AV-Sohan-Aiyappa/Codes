import plotly.graph_objects as go
import numpy as np

# Sample data (replace with your own data)
data = np.random.randint(10, 100, size=(10, 12))

# Create the heatmap
fig = go.Figure(data=go.Heatmap(z=data))
fig.update_layout(
    title='Heatmap',
    xaxis_title='X-axis',
    yaxis_title='Y-axis'
)
fig.show()
