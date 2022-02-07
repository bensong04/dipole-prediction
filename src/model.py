# atom3d imports
import atom3d
from atom3d.models.cnn import CNN3D

class DipoleModel(CNN3D):
    def __init__(self, in_dim, box_size, hidden_dim=64, dropout=0.1):
        super(DipoleModel, self).__init__(in_dim, out_dim=1, box_size=box_size, hidden_dim=hidden_dim, dropout=dropout)
