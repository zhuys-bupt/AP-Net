from models.gwcnet import GwcNet_G, GwcNet_C, GwcNet_A, GwcNet_CA
from models.loss import model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-a": GwcNet_A,
    "gwcnet-c": GwcNet_C,
    "gwcnet-ca": GwcNet_CA,
}
