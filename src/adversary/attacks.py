class AttackStrategy:
    def apply(self, model):
        raise NotImplementedError


class SignFlipAttack(AttackStrategy):
    def __init__(self, factor=-5.0):
        self.factor = factor

    def apply(self, model):
        for param in model.parameters():
            param.data = param.data * self.factor
