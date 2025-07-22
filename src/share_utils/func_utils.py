
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_module(module):
    for _, param in module.named_parameters():
        param.requires_grad = False
    module = module.eval()
    module.train = disabled_train
    return module

