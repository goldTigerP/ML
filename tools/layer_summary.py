import importlib.util, torch

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load SimpleMLP
simple_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\simple_train.py','simple_train')
SimpleMLP = simple_mod.SimpleMLP
# load SimpleCNN
cnn_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\cnn_train.py','cnn_train')
SimpleCNN = cnn_mod.SimpleCNN
# load resnet builder
resnet_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\resnet_mnist_train.py','resnet_mnist_train')
build_model = resnet_mod.build_model

models = [
    ('SimpleMLP', SimpleMLP(), torch.randn(1,1,28,28)),
    ('SimpleCNN', SimpleCNN(), torch.randn(1,1,28,28)),
    ('ResNet18_MNIST', build_model(pretrained=False, in_channels=1, num_classes=10), torch.randn(1,1,28,28)),
]

results = {}
for name, m, inp in models:
    m.eval()
    layers = []
    outputs = {}
    # forward hooks
    def make_hook(n):
        def hook(module, input, output):
            outputs[n] = output.detach()
        return hook
    handles = []
    # register hooks on modules that have parameters
    for n, module in m.named_modules():
        if len(list(module.parameters()))>0:
            handles.append((n,module.register_forward_hook(make_hook(n))))
    # run forward
    with torch.no_grad():
        out = m(inp)
    # gather stats
    for n, module in m.named_modules():
        if len(list(module.parameters()))>0:
            pcount = sum(p.numel() for p in module.parameters())
            out_t = outputs.get(n)
            if out_t is not None:
                if isinstance(out_t, (list, tuple)):
                    out_shape = tuple(o.shape for o in out_t)
                    node_count = None
                else:
                    out_shape = tuple(out_t.shape)
                    node_count = int(torch.tensor(out_t.shape[1:]).prod().item()) if out_t.dim()>1 else 0
            else:
                out_shape = None
                node_count = None
            layers.append((n, module.__class__.__name__, int(pcount), out_shape, node_count))
    # cleanup
    for n,h in handles:
        h.remove()
    results[name]=layers

import json
print(json.dumps(results, indent=2))
