import importlib.util, torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load modules
simple_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\simple_train.py','simple_train')
SimpleMLP = simple_mod.SimpleMLP
cnn_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\cnn_train.py','cnn_train')
SimpleCNN = cnn_mod.SimpleCNN
resnet_mod = load_module(r'c:\Mine\project\ML\mnist_recognition\resnet_mnist_train.py','resnet_mnist_train')
build_model = resnet_mod.build_model

models = [
    ('SimpleMLP', SimpleMLP(), torch.randn(1,1,28,28)),
    ('SimpleCNN', SimpleCNN(), torch.randn(1,1,28,28)),
    ('ResNet18_MNIST', build_model(pretrained=False, in_channels=1, num_classes=10), torch.randn(1,1,28,28)),
    # classic ImageNet ResNet-18 (3-channel, 224x224)
    ('ResNet18_ImageNet', __import__('torchvision').models.resnet18(weights=None), torch.randn(1,3,224,224)),
]


def compute_macs_for_model(m, inp):
    m.eval()
    outputs = {}
    handles = []

    def make_hook(n):
        def hook(module, input, output):
            outputs[n] = output.detach()
        return hook

    for n, module in m.named_modules():
        if any(p is not None for p in module.parameters(recurse=False)):
            handles.append((n, module.register_forward_hook(make_hook(n))))

    with torch.no_grad():
        _ = m(inp)

    total_macs = 0
    details = []
    for n, module in m.named_modules():
        params = sum(p.numel() for p in module.parameters())
        macs = 0
        out = outputs.get(n)
        cls = module.__class__.__name__
        if isinstance(module, torch.nn.Conv2d):
            # weight shape: (out_channels, in_channels, kh, kw)
            kh, kw = module.kernel_size
            oc, ic = module.weight.shape[0], module.weight.shape[1]
            if out is not None:
                _, _, h, w = out.shape
                macs = int(oc * ic * kh * kw * h * w)
        elif isinstance(module, torch.nn.Linear):
            ic = module.in_features
            oc = module.out_features
            macs = int(ic * oc)
        else:
            macs = 0
        total_macs += macs
        details.append((n, cls, int(params), macs, tuple(out.shape) if out is not None else None))

    for n,h in handles:
        h.remove()
    return total_macs, details

if __name__ == '__main__':
    import json
    results = {}
    for name, model, inp in models:
        macs, details = compute_macs_for_model(model, inp)
        results[name] = {'total_macs': int(macs), 'details': details}
    print(json.dumps(results, indent=2))
