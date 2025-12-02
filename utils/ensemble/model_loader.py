import timm

def get_model(name: str):
    if name in timm.list_models():
        model = timm.create_model(name, pretrained=True, num_classes=2)

    else:
        raise ValueError(f"Modell '{name}' ist nicht verfügbar.")

    return model