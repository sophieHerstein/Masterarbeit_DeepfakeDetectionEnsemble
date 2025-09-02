import timm

def get_model(name: str, num_classes: int, pretrained=True):
    if name in timm.list_models():
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Modell '{name}' ist nicht verfügbar.")

    return model