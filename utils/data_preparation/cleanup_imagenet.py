import os
import shutil

def cleanup_imagenet(root_dir, keep_ids):
    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} ist kein gültiger Ordner")

    removed = []
    kept = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        if folder not in keep_ids:
            shutil.rmtree(folder_path)
            removed.append(folder)
        else:
            kept.append(folder)

    print(f"Behalten: {len(kept)} Ordner")
    print(f"Gelöscht: {len(removed)} Ordner")
    if removed:
        print("Gelöschte IDs:", ", ".join(removed))


if __name__ == "__main__":
    root = "../data_raw/imagenet"

    ALLOWED_IMAGENET_IDS = [
        "n02791270",  # barbershop / Barbershop
        "n02793495",  # barn / Scheune
        "n02859443",  # boathouse / Bootshaus
        "n02871525",  # bookshop / Buchladen
        "n02980441",  # castle / Burg
        "n03028079",  # church / Kirche (Kathedrale einschl.)
        "n03032252",  # cinema / Kino
        "n03042490",  # cliff dwelling / Felsenwohnung
        "n03457902",  # greenhouse / Gewächshaus
        "n03461385",  # grocery store / Lebensmittelgeschäft
        "n03661043",  # library / Bibliothek
        "n03697007",  # lumbermill / Sägewerk (industriell)
        "n03776460",  # mobile home / Mobilheim
        "n03781244",  # monastery / Kloster
        "n03788195",  # mosque / Moschee (mit Minaretten)
        "n03877845",  # palace / Palast (Villa-ähnlich, repräsentativ)
        "n03956157",  # planetarium / (museum-ähnlich modern)
        "n04005630",  # prison / Gefängnis
        "n04081281",  # restaurant / Restaurant
        "n04613696",  # yurt / Jurte (Hütte/Chalet-Ersatz)
        "n03216828",  # dock / Hafenanlage
        "n03743016",  # megalith / Megalith
        "n03837869",  # obelisk / Obelisk
        "n03899768",  # patio / Terrasse
        "n03933933",  # pier / Pier
        "n04311004",  # steel arch bridge / Stahlbogenbrücke
        "n04346328",  # stupa / Stupa (pagodenähnlich)
        "n04366367",  # suspension bridge / Hängebrücke
        "n04326547",  # stone wall / Steinmauer
        "n04486054",  # triumphal arch / Triumphbogen (röm. Ruinen-Vibe)
        "n04532670",  # viaduct / Viadukt
        "n04562935",  # water tower / Wasserturm
        "n03220513",  # dome / Kuppel
        "n04417672",  # thatch / Reetdach
        "n04435653",  # tile roof / Ziegeldach
    ]

    cleanup_imagenet(root, ALLOWED_IMAGENET_IDS)
