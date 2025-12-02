import os
import json

# Class mapping for YOLO ids
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
}


def convert_bbox_to_yolo(img_width, img_height, bbox):
    # Converts [xmin, ymin, xmax, ymax] to YOLO normalized format
    xmin, ymin, xmax, ymax = bbox

    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return x_center, y_center, width, height


def process_split(split_dir):
    # Defines main directories
    img_dir = os.path.join(split_dir, "img")
    ann_dir = os.path.join(split_dir, "ann")
    labels_dir = os.path.join(split_dir, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".json"):
            continue

        ann_path = os.path.join(ann_dir, ann_file)

        # Loads annotation JSON
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reads image size from JSON
        w = data["size"]["width"]
        h = data["size"]["height"]

        # Matches annotation to its image name
        img_name = ann_file.replace(".json", "")
        txt_name = img_name.replace(".jpg", ".txt")
        txt_path = os.path.join(labels_dir, txt_name)

        with open(txt_path, "w") as out:
            # Iterates through all annotated objects
            for obj in data.get("objects", []):
                cls_title = obj.get("classTitle", "").lower()
                cls_id = CLASS_MAP.get(cls_title)

                # Skips unknown classes
                if cls_id is None:
                    print(f"Nezināma klase '{cls_title}' anotācijā: {ann_file}")
                    continue

                exterior = obj["points"]["exterior"]
                (xmin, ymin), (xmax, ymax) = exterior

                bbox = [xmin, ymin, xmax, ymax]
                yolo = convert_bbox_to_yolo(w, h, bbox)

                out.write(f"{cls_id} {' '.join(map(str, yolo))}\n")

        print(f"Izveidots: {txt_path}")


def main():
    print("Sākas 'train' konversija...")
    process_split("train")

    print("Sākas 'val' konversija...")
    process_split("val")

    print("Konversija pabeigta! JSON → YOLO ir gatavs.")


if __name__ == "__main__":
    main()