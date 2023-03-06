import cv2


COLORS = {      # BGR
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (0, 255, 128),
    3: (255, 255, 0),
    4: (255, 128, 0),
    5: (255, 0, 255),
    6: (102, 0, 204),
    7: (255, 153, 255),
    8: (204, 0, 0),
    9: (153, 153, 255),
    10: (153, 0, 153)
}


def visualize_different_annotators(img_info):
    merged_img = img_info.get_img()
    raw_img = merged_img.copy()

    annot_imgs = []
    for id, annots in img_info.annots.items():
        color = COLORS[id]
        annot_img = raw_img.copy()
        for vehicle in annots.vehicles:
            for inst in vehicle.instances:
                x1, y1, x2, y2 = inst.pos
                merged_img = cv2.rectangle(merged_img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                annot_img = cv2.rectangle(annot_img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        annot_imgs.append(annot_img)
                
    return merged_img, annot_imgs


if __name__ == "__main__":
    from data_analysis.core.meta import DirectVehicle, IndirectVehicle
    import os
    import pandas as pd
    import numpy as np
    from data_analysis.core.dataset import AI4ODDataset

    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original/"
    assert os.path.isdir(data_dir), "Your data directory is not a directory."

    dataset = AI4ODDataset(data_dir, load_images=False)

    # get list of all annotators
    annotator_ids = []
    for img_info in dataset.img_infos:
        ids = list(img_info.annots.keys())
        annotator_ids += ids
    
    for _, img_info in dataset:
        if img_info.img_name == '2021-11-18_18-20-04-015168.jpg':
            merged_img, annot_imgs = visualize_different_annotators(img_info)

            diff = annot_imgs[0] - annot_imgs[1]
            print(np.count_nonzero(diff))