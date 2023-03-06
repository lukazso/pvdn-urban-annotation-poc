import itertools

from data_analysis.core.meta import ImageInfo, DirectVehicle
from data_analysis.boxes.utils import jaccard


def get_distance_for_combinations(instances, distance_metric=jaccard):
    """Get distance for all combinations of instances.

    Args:
        instances (list): A list of lists of instances.
        distance_metric (function): A distance metric to compute distances between boxes. Default is jaccard.

    Returns:
        tuple: A tuple of lists containing all combinations and corresponding distances.
    """
    combs = list(itertools.product(*instances))
    ious = []
    for comb in combs:
        boxes = [item.pos for item in comb]
        iou = distance_metric(boxes)      
        ious.append(iou)
    return combs, ious


def get_matches(img_info: ImageInfo, direct: bool, indirect: bool, distance_metric=jaccard, iou_thresh: float = 0.2, exclusive: bool = True):
    """Get matched instances for the given image.

    Args:
        img_info (ImageInfo): An object containing image and its annotations.
        direct (bool): A boolean indicating whether to consider DirectVehicle instances or not.
        indirect (bool): A boolean indicating whether to consider Reflection instances or not.
        distance_metric (function): A distance metric to compute distances between boxes. Default is jaccard.
        iou_thresh (float): A float indicating IoU threshold for matching. Default is 0.2.
        exclusive (bool): A boolean indicating whether each instance can only be used in a single match or not. Default is True.

    Returns:
        tuple: A tuple of lists containing matched instances and unmatched instances.
    """
    if not direct and not indirect:
        raise ValueError("direct or indirect have to be true.")
    
    instances = []
    for id, annot in img_info.annots.items():
        if indirect:
            instances.append(annot.reflections)
        elif direct:
            instances.append([v for v in annot.vehicles if isinstance(v, DirectVehicle)])

    unmatched_instances = [i for insts in instances for i in insts]
    combs, ious = get_distance_for_combinations(instances, distance_metric)
    if len(combs) == 0:
        return [], unmatched_instances
    
    ious, combs = zip(*sorted(zip(ious, combs), key=lambda x: x[0], reverse=True))

    matches = []
    for iou, comb in zip(ious, combs):
        if iou < iou_thresh:
            break
        
        if exclusive:
            free = True
            for inst in comb:
                if not inst in unmatched_instances:
                    free = False
                    break
            if not free:
                continue

        for inst in comb:
            try:
                unmatched_instances.remove(inst)
            except ValueError:
                pass
        matches.append((comb, iou))
    
    return matches, unmatched_instances
