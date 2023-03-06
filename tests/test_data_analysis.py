import unittest

from data_analysis.core.meta import ImageAnnotation, DirectVehicle, IndirectVehicle, Reflection, ReflectionType, ImageInfo, Scene
from data_analysis.boxes.utils import count_num_boxes, jaccard as jaccard_boxes
from data_analysis.boxes.match import get_matches, get_distance_for_combinations

class TestCountNumBoxes(unittest.TestCase):
    def test_error(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            DirectVehicle((30, 30, 40, 40), 0, instances),
            IndirectVehicle([20, 20], 1, instances),
        ]
        annot = ImageAnnotation("peter", vehicles)
        self.assertRaises(ValueError, count_num_boxes, annot, False, False)

    def test_indirect(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            DirectVehicle((30, 30, 40, 40), 0, instances),
            IndirectVehicle([20, 20], 1, instances),
        ]
        annot = ImageAnnotation("peter", vehicles)
        self.assertEqual(count_num_boxes(annot, indirect=True, direct=False), 6)
    
    def test_direct(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            DirectVehicle((30, 30, 40, 40), 0, instances),
            IndirectVehicle([20, 20], 1, instances),
        ]
        annot = ImageAnnotation("peter", vehicles)
        self.assertEqual(count_num_boxes(annot, indirect=False, direct=True), 1)
    
    def test_both(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            DirectVehicle((30, 30, 40, 40), 0, instances),
            IndirectVehicle([20, 20], 1, instances),
        ]
        annot = ImageAnnotation("peter", vehicles)
        self.assertEqual(count_num_boxes(annot, indirect=True, direct=True), 7)


class TestJaccardBoxes(unittest.TestCase):
    def test_empty(self):
        boxes = []
        self.assertRaises(ValueError, jaccard_boxes, boxes)
    
    def test_one(self):
        boxes = [[10, 10, 20, 20]]
        self.assertRaises(ValueError, jaccard_boxes, boxes)
    
    def test_full_iou(self):
        boxes = [
            [[10, 10, 20, 20], [10, 10, 20, 20]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]]
        ]
        for b in boxes:
            self.assertEqual(jaccard_boxes(b), 1.0)
    
    def test_partial_iou(self):
        boxes = [
            [[10, 10, 20, 20], [10, 10, 20, 15]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 15]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 15]]
        ]
        for b in boxes:
            self.assertEqual(jaccard_boxes(b), 0.5)
    
    def test_zero_iou(self):
        boxes = [
            [[10, 10, 20, 20], [20, 20, 30, 30]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [20, 20, 30, 30]],
            [[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20], [20, 20, 30, 30]]
        ]
        for b in boxes:
            self.assertEqual(jaccard_boxes(b), 0.0)
        

class TestMatcher(unittest.TestCase):
    def test_empty(self):
        instances = [
            # Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            # Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            # Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            IndirectVehicle([20, 20], 1, instances),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles),
            1: ImageAnnotation("hans", vehicles),
            2: ImageAnnotation("martin", vehicles)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=True)

        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched_instances), 0)

    def test_equal_instances(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles = [
            IndirectVehicle([20, 20], 1, instances),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles),
            1: ImageAnnotation("hans", vehicles),
            2: ImageAnnotation("martin", vehicles)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=True)

        self.assertEqual(len(matches), 3)
        self.assertEqual(len(unmatched_instances), 0)
    
    def test_unequal_instances(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((20, 20, 30, 30), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles0 = [
            IndirectVehicle([20, 20], 1, instances),
        ]
        vehicles1 = [
            IndirectVehicle([20, 20], 1, []),
        ]
        vehicles2 = [
            IndirectVehicle([20, 20], 1, []),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles0),
            1: ImageAnnotation("hans", vehicles1),
            2: ImageAnnotation("martin", vehicles2)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=True)

        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched_instances), 3)

    def test_exclusive(self):
        instances = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 0, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 1, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 2, False),
        ]
        vehicles0 = [
            IndirectVehicle([20, 20], 1, instances),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles0),
            1: ImageAnnotation("hans", vehicles0),
            2: ImageAnnotation("martin", vehicles0)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=True)

        self.assertEqual(len(matches), 3)
        self.assertEqual(len(unmatched_instances), 0)

    def test_non_exclusive(self):
        instances0 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 10, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 11, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 12, False),
        ]
        instances1 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 20, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 21, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 22, False),
        ]
        instances2 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 30, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 31, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 32, False),
        ]
        vehicles0 = [
            IndirectVehicle([20, 20], 1, instances0),
        ]
        vehicles1 = [
            IndirectVehicle([20, 20], 1, instances1),
        ]
        vehicles2 = [
            IndirectVehicle([20, 20], 1, instances2),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles0),
            1: ImageAnnotation("hans", vehicles1),
            2: ImageAnnotation("martin", vehicles2)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=False)

        self.assertEqual(len(matches), 9)
        self.assertEqual(len(unmatched_instances), 0)
    
    def test_direct_exclusive(self):
        instances0 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 10, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 11, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 12, False),
        ]
        instances1 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 20, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 21, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 22, False),
        ]
        instances2 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 30, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 31, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 32, False),
        ]
        vehicles0 = [
            DirectVehicle([20, 20, 30, 30], 1, instances0),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        vehicles1 = [
            DirectVehicle([20, 20, 30, 30], 1, instances1),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        vehicles2 = [
            DirectVehicle([20, 20, 30, 30], 1, instances2),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles0),
            1: ImageAnnotation("hans", vehicles1),
            2: ImageAnnotation("martin", vehicles2)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=True, indirect=False, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=True)

        self.assertEqual(len(matches), 2)
        self.assertEqual(len(unmatched_instances), 0)

    def test_direct_non_exclusive(self):
        instances0 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 10, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 11, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 12, False),
        ]
        instances1 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 20, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 21, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 22, False),
        ]
        instances2 = [
            Reflection((10, 10, 20, 20), ReflectionType.CAR, 30, False),
            Reflection((15, 15, 20, 20), ReflectionType.FLOOR, 31, False),
            Reflection((30, 30, 40, 40), ReflectionType.LINE, 32, False),
        ]
        vehicles0 = [
            DirectVehicle([20, 20, 30, 30], 1, instances0),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        vehicles1 = [
            DirectVehicle([20, 20, 30, 30], 1, instances1),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        vehicles2 = [
            DirectVehicle([20, 20, 30, 30], 1, instances2),
            DirectVehicle([15, 15, 30, 30], 1, instances0),
        ]
        annots = {
            0: ImageAnnotation("peter", vehicles0),
            1: ImageAnnotation("hans", vehicles1),
            2: ImageAnnotation("martin", vehicles2)
        }

        img_info = ImageInfo("test.jpg", "", scene=None, annots=annots)
        matches, unmatched_instances = get_matches(img_info, direct=True, indirect=False, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=False)

        self.assertEqual(len(matches), 8)
        self.assertEqual(len(unmatched_instances), 0)



if __name__ == "__main__":
    unittest.main()
