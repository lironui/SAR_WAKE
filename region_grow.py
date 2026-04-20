import numpy as np
import cv2
import copy


class Point(object):
    """
    Simple Point class to represent pixel coordinates.
    """
    def __init__(self, x, y):
        self.x = x  # row index
        self.y = y  # column index

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def get_difference(img, current_point, new_point):
    """
    Compute absolute intensity difference between two pixels.

    Parameters
    ----------
    img : ndarray
        Input image.
    current_point : Point
        Current pixel.
    new_point : Point
        Neighbor pixel.

    Returns
    -------
    float
        Absolute difference in intensity.
    """
    return abs(img[current_point.x, current_point.y] - img[new_point.x, new_point.y])


def neighbor_comparison(image, target_point, mean_speed):
    """
    Check whether a pixel and its neighbors satisfy a minimum intensity condition.

    This is used as an additional constraint in high-gradient areas to avoid
    including noisy or low-value pixels.

    Parameters
    ----------
    image : ndarray
        Wind speed image.
    target_point : Point
        Pixel to evaluate.
    mean_speed : float
        Current mean intensity of the region.

    Returns
    -------
    bool
        True if the pixel passes the neighbor consistency test.
    """
    if image[target_point.x, target_point.y] < mean_speed / 2:
        return False

    # Check 8-neighborhood
    for point_shift in [
        Point(-1, -1), Point(0, -1), Point(1, -1),
        Point(1, 0), Point(1, 1), Point(0, 1),
        Point(-1, 1), Point(-1, 0)
    ]:
        if image[target_point.x + point_shift.x,
                 target_point.y + point_shift.y] < mean_speed / 2:
            return False
        else:
            return True


def select_connects(p):
    """
    Select connectivity pattern (4-connected or 8-connected).

    Parameters
    ----------
    p : int
        Connectivity flag (0 for 4-connectivity, otherwise 8-connectivity).

    Returns
    -------
    list of Point
        Neighbor offsets.
    """
    if p != 0:
        # 8-connected neighborhood
        connects = [
            Point(-1, -1), Point(0, -1), Point(1, -1),
            Point(1, 0), Point(1, 1), Point(0, 1),
            Point(-1, 1), Point(-1, 0)
        ]
    else:
        # 4-connected neighborhood
        connects = [
            Point(0, -1), Point(1, 0),
            Point(0, 1), Point(-1, 0)
        ]
    return connects


def region_grow(image, seeds, threshold, search_list,
                gradient_threshold, gradient, p=1, show_mask=True):
    """
    Region Growing algorithm for wake detection.

    This function expands from initial seed points to extract wake regions
    based on intensity similarity, gradient constraints, and a predefined
    search domain.

    Parameters
    ----------
    image : ndarray
        Input wind speed image.
    seeds : list of Point
        Initial seed points.
    threshold : tuple (lower, upper)
        Ratio threshold for region growing (relative to current mean intensity).
    search_list : list
        List of allowed pixels defining the search region.
    gradient_threshold : float
        Threshold to filter high-gradient areas.
    gradient : ndarray
        Gradient image.
    p : int, optional
        Connectivity type (default: 8-connected).
    show_mask : bool, optional
        Whether to display mask (not used here, reserved for visualization).

    Returns
    -------
    seed_mask : ndarray
        Binary mask of the grown region.
    mask_mean : float
        Mean intensity of the extracted region.
    box : ndarray
        Minimum bounding rectangle (4 points).
    """

    height, width = image.shape
    search_list = copy.deepcopy(search_list)

    seed_mask = np.zeros(image.shape)
    seed_list = []

    mask_sum = 0

    # Adaptive gradient threshold (for large search regions)
    if len(search_list) > 100:
        gradient_list = np.array(search_list)
        gradient_threshold = gradient[
            gradient_list[:, 1], gradient_list[:, 0]
        ].sum() / len(search_list)

    # Initialize seeds
    for seed in seeds:
        if seed.x in range(0, height) and seed.y in range(0, width):
            if gradient[seed.x, seed.y] < gradient_threshold and image[seed.x, seed.y] > 1:
                seed_list.append(seed)
                mask_sum += image[seed.x, seed.y]

    mask_amount = len(seed_list)

    if mask_amount == 0:
        return seed_mask, 0, 0

    else:
        mask_mean = mask_sum / mask_amount
        label = 1

        directions = 8 if p != 0 else 4
        connects = select_connects(p)

        # --- Region Growing Loop ---
        while len(seed_list) > 0:
            current_point = seed_list.pop(0)

            seed_mask[current_point.x, current_point.y] = label

            for i in range(directions):
                new_x = current_point.x + connects[i].x
                new_y = current_point.y + connects[i].y

                if new_x in range(0, height) and new_y in range(0, width):

                    # Relative intensity difference (ratio)
                    difference = image[new_x, new_y] / mask_mean

                    # Threshold condition
                    result = threshold[1] > difference > threshold[0]

                    # Additional constraints:
                    # 1. Not visited
                    # 2. Inside search region
                    result = result and seed_mask[new_x, new_y] == 0 \
                             and ([new_y, new_x] in search_list)

                    # Strong gradient constraint (noise suppression)
                    if gradient[new_x, new_y] > gradient_threshold * 2:
                        result = result and neighbor_comparison(
                            image, Point(new_x, new_y), mask_mean
                        )

                    # Accept pixel
                    if result:
                        seed_mask[new_x, new_y] = label
                        seed_list.append(Point(new_x, new_y))

                        mask_sum += image[new_x, new_y]
                        mask_amount += 1
                        mask_mean = mask_sum / mask_amount

        # --- Extract bounding box ---
        coords = np.column_stack(np.where(seed_mask != 0))
        min_rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(min_rect)

        return seed_mask, mask_mean, box