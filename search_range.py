"""
Search-region construction module for SAR-based wind farm wake detection.

This module defines:
1. the downstream wake search region, and
2. the upstream reference region,

based on wind direction, geometric constraints, and image masking conditions.
"""

import math
import numpy as np


def get_cross_condition(point1, point2, point):
    """
    Compute the 2D cross-product condition for a point relative to a line segment.

    Parameters
    ----------
    point1 : list or tuple
        First endpoint of the line segment [x, y].
    point2 : list or tuple
        Second endpoint of the line segment [x, y].
    point : list or tuple
        Query point [x, y].

    Returns
    -------
    float
        Cross-product value used to determine the relative position
        of the point with respect to the directed line.
    """
    return (point2[0] - point1[0]) * (point[1] - point1[1]) - \
           (point[0] - point1[0]) * (point2[1] - point1[1])


def is_point_in(point1, point2, point3, point4, point):
    """
    Determine whether a point lies inside a quadrilateral.

    The test is based on cross-product sign consistency.

    Parameters
    ----------
    point1, point2, point3, point4 : list or tuple
        Vertices of the quadrilateral [x, y].
    point : list or tuple
        Query point [x, y].

    Returns
    -------
    bool
        True if the point is inside the quadrilateral, otherwise False.
    """
    condition1 = get_cross_condition(point1, point2, point) * \
                 get_cross_condition(point3, point4, point) >= 0
    condition2 = get_cross_condition(point2, point3, point) * \
                 get_cross_condition(point4, point1, point) >= 0

    result = condition1 and condition2
    return result


class SearchList(object):
    """
    Generate the downstream wake search region.

    This class constructs a directional quadrilateral search area based on
    the wind direction, farm location, and predefined search range.
    Pixels inside this region are considered candidate wake pixels.
    """

    def __init__(self, k, x, y, wind_farm_mask, wake_direction, image, search_range=(200, 100)):
        """
        Parameters
        ----------
        k : float
            Slope of the wake direction line.
        x : float or int
            Reference x-coordinate.
        y : float or int
            Reference y-coordinate.
        wind_farm_mask : ndarray
            Binary mask of the wind farm area.
        wake_direction : int
            Direction indicator of wake extension.
        image : ndarray
            Input wind speed image.
        search_range : tuple, optional
            Search region size (length, width).
        """
        self.k = k
        self.x = round(x)
        self.y = round(y)
        self.wind_farm_mask = wind_farm_mask
        self.wake_direction = wake_direction
        self.image = image
        self.image_size = image.shape
        self.search_range = search_range

    def get_crossing_point(self, a, b, c):
        """
        Compute the intersection point of two lines.

        The line equations are expressed in the form:
            a1 * x + a2 * y + c1 = 0
            b1 * x + b2 * y + c2 = 0

        Parameters
        ----------
        a : list
            Coefficients of the first line.
        b : list
            Coefficients of the second line.
        c : list
            Constants of the two lines.

        Returns
        -------
        list
            Intersection point [x, y].
        """
        D = a[0] * b[1] - a[1] * b[0]
        x = (b[0] * c[1] - b[1] * c[0]) / D
        y = (a[1] * c[0] - a[0] * c[1]) / D
        return [round(x), round(y)]

    def forward(self, x):
        """
        Compute the y-coordinate on the wake-direction line for a given x.

        Parameters
        ----------
        x : float
            Input x-coordinate.

        Returns
        -------
        float
            Corresponding y-coordinate.
        """
        return self.k * (x - self.x) + self.y

    def get_boundary(self, x, y):
        """
        Compute the four boundary lines defining the wake search region.

        Parameters
        ----------
        x : float or int
            Reference x-coordinate.
        y : float or int
            Reference y-coordinate.

        Returns
        -------
        tuple
            Four boundary constants defining the directional quadrilateral.
        """
        x_shift = np.sqrt(self.search_range[0] ** 2 / (self.k ** 2 + 1))
        new_x = x + x_shift * self.wake_direction
        new_y = self.forward(new_x)

        # Two parallel boundaries
        t1 = -self.k * x + y + self.search_range[1] // 2 * math.sqrt(self.k ** 2 + 1)
        t2 = -self.k * x + y - self.search_range[1] // 2 * math.sqrt(self.k ** 2 + 1)

        # Two perpendicular boundaries
        t3 = -x / self.k - y
        t4 = -new_x / self.k - new_y

        return t1, t2, t3, t4

    def get_list(self):
        """
        Generate the list of candidate pixels in the wake search region.

        Returns
        -------
        search_list : list
            List of pixel coordinates [x, y] inside the search region.
        mask : ndarray
            Binary mask of the search region.
        """
        search_list = []
        t1, t2, t3, t4 = self.get_boundary(self.x, self.y)
        mask = np.zeros(self.image_size)

        # Four vertices of the quadrilateral search region
        point1 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t1, t3])
        point2 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t2, t3])
        point3 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t2, t4])
        point4 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t1, t4])

        for x in range(self.image_size[1]):
            for y in range(self.image_size[0]):
                if is_point_in(point1, point2, point3, point4, [x, y]) and \
                        self.wind_farm_mask[y, x] != 1 and self.image[y, x] > 0.1:
                    search_list.append([x, y])
                    mask[y, x] = 1

        return search_list, mask


class UpstreamList(object):
    """
    Generate the upstream reference region.

    This class defines a directional quadrilateral region located upstream
    of the wind farm. It is used to extract reference wind conditions
    unaffected by the wake.
    """

    def __init__(self, k, x, y, gradient, gradient_threshold, image, wind_farm_mask,
                 wake_speed, upstream_direction, search_range=(200, 100)):
        """
        Parameters
        ----------
        k : float
            Slope of the upstream direction line.
        x : float or int
            Reference x-coordinate.
        y : float or int
            Reference y-coordinate.
        gradient : ndarray
            Gradient image.
        gradient_threshold : float
            Threshold for filtering high-gradient pixels.
        image : ndarray
            Input wind speed image.
        wind_farm_mask : ndarray
            Binary mask of the wind farm area.
        wake_speed : float
            Mean wind speed of the detected wake region.
        upstream_direction : int
            Direction indicator of upstream extension.
        search_range : tuple, optional
            Search region size (length, width).
        """
        self.k = k
        self.x = round(x)
        self.y = round(y)
        self.gradient = gradient
        self.gradient_threshold = gradient_threshold
        self.image = image
        self.image_size = image.shape
        self.wind_farm_mask = wind_farm_mask
        self.wake_speed = wake_speed
        self.upstream_direction = upstream_direction
        self.search_range = search_range

    def get_crossing_point(self, a, b, c):
        """
        Compute the intersection point of two lines.

        Parameters
        ----------
        a : list
            Coefficients of the first line.
        b : list
            Coefficients of the second line.
        c : list
            Constants of the two lines.

        Returns
        -------
        list
            Intersection point [x, y].
        """
        D = a[0] * b[1] - a[1] * b[0]
        x = (b[0] * c[1] - b[1] * c[0]) / D
        y = (a[1] * c[0] - a[0] * c[1]) / D
        return [round(x), round(y)]

    def forward(self, x):
        """
        Compute the y-coordinate on the upstream-direction line for a given x.

        Parameters
        ----------
        x : float
            Input x-coordinate.

        Returns
        -------
        float
            Corresponding y-coordinate.
        """
        return self.k * (x - self.x) + self.y

    def get_boundary(self, x, y):
        """
        Compute the four boundary lines defining the upstream region.

        Parameters
        ----------
        x : float or int
            Reference x-coordinate.
        y : float or int
            Reference y-coordinate.

        Returns
        -------
        tuple
            Four boundary constants defining the directional quadrilateral.
        """
        x_shift = np.sqrt(self.search_range[0] ** 2 / (self.k ** 2 + 1))
        new_x = x + x_shift * self.upstream_direction
        new_y = self.forward(new_x)

        # Two parallel boundaries
        t1 = -self.k * x + y + self.search_range[1] // 2 * math.sqrt(self.k ** 2 + 1)
        t2 = -self.k * x + y - self.search_range[1] // 2 * math.sqrt(self.k ** 2 + 1)

        # Two perpendicular boundaries
        t3 = -x / self.k - y
        t4 = -new_x / self.k - new_y

        return t1, t2, t3, t4

    def get_list(self):
        """
        Generate the list of valid pixels in the upstream reference region.

        Pixels are excluded if they:
        - belong to the wind farm,
        - have very low wind speed,
        - or are likely affected by strong gradients and wake-related disturbances.

        Returns
        -------
        upstream_list : list
            List of pixel coordinates [x, y] inside the upstream region.
        mask : ndarray
            Binary mask of the upstream region.
        float
            Mean wind speed within the upstream reference region.
        """
        upstream_list = []
        t1, t2, t3, t4 = self.get_boundary(self.x, self.y)
        mask = np.zeros(self.image_size)

        # Four vertices of the quadrilateral upstream region
        point1 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t1, t3])
        point2 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t2, t3])
        point3 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t2, t4])
        point4 = self.get_crossing_point([self.k, 1 / self.k], [-1, 1], [t1, t4])

        for x in range(self.image_size[1]):
            for y in range(self.image_size[0]):
                if is_point_in(point1, point2, point3, point4, [x, y]):
                    if (self.image[y, x] < self.wake_speed and self.gradient[y, x] > self.gradient_threshold) \
                            or self.wind_farm_mask[y, x] == 1 \
                            or self.image[y, x] < self.wake_speed * 0.5:
                        pass
                    else:
                        upstream_list.append([x, y])
                        mask[y, x] = 1

        upstream_area = np.where(mask != 0)

        if upstream_area[0].shape[0] == 0:
            return upstream_list, mask, 0
        else:
            return upstream_list, mask, self.image[upstream_area].sum() / upstream_area[0].shape[0]