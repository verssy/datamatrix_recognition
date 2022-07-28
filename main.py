from multiprocessing.dummy import Manager, Process
import os
import cv2
import numpy as np
from threading import Thread
from queue import Queue
import math
from operator import add
from functools import reduce, partial
from random import randint
from PyQt5.QtGui import QColor
import pylibdmtx.pylibdmtx as pylibdmtx
import pytesseract.pytesseract as pytesseract
import re

pytesseract.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class FinderPattern:
    def __init__(self, corner, vec_base, vec_side):
        self.corner = corner
        self.baseVector = vec_base
        self.sideVector = vec_side
        self.baseLength = vec_base.length()
        self.sideLength = vec_side.length()
        self.c1 = corner
        self.c2 = corner + vec_base
        self.c3 = corner + vec_side
        self.center = (corner + ((vec_base + vec_side) / 2.0)).intify()
        self.radius = corner.distance_to(self.center)
    def point_in_radius(self, point):
        return self.bounds().contains_point(point)
    def bounds(self):
        return Circle(self.center, self.radius)
    def draw_to_image(self, image, color=None):
        if color is None:
            color = Color.Green()
        image.draw_line(self.c1, self.c2, color, 1)
        image.draw_line(self.c3, self.c1, color, 1)
    def correct_lengths(self, expected_length):
        if abs(self.baseLength - expected_length) < abs(self.sideLength - expected_length):
            factor = self.baseLength / self.sideLength
            new_base_vec = self.baseVector
            new_side_vec = self.sideVector * factor
        else:
            factor = self.sideLength / self.baseLength
            new_base_vec = self.baseVector * factor
            new_side_vec = self.sideVector
        return FinderPattern(self.corner, new_base_vec, new_side_vec)
class Circle:
    def __init__(self, center, radius):
        self._center = center
        self._radius = radius
    def __str__(self):
        return "Circle - center = ({:.2f}, {:.2f}); radius = {:.2f}".format(self.x(), self.y(), self._radius)
    def center(self):
        return self._center
    def radius(self):
        return self._radius
    def x(self):
        return self._center.x
    def y(self):
        return self._center.y
    def diameter(self):
        return self._radius * 2
    def circumference(self):
        return 2 * math.pi * self._radius
    def area(self):
        return math.pi * (self._radius ** 2)
    def offset(self, point):
        """ Returns a new circle which is the same size as this one but offset (moved by the specified amount). """
        return Circle(self._center + point, self._radius)
    def scale(self, factor):
        """ Returns a new circle which is a scaled version of this one. """
        return Circle(self._center, self._radius * factor)
    def contains_point(self, point):
        """ Returns true if the specified point is within the Circle's radius"""
        radius_sq = self._radius ** 2
        distance_sq = point.distance_to_sq(self._center)
        return distance_sq < radius_sq
    def intersects(self, circle):
        """ Returns true if the two circles intersect. """
        center_sep_sq = self._center.distance_to_sq(circle.center())
        radius_sum_sq = (self.radius() + circle.radius()) ** 2
        return center_sep_sq < radius_sum_sq
    def serialize(self):
        return "{}:{}:{}".format(self.x(), self.y(), self._radius)
    @staticmethod
    def deserialize(string):
        tokens = string.split(":")
        x = float(tokens[0])
        y = float(tokens[1])
        r = float(tokens[2])
        center = Point(x, y)
        return Circle(center, r)
class Color:
    SEP = ","
    CONSTRUCTOR_ERROR = "Values must be integers in range 0-255"
    STRING_PARSE_ERROR = "Input string must be 3 or 4 integers (0-255) separated by '{}'".format(SEP)
    def __init__(self, r, g, b, a=255):
        try:
            r, g, b, a = int(r), int(g), int(b), int(a)
        except ValueError:
            raise ValueError(self.CONSTRUCTOR_ERROR)
        for val in [r, g, b, a]:
            if val < 0 or val > 255:
                raise ValueError(self.CONSTRUCTOR_ERROR)
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    def __str__(self):
        return "{1}{0}{2}{0}{3}{0}{4}".format(self.SEP, self.r, self.g, self.b, self.a)
    def bgra(self):
        return self.b, self.g, self.r, self.a
    def bgr(self):
        return self.b, self.g, self.r
    def mono(self):
        return int(round(0.3*self.r + 0.6*self.g + 0.1*self.b))
    def to_qt(self):
        return QColor(self.r, self.g, self.b, self.a)
    def to_hex(self):
        hex_str = '#'
        for val in [self.r, self.g, self.b]:
            hex_str += '{:02x}'.format(val)
        return hex_str
    def rgb(self):
        return self.r, self.g, self.b
    @staticmethod
    def from_qt(qt_color):
        return Color(qt_color.red(), qt_color.green(), qt_color.blue(), qt_color.alpha())
    @staticmethod
    def from_string(string, sep=SEP):
        tokens = string.split(sep)
        if len(tokens) == 3:
            r, g, b = tuple(tokens)
            color = Color(r, g, b)
        elif len(tokens) == 4:
            r, g, b, a = tuple(tokens)
            color = Color(r, g, b, a)
        else: raise ValueError(Color.STRING_PARSE_ERROR)
        return color
    @staticmethod
    def Random():
        return Color(randint(0, 255), randint(0, 255), randint(0, 255), 255)
    @staticmethod
    def TransparentBlack(): return Color(0, 0, 0, 0)
    @staticmethod
    def TransparentWhite(): return Color(255, 255, 255, 0)
    @staticmethod
    def White(): return Color(255, 255, 255)
    @staticmethod
    def Black(): return Color(0, 0, 0)
    @staticmethod
    def Grey(): return Color(128, 128, 128)
    @staticmethod
    def Blue(): return Color(0, 0, 255)
    @staticmethod
    def Red(): return Color(255, 0, 0)
    @staticmethod
    def Green(): return Color(0, 255, 0)
    @staticmethod
    def Yellow(): return Color(255, 255, 0)
    @staticmethod
    def Cyan(): return Color(0, 255, 255)
    @staticmethod
    def Magenta(): return Color(255, 0, 255)
    @staticmethod
    def Orange(): return Color(255, 128, 0)
    @staticmethod
    def Purple(): return Color(128, 0, 255)
class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    def __neg__(self):
        """ Unary minus operator. """
        return Point(-self.x, -self.y)
    def __add__(self, p):
        """ PointA + PointB addition operator. """
        return Point(self.x+p.x, self.y+p.y)
    def __sub__(self, p):
        """ PointA - PointB subtraction operator. """
        return Point(self.x-p.x, self.y-p.y)
    def __mul__(self, scalar):
        """ PointA * scalar multiplication operator. """
        return Point(self.x*scalar, self.y*scalar)
    def __div__(self, scalar):
        """ PointA / scalar division operator. """
        return Point(self.x/scalar, self.y/scalar)
    def __floordiv__(self, scalar):
        """ PointA / scalar integer division operator. """
        return Point(self.x//scalar, self.y//scalar)
    def __truediv__(self, scalar):
        """ PointA / scalar true (float) division operator. """
        return Point(self.x/scalar, self.y/scalar)
    def __str__(self):
        """ Human-readable string representation. """
        return "({:.2f}, {:.2f})".format(self.x, self.y)
    def __repr__(self):
        """ Unambiguous string representation. """
        return "{}({}, {})".format(self.__class__.__name__, self.x, self.y)
    def length(self):
        """ Distance from the origin to the point. """
        return math.sqrt(self.length_sq())
    def length_sq(self):
        """ Square of the distance from the origin to the point. """
        return self.x**2 + self.y**2
    def distance_to(self, p):
        """ Distance between the two points. """
        return (self - p).length()
    def distance_to_sq(self, p):
        """ Square of the distance between the two points. """
        return (self - p).length_sq()
    def scale(self, factor):
        """ Returns a scaled version of the Point (from the origin). """
        return Point(self.x*factor, self.y*factor)
    def intify(self):
        """ Return a new point which is the same as this but with (rounded) integer coordinates. """
        return Point(int(round(self.x, 0)), int(round(self.y, 0)))
    def floatify(self):
        """ Return a new point which is the same as this but with float coordinates. """
        return Point(float(self.x), float(self.y))
    def tuple(self):
        """ Return the coordinates as an (x, y) tuple. """
        return self.x, self.y
    @staticmethod
    def from_array(arr):
        """ Create a new point from a length-2 array of x,y coordinates. """
        return Point(arr[0], arr[1])

class ColdCameraReader(object):
    def __init__(self, ip, login, password):
        self.stream = cv2.VideoCapture(f'rtsp://{login}:{password}@{ip}')
        self.stopped = False
        self.Q = Queue(3)
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        try:
            while True:
                if self.stopped: return
                if not self.Q.full():
                    _, image = self.stream.read()
                    if not _:
                        self.stop()
                        return
                    self.Q.put(image)
                    while self.Q.qsize() > 1: self.Q.get()
        except Exception as e: print("error: " + str(e))
    def read(self): return self.Q.get()
    def more(self): return self.Q.qsize() > 0
    def stop(self): self.stopped = True
    def __exit__(self, exception_type, exception_value, traceback): self.stream.release()


DIM=(1576, 663)
K=np.array([[935.9870786585898, 0.0, 788.7869957151664], [0.0, 662.6319360114222, 341.7118627563714], [0.0, 0.0, 1.0]])
D=np.array([[-0.17574049635631422], [0.18891297353073158], [-0.26821132512804], [0.15348226070040832]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
def anti_fisheye(frame) -> cv2.Mat:
    frame = cv2.resize(frame, DIM)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_frame

## FILTERS
def filter_more_than_six(edge_set):
    return len(edge_set) > 6
def filter_longest_adjacent(edge_set):
    i, j = longest_pair_indices(edge_set)
    return abs(i - j) in (1, len(edge_set) - 1)
def filter_longest_approx_orthogonal(edge_set):
    i, j = longest_pair_indices(edge_set)
    v_i, v_j = (np.subtract(*edge_set[x]) for x in (i, j))
    #return abs(_cosine(v_i, v_j)) < 0.15
    return abs(_cosine(v_i, v_j)) < 0.15 # ~16 degrees
def filter_longest_similiar_in_length(edge_set):
    i, j = longest_pair_indices(edge_set)
    l_i, l_j = (_length(edge_set[x]) for x in (i, j))
    return abs(l_i - l_j)/abs(l_i + l_j) < 0.2
def pairs_circular(iterable):
    iterator = iter(iterable)
    x = next(iterator)
    zeroth = x
    while True:
        try:
            y = next(iterator)
            yield((x, y))
        except StopIteration:
            try:
                yield((y, zeroth))
            except UnboundLocalError:
                pass
            break
        x = y
def polygons_to_edges(vertices):
    return list(pairs_circular(vertices))
def _length(edge):
    return _distance(*edge)
def _distance(a, b):
    return _modulus(np.subtract(a, b))
def _modulus(vector):
    return quadrature_add(*vector)
def _cosine(a, b):
    return np.dot(a, b) / (_modulus(a) * _modulus(b))
def quadrature_add(*values):
    return math.sqrt(reduce(add, (c * c for c in values)))
def longest_pair_indices(edge_set):
    lengths = list(map(_length, edge_set))
    return np.asarray(lengths).argsort()[-2:][::-1]

## FINDER

def get_finder_pattern(edge_set):
    i, j = longest_pair_indices(edge_set)
    pair_longest_edges = [edge_set[x] for x in (i, j)]
    x_corner = get_shared_vertex(*pair_longest_edges)
    c, d = map(partial(get_other_vertex, x_corner), pair_longest_edges)
    vec_c, vec_d = map(partial(np.add, -x_corner), (c, d))
    if vec_c[0] * vec_d[1] - vec_c[1] * vec_d[0] < 0: vec_base, vec_side = vec_c, vec_d
    else: vec_base, vec_side = vec_d, vec_c
    x_corner = Point(x_corner[0], x_corner[1]).intify()
    vec_base = Point(vec_base[0], vec_base[1]).intify()
    vec_side = Point(vec_side[0], vec_side[1]).intify()
    return FinderPattern(x_corner, vec_base, vec_side)
def get_shared_vertex(edge_a, edge_b):
    for v_a in edge_a:
        for v_b in edge_b:
            if (v_a == v_b).all():
                return v_a
def get_other_vertex(vertex, edge):
    for v_a in edge:
        if not (v_a == vertex).all():
            return v_a

def vector_to_degrees(point_a, point_b):
    vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    veccos = vector[0] / math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    rads = math.acos(veccos)
    degrs = round(rads * 180.0 / math.pi)
    if vector[1] < 0:
        degrs = 360 - int(degrs)
    return degrs
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))
def gs1_is_valid(string: str):
    regex_filter = r'01[0-9]{14}21\S{10}93\S{4}'
    result = re.match(regex_filter, string)
    if not result:
        return False
    return True
def decrease_resolution_by_ratio(resx, resy, expec_area):
    res_ratio = resx / resy
    height = int(math.sqrt(expec_area / res_ratio))
    width = int(height * res_ratio)
    return [width, height]

def process_frame(frame, bounds, thread_ptr, mutable_list):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 8)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, element, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(rc, 6.0, True).reshape(-1, 2) for rc in contours]
    edge_sets = map(polygons_to_edges, polygons)
    edge_sets = filter(filter_more_than_six, edge_sets)
    edge_sets = filter(filter_longest_adjacent, edge_sets)
    edge_sets = filter(filter_longest_approx_orthogonal, edge_sets)
    edge_sets = filter(filter_longest_similiar_in_length, edge_sets)
    fps = [get_finder_pattern(es) for es in edge_sets]

    #image_data = pytesseract.image_to_string(frame, config='-l eng --psm 6 --oem 3')
    #m = re.findall('[0-3][0-9][.,\/-][0-1][0-9][.,\/-][0-9][0-9]', image_data)
    #if len(m) > 0:
    #    str_m = m[0]
    #    str_m = str_m.replace('/', '.')
    #    str_m = str_m.replace(',', '.')
    #    str_m = str_m.replace('-', '.')
    #    print(str_m)

    for finder in fps:
        if finder.c1.x > 5 and finder.c1.y > 5 and finder.c1.x < bounds[1] - 5 and finder.c1.y < bounds[0] - 5:
            angel = vector_to_degrees((0, 0), (finder.c3.x - finder.c1.x, finder.c1.y - finder.c3.y)) - 180
            circle_coords = (finder.bounds().x(), finder.bounds().y())
            circle_radius = finder.bounds().radius()
            x_min = int(max(circle_coords[0] - circle_radius - 10, 0))
            x_max = int(min(circle_coords[0] + circle_radius + 10, bounds[1]))
            y_min = int(max(circle_coords[1] - circle_radius - 10, 0))
            y_max = int(min(circle_coords[1] + circle_radius + 10, bounds[0]))
            temp_frame = frame[y_min:y_max, x_min:x_max]
            decoded = pylibdmtx.decode(temp_frame, max_count=1)
            if len(decoded) > 0:
                text = decoded[0].data.decode('UTF-8')
                text = text.replace('\x1d', '\\x1d')
                print(text, end=' ')
#   TODO: ISPRAVIT' !!!
                if gs1_is_valid(text): print('valid')
                else: print(':: Invalid ::')
                a = (finder.c1, finder.c2)
                b = (finder.c1, finder.c3)
                #cv2.line(frame, (a[0].x, a[0].y), (a[1].x, a[1].y), (127, 255, 63), 2)
                cv2.line(frame, (a[0].x, a[0].y), (a[1].x, a[1].y), (0, 0, 255), 2)
                cv2.line(frame, (b[0].x, b[0].y), (b[1].x, b[1].y), (0, 0, 255), 2)
    
    mutable_list[thread_ptr] = mutable_list[thread_ptr] + [frame]


def main():
    cold_reader = ColdCameraReader('192.168.20.190', 'admin', 'test12345')
    cold_reader.start()

    image = cold_reader.read()
    bounds = image.shape[:-1]
    print(bounds)
    bounds = decrease_resolution_by_ratio(bounds[1], bounds[0], 400 * 300)
    print(bounds)

    cpu_count = os.cpu_count()
    thread_pool = [Process()] * cpu_count
    thread_ptr = 0
    queue_ptr = 0

    manager = Manager()
    mutable_list = manager.list([[]] * cpu_count)
    end_flag =  False

    while True:
        image = cold_reader.read()
        #image = anti_fisheye(image)
        image = cv2.resize(image, (bounds[0], bounds[1]), interpolation=cv2.INTER_LINEAR)
        
        while True:
            if thread_pool[queue_ptr].is_alive():
                thread_pool[queue_ptr].join()
            if len(mutable_list[queue_ptr]) > 0:
                frame = mutable_list[queue_ptr][0]
                mutable_list[queue_ptr] = mutable_list[queue_ptr][1::]
                queue_ptr = (queue_ptr + 1) % cpu_count
                cv2.imshow('-', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    for process in thread_pool:
                        if process.is_alive():
                            process.join()
                    end_flag = True
                    break
            else:
                break
        
        if end_flag:
            break

        thread_pool[thread_ptr] = Process(target=process_frame, args=(image, bounds, thread_ptr, mutable_list))
        thread_pool[thread_ptr].start()
        thread_ptr = (thread_ptr + 1) % cpu_count

    cold_reader.stop()
    for process in thread_pool:
        if process.is_alive():
            process.join()


if __name__=='__main__':
    main()