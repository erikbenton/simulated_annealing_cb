from random import random
import math

number_of_solutions = 100

class Point:
    def __init__(self, x_in, y_in):
        self.x: float = x_in
        self.y: float = y_in

    def distance(self, point):
        return math.sqrt(self.distance_squared(point))

    def distance_squared(self, point):
        return (self.x - point.x) ** 2 + (self.y - point.y) ** 2

    def closest(self, point1, point2):
        da: float = point2.y - point1.y
        db: float = point2.x - point1.x
        c1: float = da * point1.x + db * point1.y
        c2: float = -db * self.x + da * self.y
        det: float = da ** 2 + db ** 2
        if det != 0:
            cx: float = (da * c1 - db * c2) / det
            cy: float = (da * c2 + db * c1) / det
        else:
            cx: float = self.x
            cy: float = self.y
        return Point(cx, cy)


class Unit(Point):
    def __init__(self, x_in, y_in, vx_in, vy_in, id_in, radius_in, num_targets_in):
        super(Unit, self).__init__(x_in, y_in)
        self.next_target_id: int = id_in
        self.radius: float = radius_in
        self.vx: float = vx_in
        self.vy: float = vy_in
        self.num_targets: int = num_targets_in

    def collision(self, unit):
        # print("Collision", file=sys.stderr)
        distance: float = self.distance_squared(unit)
        radius_sum: float = (self.radius + unit.radius) ** 2
        if distance < radius_sum:  # Objects already touching
            return Collision(self, unit, 0.0)
        if self.vx == unit.vx and self.vy == unit.vy:  # Parallel objects
            return None
        collision_x: float = self.x - unit.x
        collision_y: float = self.y - unit.y
        collision_point: Point = Point(collision_x, collision_y)
        collision_vx: float = self.vx - unit.vx
        collision_vy: float = self.vy - unit.vy
        unit_point: Point = Point(0, 0)
        closest_point: Point = unit_point.closest(collision_point,
                                                  Point(collision_x + collision_vx, collision_y + collision_vy))
        closest_point_distance: float = unit_point.distance_squared(closest_point)
        collision_point_distance: float = collision_point.distance_squared(closest_point)
        if closest_point_distance < radius_sum:
            length: float = math.sqrt(collision_vx ** 2 + collision_vy ** 2)
            backup_distance: float = math.sqrt(radius_sum - closest_point_distance)
            closest_point.x = closest_point.x - backup_distance * (collision_vx / length)
            closest_point.y = closest_point.y - backup_distance * (collision_vy / length)
            if collision_point.distance_squared(closest_point) > collision_point_distance:
                return None
            closest_point_distance = closest_point.distance(collision_point)
            if closest_point_distance > length:
                return None
                # print(str(t), file=sys.stderr)
            t: float = closest_point_distance / length
            return Collision(self, unit, t)
        return None


class Checkpoint(Unit):
    def __init__(self, x_in, y_in, id_in, radius_in, num_targets_in):
        super(Checkpoint, self).__init__(x_in, y_in, 0.0, 0.0, id_in, radius_in, num_targets_in)

    def bounce(self, unit):
        return


class Move:
    def __init__(self, place):
        position: Point = place
        thrust: int = 0
        angle: float = 0.0
        score: float = 0.0

    def score(self, final_place):
        distance = self.position.distance(final_place)


class Solution:
    def __init__(self):
        score_one: float = 0.0
        score_two: float = 0.0
        score_three: float = 0.0
        score_four: float = 0.0
        moves_one: list = []
        moves_two: list = []
        moves_three: list = []
        moves_four: list = []


class Collision:
    def __init__(self, unit1, unit2, t_factor):
        self.unit_a: Unit = unit1
        self.unit_b: Unit = unit2
        self.time: float = t_factor


def acceptance_probability(old, new, temp):
    return math.exp((new - old)/temp)


def cost


def anneal(solution):
    old_cost = cost(solution)
    temperature = 1.0
    minimum_temperature = 0.00001
    cooling_rate = 0.9
    while temperature > minimum_temperature:
        for i in range(number_of_solutions):
            new_solution = neighbor(solution)
            new_cost = cost(new_solution)
            acceptance = acceptance_probability(old_cost, new_cost, temperature)
            if acceptance > random():
                solution = new_solution
                old_cost = new_cost
            temperature = temperature * cooling_rate
    return (solution, cost)