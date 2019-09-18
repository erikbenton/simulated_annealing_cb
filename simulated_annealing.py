import sys
from random import random
import math
import numpy as np


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
    def __init__(self, x_in, y_in, vx_in, vy_in, id_in, radius_in, num_targets_in, name="N/A"):
        super(Unit, self).__init__(x_in, y_in)
        self.next_target_id: int = id_in
        self.radius: float = radius_in
        self.vx: float = vx_in
        self.vy: float = vy_in
        self.num_targets: int = num_targets_in
        self.name: str = name

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


class Pod(Unit):
    def __init__(self, x_in, y_in, vx_in, vy_in, angle_in, id_in, radius_in, num_targets_in, laps_in, name="N/A"):
        super(Pod, self).__init__(x_in, y_in, vx_in, vy_in, id_in, radius_in, num_targets_in, name)
        self.angle: float = angle_in
        self.checkpoint = None
        self.checked: int = 0
        self.timeout: int = 100
        self.partner = None
        self.shield: bool = False
        self.number_laps: int = laps_in
        self.solutions = []
        self.fitness: float = 0
        return

    def add_partner(self, pod):
        self.partner = pod
        return

    def get_angle(self, point):
        distance = self.distance(point)
        dx = (point.x - self.x) / distance
        dy = (point.y - self.y) / distance
        theta: float = math.acos(dx)
        theta = math.degrees(theta)
        if dy < 0:
            theta = 360 - theta
        return theta

    def difference_angle(self, point):
        theta = self.get_angle(point)
        right = self.angle - theta if self.angle <= theta else 360.0 - self.angle + theta
        left = self.angle - theta if self.angle <= theta else self.angle + 360.0 - theta
        if right < left:
            return right
        else:
            return -left

    def rotate(self, point):
        theta = self.get_angle(point)
        if theta > 18.0:
            theta = 18
        elif theta < -18.0:
            theta = -18
        self.angle += theta
        if self.angle >= 360.0:
            self.angle = self.angle - 360.0
        elif self.angle < 0.0:
            self.angle += 360.0
        return

    def boost(self, thrust):
        if self.shield:
            return
        theta_radians = math.radians(self.angle)
        self.vx += math.cos(theta_radians) * thrust
        self.vy += math.sin(theta_radians) * thrust
        return

    def move(self, t):
        self.x += self.vx * t
        self.y += self.vy * t
        return

    def end(self, checkpoints):
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = math.trunc(self.vx * 0.85)
        self.vy = math.trunc(self.vy * 0.85)
        self.timeout -= 1
        self.fitness = self.score(checkpoints)
        return self.fitness

    def play(self, point, thrust, checkpoints):
        self.fitness = 0
        self.checked = 0
        self.rotate(point)
        self.boost(thrust)
        self.move(1.0)
        self.end(checkpoints)
        return

    def bounce(self, unit):
        if isinstance(unit, Checkpoint):
            self.bounce_with_checkpoint()
        else:
            mass1: float = 10 if self.shield else 1
            mass2: float = 10 if unit.shield else 1
            mass_coefficient: float = (mass1 + mass2) / (mass1 * mass2)
            new_x: float = self.x - unit.x
            new_y: float = self.y - unit.y
            new_xy_squared: float = new_x ** 2 + new_y ** 2
            dvx: float = self.vx - unit.vx
            dvy: float = self.vy - unit.vy
            product: float = new_x * dvx + new_y * dvy
            fx: float = (new_x * product) / (new_xy_squared * mass_coefficient)
            fy: float = (new_y * product) / (new_xy_squared * mass_coefficient)
            self.vx -= fx / mass1
            self.vy -= fy / mass1
            unit.vx += fx / mass2
            unit.vy += fy / mass2
            impulse: float = math.sqrt(fx ** 2 + fy ** 2)
            if impulse < 120:
                fx = fx * 120.0 / impulse
                fy = fy * 120.0 / impulse
            self.vx -= fx / mass1
            self.vy -= fy / mass1
            unit.vx += fx / mass2
            unit.vy += fy / mass2
        return

    def bounce_with_checkpoint(self):
        self.next_target_id += 1
        if self.next_target_id == self.num_targets:
            self.next_target_id = 0
        self.timeout = 100
        self.checked += 1
        return

    def score(self, checkpoints):
        self.fitness = (self.checked * 50000 - self.distance(checkpoints[self.next_target_id]))
        return self.fitness

    def output(self, move):
        theta: float = self.angle + move.angle
        if theta >= 360.0:
            theta = theta - 360.0
        elif theta < 0.0:
            theta += 360.0
        theta = math.radians(theta)
        px: float = self.x + math.cos(theta) * 10000.0
        py: float = self.y + math.sin(theta) * 10000.0
        print(str(round(px)) + " " + str(round(py)) + " " + str(move.thrust))
        return

    def autopilot_point(self, target, next_target, radius=100):
        # Figure out the desired x & y
        # Set up vectors with the current target as the origin
        target_pod_x = self.x - target.x
        target_pod_y = self.y - target.y
        target_pod_vect = np.array([target_pod_x, target_pod_y])
        target_pod_vect_mag = math.sqrt(target_pod_x ** 2 + target_pod_y ** 2)
        # Set up vectors with the current target and the next target
        target_next_x = next_target.x - target.x
        target_next_y = next_target.y - target.y
        target_next_vect = np.array([target_next_x, target_next_y])
        target_next_vect_mag = math.sqrt(target_next_x ** 2 + target_next_y ** 2)
        # Get the bisecting vector
        unscaled_desired_vect_x = target_pod_vect_mag * target_next_vect[0] + target_next_vect_mag * target_pod_vect[0]
        unscaled_desired_vect_y = target_pod_vect_mag * target_next_vect[1] + target_next_vect_mag * target_pod_vect[1]
        unscaled_desired_vect_theta = math.atan2(unscaled_desired_vect_y, unscaled_desired_vect_x)
        desired_x = radius * math.cos(unscaled_desired_vect_theta) + target.x
        desired_y = radius * math.sin(unscaled_desired_vect_theta) + target.y
        # Create point from desired x & y
        target_point = Point(self.x + desired_x - self.x - self.vx, self.y + desired_y - self.y - self.vy)
        return target_point

    def autopilot_thrust(self, point, style):
        # Calculate the thrust
        distance = self.distance(point)
        scale_factor = 100 / math.pi
        stretch_factor = 0.002
        far_away = 2000
        approach_dist = 1200
        base = 45
        if distance > far_away:
            thrust = 100
        elif distance > approach_dist:
            if style == 0:
                thrust = (100 * (distance / (far_away - approach_dist)) - base) + base
            elif style == 1:
                thrust = scale_factor * math.atan(stretch_factor * (distance - approach_dist)) + base
            elif style == 2:
                thrust = 75
            elif style == 3:
                thrust = 25
            else:
                thrust = (100 * (distance / (far_away - approach_dist)) - base) + base
        else:
            thrust = base
        if 130 <= self.difference_angle(point) <= 230:
            thrust = 20
        return thrust

    def autopilot(self, target, next_target, checkpoints, style=0):
        # Determine if on target
        target_point = self.autopilot_point(target, next_target)
        theta = self.get_angle(target_point)
        print(str(self.name) + ", " + str(self.difference_angle(target_point)), file=sys.stderr)
        thrust = self.autopilot_thrust(target_point, style)
        # Play the turn
        # self.play(target_point, thrust, checkpoints)
        return Move(Point(self.x, self.y), target_point, theta, thrust)

    def on_target(self):

        return


class Move:
    def __init__(self, first_place, second_place, theta=0.0, speed=0):
        self.point_one: Point = first_place
        self.point_two: Point = second_place
        self.thrust: int = speed
        self.angle: float = theta
        self.score: float = 0.0

    def neighbor(self, amplitude):
        ramin: float = -18.0 * np.random.random() * amplitude
        ramax: float = 18.0 * np.random.random() * amplitude
        if ramin < -18.0:
            ramin = -18.0
        if ramax > 18.0:
            ramax = 18.0
        self.angle = (ramax - ramin) * np.random.random() + ramin
        self.point_two = self.neighbor_point(self.angle)
        pmin: int = self.thrust - 100 * amplitude
        pmax: int = self.thrust + 100 * amplitude
        if pmin < 0:
            pmin = 0
        if pmax > 0:
            pmax = 200
        self.thrust = (pmax - pmin) * np.random.random() + pmin
        return self

    def neighbor_point(self, angle):
        # print(str(angle), file=sys.stderr)
        dx = self.point_two.x - self.point_one.x
        dy = self.point_two.y - self.point_one.y
        # print(str(dx) + ", " + str(dy) + ", " + str(self.point_two.x + dx) + ", " + str(self.point_two.y + dy), file=sys.stderr)
        x2 = np.cos(math.radians(angle)) * dx - np.sin(math.radians(angle)) * dy
        y2 = np.sin(math.radians(angle)) * dx + np.cos(math.radians(angle)) * dy
        return Point(self.point_one.x + x2, self.point_one.y + y2)


class Solution:
    def __init__(self):
        self.score_one: list = []
        self.score_two: list = []
        self.score_three: list = []
        self.score_four: list = []
        self.moves_one: list = []
        self.moves_two: list = []
        self.moves_three: list = []
        self.moves_four: list = []
        self.scores: list = []
        self.moves: list = []
        self.package_up()

    def package_up(self):
        self.scores = [self.score_one, self.score_two, self.score_three, self.score_four]
        self.moves = [self.moves_one, self.moves_two, self.moves_three, self.moves_four]

    def score(self):
        sum_one = 0
        sum_two = 0
        sum_three = 0
        sum_four = 0
        for i in range(len(self.score_one)):
            sum_one += self.score_one[i]
        for i in range(len(self.score_two)):
            sum_two += self.score_two[i]
        for i in range(len(self.score_three)):
            sum_three += self.score_three[i]
        for i in range(len(self.score_four)):
            sum_four += self.score_four[i]
        return 1 / (((sum_one + sum_two) / 2) - ((sum_three + sum_four) / 2))

class Collision:
    def __init__(self, unit1, unit2, t_factor):
        self.unit_a: Unit = unit1
        self.unit_b: Unit = unit2
        self.time: float = t_factor


def acceptance_probability(old, new, temp):
    # print(str(new) + ", " + str(old) + ", " + str(temp), file=sys.stderr)
    return np.exp((new - old) / temp)


def cost(solution):
    # Package up the moves and scores for looping through them
    solution.package_up()
    # Loop through all the lists of moves
    for i in range(len(solution.moves)):
        number_moves = len(solution.moves[i])
        for j in range(number_moves):
            # Score is weighted by how "deep" in the solution the move is
            solution.scores[i][j] += ((number_moves - j) / number_moves) * solution.moves[i][j].score
    return solution.score()


def create_turn(pod, target, next_target, checkpoints, style=0):
    return pod.autopilot(target, next_target, checkpoints, style)


def neighbor(solution, pods, targets):
    clones = pods
    new_solution = Solution()
    new_solution.moves_one.append(solution.moves_one[0].neighbor(np.random.random()/2))
    new_solution.moves_two.append(solution.moves_two[0].neighbor(np.random.random()/2))
    new_solution.moves_three.append(solution.moves_three[0])
    new_solution.moves_four.append(solution.moves_four[0])
    # Determine where all the pods end up for projected turn and their scores/fitness
    score_results = determine_trajectory(clones, targets, [new_solution.moves_one[0], new_solution.moves_two[0],
                                                           new_solution.moves_three[0], new_solution.moves_four[0]])
    # Get all the scores for the first turn
    for q in range(len(score_results)):
        new_solution.scores[q].append(score_results[q])
    for i in range(1, len(solution.moves_one)):
        for p in range(len(clones)):
            # Get target id
            if clones[p].next_target_id + 1 == clones[p].num_targets:
                next_target = 0
            else:
                next_target = clones[p].next_target_id + 1
            # Create a turn then save it to the solution
            new_solution.moves[p].append(create_turn(clones[p], targets[clones[p].next_target_id],
                                                     targets[next_target], targets, 0))
        # Determine where all the pods end up for projected turn and their scores/fitness
        score_results = determine_trajectory(clones, targets, [new_solution.moves[0][0], new_solution.moves[1][0],
                                                               new_solution.moves[2][0], new_solution.moves[3][0]])
        # Get all the scores for the turn
        for q in range(len(score_results)):
            new_solution.scores[q].append(score_results[q])

    return new_solution


def determine_trajectory(pods, checkpoints, moves):
    for j in range(len(pods)):
        pods[j].rotate(moves[j].point_two)
        pods[j].boost(moves[j].thrust)

    return play_turn(pods, checkpoints)


def play_turn(pod_list, checkpoints):
    bug_found = False
    fitness_results = []
    t: float = 0.0
    while t < 1.0:
        first_collision = None
        for j in range(len(pod_list)):
            for k in range(j + 1, len(pod_list)):
                collision = pod_list[j].collision(pod_list[k])
                if (collision is not None) and (collision.time + t < 1.0) and (
                        (first_collision is None) or (collision.time < first_collision.time)):
                    first_collision = collision
                    if bug_found:
                        if first_collision.time <= 0:
                            first_collision = None
            collision = pod_list[j].collision(checkpoints[pod_list[j].next_target_id])
            if (collision is not None) and (collision.time + t < 1.0) and (
                    (first_collision is None) or (collision.time < first_collision.time)):
                first_collision = collision
                if bug_found:
                    if first_collision.time <= 0:
                        first_collision = None
        if first_collision is None:
            for j in range(len(pod_list)):
                pod_list[j].move(1.0 - t)
            t = 1.0
        else:
            for j in range(len(pod_list)):
                pod_list[j].move(first_collision.time)
            first_collision.unit_a.bounce(first_collision.unit_b)
            bug_found = True
            t += first_collision.time
    for j in range(len(pod_list)):
        fitness_results.append(pod_list[j].end(checkpoints))
    return fitness_results


def anneal(old_solution, pods, checkpoints):
    current_solution = old_solution
    current_cost = cost(current_solution)
    temperature = 1.0
    minimum_temperature = 0.01
    cooling_rate = 0.9
    while temperature > minimum_temperature:
        for i in range(number_of_solutions):
            new_solution = neighbor(current_solution, pods, checkpoints)
            new_cost = cost(new_solution)
            acceptance = acceptance_probability(current_cost, new_cost, temperature)
            if acceptance > random():
                current_solution = new_solution
                current_cost = new_cost
            temperature = temperature * cooling_rate
    return solution, cost


first_turn: bool = True
targets = []

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

laps: int = int(input())

checkpoint_count: int = int(input())

for i in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    # Add the checkpoints to the targets list
    targets.append(Checkpoint(checkpoint_x, checkpoint_y, i, 600, checkpoint_count))

# game loop
while True:
    input_list = []
    pods = []
    pod_clones = []
    enemies = []
    enemy_clones = []
    for i in range(2):
        # x: x position of your pod
        # y: y position of your pod
        # vx: x speed of your pod
        # vy: y speed of your pod
        # angle: angle of your pod
        # next_check_point_id: next check point id of your pod
        x, y, vx, vy, angle, next_check_point_id = [int(j) for j in input().split()]
        if i == 0:
            name = "Erik"
        else:
            name = "Greg"
        pod = Pod(x, y, vx, vy, angle, next_check_point_id, 400, checkpoint_count, laps, name)

        pods.append(pod)

    for i in range(2):
        # x_2: x position of the opponent's pod
        # y_2: y position of the opponent's pod
        # vx_2: x speed of the opponent's pod
        # vy_2: y speed of the opponent's pod
        # angle_2: angle of the opponent's pod
        # next_check_point_id_2: next check point id of the opponent's pod
        x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2 = [int(j) for j in input().split()]
        if i == 0:
            name = "Darth"
        else:
            name = "Boss"
        pod = Pod(x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2, 400, checkpoint_count, laps, name)
        pods.append(pod)

    pod_clones = pods

    number_of_solutions = 25
    number_of_turns = 5
    # for i in range(number_of_solutions):
    # Create initial solution to begin annealing process
    solution = Solution()

    # Create moves using autopilot
    for j in range(number_of_turns):
        for p in range(len(pod_clones)):
            # Get target id
            if pod_clones[p].next_target_id + 1 == pod_clones[p].num_targets:
                next_target = 0
            else:
                next_target = pod_clones[p].next_target_id + 1
            # Create a turn then save it to the solution
            # create_turn(pod, target, next_target, checkpoints, style=0)
            solution.moves[p].append(create_turn(pod_clones[p], targets[pod_clones[p].next_target_id],
                                                 targets[next_target], targets, 0))
        # Determine where all the pods end up for projected turn and their scores/fitness
        fitness_result = determine_trajectory(pod_clones, targets, [solution.moves[0][0], solution.moves[1][0],
                                                                    solution.moves[2][0], solution.moves[3][0]])
        # Get all the scores for the turn
        for q in range(len(fitness_result)):
            solution.scores[q].append(fitness_result[q])
    print((str(int(solution.moves_one[0].point_two.x)) + " " + str(int(solution.moves_one[0].point_two.y)) + " " +
          str(int(solution.moves_one[0].thrust))), file=sys.stderr)
    print((str(int(solution.moves_two[0].point_two.x)) + " " + str(int(solution.moves_two[0].point_two.y)) + " " +
          str(int(solution.moves_two[0].thrust))), file=sys.stderr)
    # One solution has now been made, time to anneal it
    final_solution, final_cost = anneal(solution, pods, targets)
    if final_solution.moves_one[0].thrust > 100:
        final_solution.moves_one[0].thrust = 100
    if final_solution.moves_two[0].thrust > 100:
        final_solution.moves_two[0].thrust = 100
    print(str(int(final_solution.moves_one[0].point_two.x)) + " " + str(int(final_solution.moves_one[0].point_two.y)) + " " +
          str(int(final_solution.moves_one[0].thrust)))
    print(str(int(final_solution.moves_two[0].point_two.x)) + " " + str(int(final_solution.moves_two[0].point_two.y)) + " " +
          str(int(final_solution.moves_two[0].thrust)))
