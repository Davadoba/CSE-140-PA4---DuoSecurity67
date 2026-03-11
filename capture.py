import math
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.gamestate
import pacai.capture.gamestate
import pacai.capture.agents
import pacai.search.distance
import pacai.capture.board
import pacai.core.board
from pacai.core.board import Position
import pacai.agents.greedy
from pacai.search.distance import DistancePreComputer
from pacai.pacman.board import MARKER_CAPSULE

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """
    agent1_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.MyAgent1")
    agent2_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.MyAgent2")
    return [agent1_info, agent2_info]

class BaseCaptureAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """ Shared Logic Between Agents """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.maze_cache = {}
        self.kill_switch = 0
        self.repeated_positions_counter = 0
        self.prev_food_count = None
        self.prev_position = None
        self.start_offense = False
        self.opportunistic_offense = False
        self.distance_precomputer = DistancePreComputer()
        self.distances_ready = False

    # def cached_maze_distance(self, pos1, pos2):
    #     """ Cache Expensive Maze Distance Calculations """
        # key1 = (pos1.row, pos1.col)
        # key2 = (pos2.row, pos2.col)
        # key = tuple(sorted([key1, key2]))

        # if key in self.maze_cache:
        #     return self.maze_cache[key]

        # dist = pacai.search.distance.maze_distance(pos1, pos2, state)
        # self.maze_cache[key] = dist
        # return dist

    def dist(self, a, b):
        """ Use Distance Precomputer To Get Distance """
        return self.distance_precomputer.get_distance_default(a, b, 999)
    
    def is_stuck(self, state):
        """ Detect If Offensive Agent Is Stuck """
        window = 12
        last_actions = state.get_agent_actions(self.agent_index)
        
        if len(last_actions) < window:
            return False

        recent = last_actions[-window:]

        # Ignore STOP actions
        recent = [a for a in recent if a != pacai.core.action.STOP]

        if len(recent) < 6:
            return False

        # Check if alternating between two actions
        a = recent[0]
        b = recent[1]

        if all(recent[i] == (a if i % 2 == 0 else b) for i in range(len(recent))):
            return True

        return False
    
    def get_team_modifier(self):
        """ Get This Agent's Team Modifier """
        return -1 if self.agent_index % 2 == 0 else 1
    
    def get_border_positions(self, state):
        """ Get Border Positions Of This Agent's Team """
        mid = state.board.width // 2
        modifier = self.get_team_modifier()

        if modifier == -1:
            border_col = mid - 1   # west side border
        else:
            border_col = mid       # east side border

        border_positions = []

        for row in range(state.board.height):
            p = Position(row, border_col)
            if not state.board.is_wall(p):
                border_positions.append(p)

        return border_positions
    
    def score_border_tile(self, tile, state, my_pos):
        """ Used By Offensive Agent To Evaluate Border Crossing Points"""
        score = 0
        # Prefer tiles closer to us
        dist_to_me = self.dist(my_pos, tile)
        score -= dist_to_me * 2

        # Avoid defenders near the border tile
        for (_, enemy_pos) in state.get_nonscared_opponent_positions(
            agent_index=self.agent_index
        ).items():
            enemy_dist = self.dist(enemy_pos, tile)
            if enemy_dist < 6:
                score -= (6 - enemy_dist) * 20

        # Encourage Going To Tiles With Food Behind Them
        food = state.get_food(agent_index=self.agent_index)
        if len(food) > 0:
            closest_food = min(
                food,
                key=lambda f: self.dist(tile, f)
            )
            food_dist = self.dist(tile, closest_food)
            score -= food_dist

        return score
    
    def get_best_border_tile(self, state, my_pos):
        """ Helper For Offensive Agent To Get Best Crossing Point"""
        border_positions = self.get_border_positions(state)
        best_tile = None
        best_score = float('-inf')
        for tile in border_positions:
            score = self.score_border_tile(tile, state, my_pos)

            if score > best_score:
                best_score = score
                best_tile = tile

        return best_tile
    
    def distance_to_border(self, state, pos):
        """ Distance To Border(Used For Role Switching Logic)"""
        possible_positions = self.get_border_positions(state)
        return min(self.dist(pos, b)
                   for b in possible_positions
                )
    
    def get_power_pellets(self, state):
        """Return list of positions with power capsules on the enemy side."""
        pellets = []
        mid = state.board.width // 2
        modifier = self.get_team_modifier()  # -1 = left team, 1 = right team

        # Define enemy side bounds
        if modifier == -1:  # we are left team → enemy is right half
            enemy_cols = range(mid, state.board.width)
        else:  # we are right team → enemy is left half
            enemy_cols = range(0, mid)

        # Only scan if the marker exists in the board
        if MARKER_CAPSULE not in state.board._nonwall_objects:
            return pellets  # empty list, no pellets left

        for row in range(state.board.height):
            for col in enemy_cols:
                pos = Position(row, col)
                if state.board.is_marker(MARKER_CAPSULE, pos):
                    pellets.append(pos)
        return pellets
    
    def evaluate_offense(self, state):
        """ Offensive Agent Logic """
        agent_positions = state.get_agent_positions()
        this_agent_pos = agent_positions[self.agent_index]
        if this_agent_pos is None:
            self.kill_switch = False
            self.repeated_positions_counter = 0
            return float('-inf')

        enemy_food_positions = state.get_food(agent_index=self.agent_index)
        num_food = state.food_count(agent_index=self.agent_index)

        if num_food == 0:
            return float('inf')

        evaluation = 0
        
        # Intentionally Die If We've Been Trapped ------------------
        if self.is_stuck(state) and state.is_pacman(self.agent_index):
            self.repeated_positions_counter += 1
        else:
            self.repeated_positions_counter = 0
        
        if self.repeated_positions_counter > 6:
            self.kill_switch = True
        
        if self.kill_switch:
            closest_opp_distance = float('inf')
            for (_, pos) in state.get_nonscared_opponent_positions(
                agent_index=self.agent_index
            ).items():
                dist = self.dist(pos, this_agent_pos)
                if dist < closest_opp_distance:
                    closest_opp_distance = dist

            evaluation -= closest_opp_distance * 20
            return evaluation

        # Normal offensive Logic --------------

        # Food priority
        evaluation -= num_food * 90
        if self.prev_food_count is not None and num_food < self.prev_food_count:
            evaluation += 300
        
        min_food_distance = min(
            self.dist(f, this_agent_pos)
            for f in enemy_food_positions
        )
        evaluation -= min_food_distance * 8

        if state.is_ghost(self.agent_index):
            best_border = self.get_best_border_tile(state, this_agent_pos)

            if best_border is not None:
                dist_to_border = self.dist(this_agent_pos, best_border)
                evaluation -= dist_to_border * 15

        # Power Pellet Priority
        power_pellets = self.get_power_pellets(state)
        if power_pellets:
            closest_pellet = min(power_pellets, key=lambda p: self.dist(p, this_agent_pos))
            evaluation -= self.dist(closest_pellet, this_agent_pos) * 3
            scared_opps = state.get_scared_opponent_positions(agent_index=self.agent_index)
            if len(scared_opps) > 0:
                evaluation += 300
        
        # Avoid ghosts while invading
        if state.is_pacman(self.agent_index):
            for (_, pos) in state.get_nonscared_opponent_positions(
                agent_index=self.agent_index
            ).items():
                dist = self.dist(pos, this_agent_pos)
                if dist <= 1:
                    evaluation -= 500
                elif dist <= 3:
                    evaluation -= 200
                elif dist <= 5:
                    evaluation -= 80
        
        return evaluation

    def evaluate_defense(self, state):
        """ Defensive Agent Logic """
        agent_positions = state.get_agent_positions()
        this_agent_pos = agent_positions[self.agent_index]
        if this_agent_pos is None:
            return float('-inf')

        invader_dict = state.get_invader_positions(self.agent_index)
        invader_positions = set(invader_dict.values())

        opp_dict = state.get_opponent_positions(self.agent_index)
        opp_ghost_positions = set(opp_dict.values()) - invader_positions

        opp_indices = list(state.get_opponent_positions().keys())
        if opp_indices:
            def_food = state.get_food(opp_indices[0])
        else:
            def_food = None
        evaluation = 0

        border_col = state.board.width // 2
        if self.agent_index % 2 == 1:
            border_col += 1

        border_positions = self.get_border_positions(state)
        border_dist = min(self.dist(this_agent_pos, b) for b in border_positions)
        evaluation -= border_dist * 10
        
        # Prioritize chasing invaders
        evaluation -= len(invader_positions) * 200

        if len(invader_positions) > 0:
            best_intercept_score = 0

            for invader in invader_positions:
                invader_dist_to_me = self.dist(this_agent_pos, invader)

                # --- Direct Pursuit Term ---
                # Stronger if invader is close, weaker if far
                pursuit_weight = max(80, 150 - 5 * invader_dist_to_me)
                evaluation -= pursuit_weight * invader_dist_to_me

                if def_food:
                    # Assume invader targets closest food
                    target_food = min(
                        def_food,
                        key=lambda f, inv=invader: self.dist(inv, f)
                    )

                    invader_to_food = self.dist(invader, target_food)
                    my_to_food = self.dist(this_agent_pos, target_food)

                    # Intercept if we can reach before or roughly the same time
                    intercept_score = invader_to_food - my_to_food
                    intercept_score = min(intercept_score * 25, 50)  # cap to avoid runaway
                    if my_to_food <= invader_to_food + 1:  # small slack
                        best_intercept_score = max(best_intercept_score, intercept_score)
            evaluation += best_intercept_score
        else:
            # Border hovering logic
            legal_border_positions = []
            for row in range(state.board.height):
                p = Position(row, border_col)
                if not state.board.is_wall(p):
                    legal_border_positions.append(p)

            if len(opp_ghost_positions) > 0 and len(legal_border_positions) > 0:
                total_patrol_score = 0

                for g in opp_ghost_positions:
                    candidate_tiles = sorted(
                        legal_border_positions,
                        key=lambda b: abs(b.row - g.row)
                    )[:3]
                    
                    ghost_score = 0
                    weight_sum = 0
                    
                    for b in candidate_tiles:
                        enemy_dist = self.dist(g, b)
                        weight = math.exp(-0.5 * enemy_dist)
                        weight = min(weight, 0.3)  # Each Ghost Has A Max-Influence
                        
                        my_dist = self.dist(this_agent_pos, b)
                        ghost_score += weight * (-my_dist)
                        weight_sum += weight
                    if weight_sum > 0:
                        ghost_score /= weight_sum  # This Should Never Happen?
                    total_patrol_score += ghost_score
                evaluation += total_patrol_score * 5

        # Defensive agent should not cross border
        if state.is_pacman(self.agent_index):
            evaluation -= 1000

        return evaluation

class MyAgent1(BaseCaptureAgent):
    """ Permanent Defense For Now """
    
    def get_action(self, state):
        
        # Distance Precomputation
        if not self.distances_ready:
            self.distance_precomputer.compute(state.board)
            self.distances_ready = True
        
        legal_actions = state.get_legal_actions()
        # legal_actions = [a for a in state.get_legal_actions() if a != pacai.core.action.STOP]
        max_score = float('-inf')
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            score = 0

            # Decide Offense vs Defense
            score = self.evaluate_defense(successor)
            
            if score > max_score:
                max_score = score
                best_action = action
        
        # Update Class Variables
        self.prev_food_count = state.food_count(agent_index=self.agent_index)
        self.prev_position = state.get_agent_positions()[self.agent_index]
        
        return best_action

class MyAgent2(BaseCaptureAgent):
    """ Defensive, But Goes On Offense If Close To Border """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opportunistic_offense = False

    def is_offensive_role(self, state):
        """ When To Go On Offense """
        agent_positions = state.get_agent_positions()
        my_pos = agent_positions[self.agent_index]

        # If we died → reset behavior
        if my_pos is None:
            self.opportunistic_offense = False
            self.repeated_positions_counter = 0
            self.kill_switch = 0
            return False

        # Once offensive, remain offensive
        if self.opportunistic_offense:
            return True

        # Switch when near border
        border_dist = self.distance_to_border(state, my_pos)
        if border_dist <= 5:
            self.opportunistic_offense = True
            return True

        return False
    
    def get_action(self, state):
        
        # Distance Precomputation
        if not self.distances_ready:
            self.distance_precomputer.compute(state.board)
            self.distances_ready = True
        
        legal_actions = state.get_legal_actions()
        # legal_actions = [a for a in state.get_legal_actions() if a != pacai.core.action.STOP]
        max_score = float('-inf')
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            score = 0

            # Decide Offense vs Defense
            if self.is_offensive_role(successor):
                score = self.evaluate_offense(successor)
            else:
                score = self.evaluate_defense(successor)

            if score > max_score:
                max_score = score
                best_action = action

        # Update Class Variables
        self.prev_food_count = state.food_count(agent_index=self.agent_index)
        self.prev_position = state.get_agent_positions()[self.agent_index]

        return best_action
