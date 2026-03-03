import typing
import heapq
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

    def cached_maze_distance(self, pos1, pos2, state):
        """ Cache Expensive Maze Distance Calculations """
        key1 = (pos1.row, pos1.col)
        key2 = (pos2.row, pos2.col)
        key = tuple(sorted([key1, key2]))

        if key in self.maze_cache:
            return self.maze_cache[key]

        dist = pacai.search.distance.maze_distance(pos1, pos2, state)
        self.maze_cache[key] = dist
        return dist

    def evaluate_offense(self, state):
        """ Offensive Agent Logic """
        agent_positions = state.get_agent_positions()
        this_agent_pos = agent_positions[self.agent_index]
        if this_agent_pos is None:
            return float('-inf')

        enemy_food_positions = state.get_food(agent_index=self.agent_index)
        num_food = state.food_count(agent_index=self.agent_index)

        if num_food == 0:
            return float('inf')

        evaluation = 0

        # Oscillation prevention
        if len(self.last_positions) >= 2:
            if this_agent_pos == self.last_positions[-2]:
                evaluation -= 15

        # Food priority
        evaluation -= num_food * 90
        closest_foods = heapq.nsmallest(
            5,
            enemy_food_positions,
            key=lambda pos: pacai.search.distance.euclidean_distance(
                pos, this_agent_pos, state
            )
        )
        if closest_foods:
            min_food_distance = min(
                self.cached_maze_distance(f, this_agent_pos, state)
                for f in closest_foods
            )
            evaluation -= 8 * math.sqrt(min_food_distance)

        # Encourage being Pacman
        if state.is_ghost(self.agent_index):
            evaluation -= 50

        # Avoid ghosts while invading
        if state.is_pacman(self.agent_index):
            for (_, pos) in state.get_nonscared_opponent_positions(
                agent_index=self.agent_index
            ).items():
                dist = self.cached_maze_distance(pos, this_agent_pos, state)
                evaluation -= 40 / (dist + 1)

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

        evaluation = 0

        border_col = state.board.width // 2
        if self.agent_index % 2 == 1:
            border_col += 1

        # Prioritize chasing invaders
        evaluation -= len(invader_positions) * 200

        if len(invader_positions) > 0:
            min_invader_distance = min(
                self.cached_maze_distance(i, this_agent_pos, state)
                for i in invader_positions
            )
            evaluation -= min_invader_distance * 6
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
                        enemy_dist = self.cached_maze_distance(
                            g, b, state
                        )
                        weight = math.exp(-0.5 * enemy_dist)
                        
                        my_dist = self.cached_maze_distance(
                            this_agent_pos, b, state
                        )
                        ghost_score += weight * (-my_dist)
                        weight_sum += weight
                    if weight_sum > 0:
                        ghost_score /= weight_sum # This Should Never Happen?
                    total_patrol_score += ghost_score
                evaluation += total_patrol_score * 5

        # Defensive agent should not cross border
        if state.is_pacman(self.agent_index):
            evaluation -= 1000

        return evaluation

class MyAgent1(BaseCaptureAgent):
    """ Agent 1(Red?) - Offensive Bias """

    def get_action(self, state):
        legal_actions = state.get_legal_actions()
        max_score = float('-inf')
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            score = 0

            # Decide Offense vs Defense
            score = self.evaluate_offense(successor)
            
            if score > max_score:
                max_score = score
                best_action = action

        return best_action

class MyAgent2(BaseCaptureAgent):
    """ Second Agent (Orange?) - Defensive Bias """

    def get_action(self, state):
        legal_actions = state.get_legal_actions()
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

        return best_action
