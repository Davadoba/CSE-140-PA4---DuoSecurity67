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

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """
    agent1_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.MyAgent1")
    agent2_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.MyAgent2")
    return [agent1_info, agent2_info]

class MyAgent1(pacai.capture.agents.OffensiveAgent):
    """ Offensive Agent """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.maze_cache = {}

    def cached_maze_distance(self, pos1, pos2, state):
        """ Cache Expensive Maze_Distance Calculations"""
        # This key stays consistent no matter the position ordering
        key1 = (pos1.row, pos1.col)
        key2 = (pos2.row, pos2.col)
        if key1 <= key2:
            key = (key1, key2)
        else:
            key = (key2, key1)
        
        if key in self.maze_cache:
            return self.maze_cache[key]

        dist = pacai.search.distance.maze_distance(pos1, pos2, state)
        self.maze_cache[key] = dist
        return dist

    def get_action(self, state: pacai.capture.gamestate.GameState) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        max_score = float('-inf')
        best_action = None
        for action in legal_actions:
            successor = state.generate_successor(action)
            score = self.evaluate_state(successor, action)
            if action == pacai.core.action.STOP:
                score -= 100
            if score > max_score:
                max_score = score
                best_action = action
        return best_action

    def evaluate_state(self,
        state: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action | None = None,
        **kwargs
    ) -> float:
        agent_positions = state.get_agent_positions()
        this_agent_pos = agent_positions[self.agent_index]
        if this_agent_pos is None:
            return float('-inf')

        # Setup
        enemy_food_positions = state.get_food(agent_index=self.agent_index)
        num_food = state.food_count(agent_index=self.agent_index)
        if num_food == 0:
            return float('inf')

        evaluation = 0

        # Light Recency Penalty To Prevent Back-Forth Oscillation
        if len(self.last_positions) >= 2:
            if this_agent_pos == self.last_positions[-2]:
                evaluation -= 15

        # Chase Enemy Food & Incentivize Eating
        # Find The 5 Closest Food By Euclidean, Then Find
        # The Closest One By Maze_Distance From Those 5
        evaluation -= num_food * 90
        closest_foods = []
        if not num_food <= 0:
            closest_foods = heapq.nsmallest(5, enemy_food_positions,
                key=lambda pos: pacai.search.distance.euclidean_distance(pos, this_agent_pos, state)
            )
        min_food_distance = min(
            self.cached_maze_distance(f, this_agent_pos, state)
            for f in closest_foods
        )
        evaluation -= 8 * math.sqrt(min_food_distance)

        # Incentivize being a pacman
        if state.is_ghost(self.agent_index):
            evaluation -= 50

        # Avoid Ghosts While Invading
        if state.is_pacman(self.agent_index):
            for (_, pos) in state.get_nonscared_opponent_positions(agent_index=self.agent_index).items():
                dist = self.cached_maze_distance(pos, this_agent_pos, state)
                # Continuous Penalty That Increases More Sharply The Closer A Ghost Is
                evaluation -= 40 / (dist + 1)

        return evaluation

class MyAgent2(pacai.capture.agents.DefensiveAgent):
    """ Defensive Agent """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.maze_cache = {}

    def cached_maze_distance(self, pos1, pos2, state):
        """ Cache Expensive Maze_Distance Calculations"""
        # This key stays consistent no matter the position ordering
        key = (
            min(pos1.row, pos2.row),
            min(pos1.col, pos2.col),
            max(pos1.row, pos2.row),
            max(pos1.col, pos2.col)
        )
        if key in self.maze_cache:
            return self.maze_cache[key]

        dist = pacai.search.distance.maze_distance(pos1, pos2, state)
        self.maze_cache[key] = dist
        return dist

    def get_action(self, state: pacai.capture.gamestate.GameState) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        max_score = float('-inf')
        best_action = None
        for action in legal_actions:
            successor = state.generate_successor(action)
            score = self.evaluate_state(successor, action)
            if score > max_score:
                max_score = score
                best_action = action
        return best_action

    def evaluate_state(self,
        state: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action | None = None,
        **kwargs
    ) -> float:
        agent_positions = state.get_agent_positions()
        this_agent_pos = agent_positions[self.agent_index]
        if this_agent_pos is None:
            return float('-inf')

        # Invader Info
        invader_dict = state.get_invader_positions(self.agent_index)
        invader_positions = set(invader_dict.values())

        # Non-Invading Opponent Info
        opp_dict = state.get_opponent_positions(self.agent_index)
        opp_ghost_positions = set(opp_dict.values()) - invader_positions

        evaluation = 0

        # Edge of Border X Position
        border_col = state.board.width // 2
        if self.agent_index % 2 == 1:
            border_col += 1

        # Prioritize Chasing Invaders.
        # If No Invaders, Approach Enemy Ghosts But Don't Cross Border.
        evaluation -= len(invader_positions) * 200
        if len(invader_positions) > 0:
            min_invader_distance = min(
                self.cached_maze_distance(i, this_agent_pos, state)
                for i in invader_positions
            )
            evaluation -= min_invader_distance * 6
        else:
            # What Are The Legal Border Positions?
            legal_border_positions = []
            for row in range(state.board.height):
                p = Position(row, border_col)
                if not state.board.is_wall(p):
                    legal_border_positions.append(p)

            # If The Enemies Aren't Dead & We Can Hover On The Border
            if len(opp_ghost_positions) > 0 and len(legal_border_positions) > 0:
                best_intercept_score = float('-inf')
                for g in opp_ghost_positions:
                    # Improve Performance By Only Considering A Few
                    # Of The Most Likely Enemy Crossing Points
                    candidate_tiles = sorted(
                        legal_border_positions,
                        key=lambda b: abs(b.row - g.row)
                    )[:3]

                    for b in candidate_tiles:
                        my_dist = self.cached_maze_distance(
                            this_agent_pos,
                            b,
                            state
                        )
                        enemy_dist = self.cached_maze_distance(
                            g,
                            b,
                            state
                        )
                        # Prefer Tiles We Can Reach Before The Enemy
                        intercept_score = my_dist - enemy_dist
                        best_intercept_score = max(best_intercept_score, intercept_score)
                evaluation -= best_intercept_score * 5

        # For Now, This Agent Should Never Be A Pacman
        if state.is_pacman(self.agent_index):
            evaluation -= 1000

        return evaluation
