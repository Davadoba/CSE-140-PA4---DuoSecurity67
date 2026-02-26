import typing
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

# https://chatgpt.com/share/699fea4f-5070-8008-a494-1e86e946df76
# This chatGPT convo has some suggestions for improvements(most recent message)

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

        # Penalize for being in recent positions
        recency_penalty = 0
        history = self.last_positions
        lookback = 10
        for i, past_pos in enumerate(reversed(history[-lookback:])):
            if past_pos is None:
                continue
            if past_pos == this_agent_pos:
                recency_penalty += (lookback - i) * 5
        evaluation -= recency_penalty

        # Chase Enemy Food & Incentivize Eating
        # Use euclidean distance for speed
        evaluation -= num_food * 100
        min_food_distance = min(
            pacai.search.distance.euclidean_distance(f, this_agent_pos, state)
            for f in enemy_food_positions
        )
        evaluation -= min_food_distance * 3

        # Incentivize being a pacman
        if state.is_ghost(self.agent_index):
            evaluation -= 20

        # Avoid Ghosts While Invading
        if state.is_pacman(self.agent_index):
            for (_, pos) in state.get_nonscared_opponent_positions(agent_index=self.agent_index).items():
                dist = pacai.search.distance.maze_distance(pos, this_agent_pos, state)
                if dist < 3:
                    evaluation -= (4 - dist) * 20

        return evaluation

class MyAgent2(pacai.capture.agents.DefensiveAgent):
    """ Defensive Agent """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

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
        evaluation -= len(invader_positions) * 100
        if len(invader_positions) > 0:
            min_invader_distance = min(
                pacai.search.distance.maze_distance(i, this_agent_pos, state)
                for i in invader_positions
            )
            evaluation -= min_invader_distance * 3
        else:
            legal_border_positions = []
            for row in range(state.board.height):
                p = Position(row, border_col)
                if not state.board.is_wall(p):
                    legal_border_positions.append(p)

            if len(opp_ghost_positions) > 0 and len(legal_border_positions) > 0:
                min_opp_distance = float('inf')
                for g in opp_ghost_positions:
                    closest_border_tile = min(
                        legal_border_positions,
                        key=lambda b, g=g: abs(b.row - g.row)
                    )

                    dist = pacai.search.distance.maze_distance(
                        closest_border_tile,
                        this_agent_pos,
                        state
                    )
                    min_opp_distance = min(min_opp_distance, dist)
                evaluation -= min_opp_distance * 2

        # For Now, This Agent Should Never Be A Pacman
        if state.is_pacman(self.agent_index):
            evaluation -= 1000

        return evaluation
