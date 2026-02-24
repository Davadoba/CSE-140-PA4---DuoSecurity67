import typing
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.gamestate
import pacai.capture.gamestate
import pacai.capture.agents
import pacai.search.distance
import pacai.capture.board

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

        enemy_food_positions = state.get_food(agent_index=self.agent_index)
        num_food = state.food_count(agent_index=self.agent_index)
        if num_food == 0:
            return float('inf')

        evaluation = 0

        # Chase Enemy Food & Incentivize Eating
        evaluation -= num_food * 100
        min_food_distance = min(
            pacai.search.distance.maze_distance(f, this_agent_pos, state)
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

        enemy_food_positions = state.get_food(agent_index=self.agent_index)
        num_food = state.food_count(agent_index=self.agent_index)
        if num_food == 0:
            return float('inf')

        evaluation = 0

        # Chase Enemy Food & Incentivize Eating
        evaluation -= num_food * 100
        min_food_distance = min(
            pacai.search.distance.maze_distance(f, this_agent_pos, state)
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
