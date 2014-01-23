from collections import defaultdict
from functools import partial

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        dist = util.manhattanDistance
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        try:
            closestFood  = list(sorted([(dist(newPos, f), f) for f in newFood.asList()]))[0][1]
        except IndexError:
            closestFood = None
        mult = 1
        if closestFood is not None:
            if dist(oldPos, closestFood) > dist(newPos, closestFood):
                mult = 5
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        _dist = [dist(newPos, g.getPosition()) for g in newGhostStates]
        for d, t in zip(_dist, newScaredTimes):
            if d <= 2 and t == 0:
                mult = 0.1
                break
        if successorGameState.getScore() < 0:
            mult *= -1
        return successorGameState.getScore() * mult

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class Node(object):

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minimax_decision(gameState, self.depth, 0)[1]

    def minimax_decision(self, state, depth, agent):
        if agent == state.getNumAgents():
            agent = 0
            depth -= 1
        actions = state.getLegalActions(agent)
        win_action = None
        if depth == 0 or not actions:
            return self.evaluationFunction(state), None
        if agent == 0:
            v = -99999999999999
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v_new = max(v, self.minimax_decision(successor, depth, agent + 1)[0])
                if v_new > v:
                    v = v_new
                    win_action = action
            return v, win_action
        else:
            v = 9999999999999
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v_new = min(v, self.minimax_decision(successor, depth, agent + 1)[0])
                if v_new < v:
                    v = v_new
                    win_action = action
            return v, win_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alpha_beta_decision(gameState, self.depth, -999999999, 999999999, 0)[1]

    def alpha_beta_decision(self, state, depth, alpha, beta, agent):
        if agent == state.getNumAgents():
            agent = 0
            depth -= 1
        actions = state.getLegalActions(agent)
        win_action = None
        if depth == 0 or not actions:
            return self.evaluationFunction(state), None
        if agent == 0:
            v = -99999999999999
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v_new = max(v, self.alpha_beta_decision(successor, depth, alpha, beta, agent + 1)[0])
                if v_new > v:
                    v = v_new
                    win_action = action
                if v > beta:
                    return v, win_action
                alpha = max(alpha, v)
            return v, win_action
        else:
            v = 9999999999999
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v_new = min(v, self.alpha_beta_decision(successor, depth, alpha, beta, agent + 1)[0])
                if v_new < v:
                    v = v_new
                    win_action = action
                if v < alpha:
                    return v, win_action
                beta = min(beta, v)
            return v, win_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax_decision(gameState, self.depth, 0)[1]

    def expectimax_decision(self, state, depth, agent):
        if agent == state.getNumAgents():
            agent = 0
            depth -= 1
        actions = state.getLegalActions(agent)
        win_action = None
        if depth == 0 or not actions:
            return self.evaluationFunction(state), None
        if agent == 0:
            v = -99999999999999
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v_new = max(v, self.expectimax_decision(successor, depth, agent + 1)[0])
                if v_new > v:
                    v = v_new
                    win_action = action
            return v, win_action
        else:
            v = 0
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                v += self.expectimax_decision(successor, depth, agent + 1)[0]/float(len(actions))
            return v, None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    _dist = util.manhattanDistance
    score = currentGameState.getScore()
    pac_pos = currentGameState.getPacmanPosition()
    dist = lambda x: [_dist(pac_pos, i) for i in x]
    ghost_pos = currentGameState.getGhostPositions()
    closest_ghost = min(dist(ghost_pos))
    if closest_ghost == 0:
        closest_ghost = 1
    return score + 1/float(closest_ghost)

# Abbreviation
better = betterEvaluationFunction
