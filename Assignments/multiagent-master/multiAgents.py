# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    # Dhriti's Part
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Dhriti
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #distance = float("inf") 
        #if successorGameState.hasFood(newPos[0],newPos[1]):
            #distance = util.manhattanDistance(currentGameState.getPacmanPosition(),successorGameState.getPacmanPosition())

        #else:
        newFood = list(successorGameState.getFood())
        distance = float("inf")
        for food in newFood:
            for i in food:
                if i == True:
                    distance = min(distance, manhattanDistance(newPos, (newFood.index(food),food.index(i))))
        AgentIndexCounter = 1

        for i in successorGameState.getGhostStates():
                if manhattanDistance(successorGameState.getPacmanPosition(),successorGameState.getGhostPosition(AgentIndexCounter)) < 2:
                    return -successorGameState.getScore()
                AgentIndexCounter+=1

        return successorGameState.getScore() + 1/distance

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

# Brittney's Part
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Note-to-self: Can use underscore '_' to ignore specific values

        # Call minimax to determine the best action for the current game state
        # * Not interested in the second element of the returned tuple *
        bestAction, _ = self.minimax(gameState, 0, 0)
        return bestAction

    #  Evaluates the current game state using the Minimax algorithm
    def minimax(self, gameState, agentIndex, depth):
        # Check if the game is in a terminal state or the maximum depth is reached
        if gameState.isWin() or gameState.isLose() or depth >= self.depth * gameState.getNumAgents():
            return 'Stop', self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.chooseBestAction(gameState, 0, depth, comparisonFunction=max)
        
        else:
            return self.chooseBestAction(gameState, agentIndex, depth, comparisonFunction=min)

    # Chooses the best action for the current player using the specified comparison function 
    def chooseBestAction(self, gameState, agentIndex, depth, comparisonFunction):
        nextLegalActions = gameState.getLegalActions(agentIndex)
        nextStates = []  # Initialize an empty list to store the next states

        # Iterate over each action in the list
        for action in nextLegalActions:
            # Create the successor state for the current action and agent index
            successorState = gameState.generateSuccessor(agentIndex, action)

            nextStates.append(successorState)

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        storeValues = []  # Initialize an empty list to store the values

        # Iterate over each state in the list
        for state in nextStates:
            # Call self.minimax to get the value for the current next_state
            # Assume that the minimax function returns a tuple where the second element is the value we want
            value = self.minimax(state, nextAgentIndex, depth + 1)[1]

            storeValues.append(value)

        # Choose the best value according to the specified best function
        bestValue = comparisonFunction(storeValues)

        # Determine the corresponding action for the best value
        bestAction = nextLegalActions[storeValues.index(bestValue)]

        return bestAction, bestValue  

# Dhriti's Part
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestAction, _ = self.minimax(gameState, 0, 0, -float("inf"), float("inf"))
        return bestAction

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        # Check if the game is in a terminal state or the depth limit is reached
        if gameState.isWin() or gameState.isLose() or depth >= self.depth * gameState.getNumAgents():
            return 'Stop', self.evaluationFunction(gameState)

        if agentIndex == 0:  # If it's Pacman's turn (maximizing player)
            # Call the chooseBestAction function to select the best action for Pacman
            functioncall = self.chooseBestAction(gameState, 0, depth, alpha, beta, comparisonFunction=max)
           # Update alpha value
            return functioncall
        else:  # If it's a ghost's turn (minimizing player)
            # Call the chooseBestAction function to select the best action for the ghost
            functioncall = self.chooseBestAction(gameState, agentIndex, depth, alpha, beta, comparisonFunction=min)
              # Update beta value
            return functioncall
    def chooseBestAction(self, gameState, agentIndex, depth, alpha, beta, comparisonFunction):
        # Get the legal actions for the current player
        nextLegalActions = gameState.getLegalActions(agentIndex)
        # Iterate over each action in the list
        storeValues = []
        for action in nextLegalActions:
            # Create the successor state for the current action and agent index
            if beta < alpha:
                break
            successorState = gameState.generateSuccessor(agentIndex, action)
            #print(gameState.getNumAgents())
            value = self.minimax(successorState, (depth + 1)%gameState.getNumAgents(),depth + 1, alpha, beta)[1]
            storeValues.append(value)
            if comparisonFunction == min:
                beta = min(beta,value)
                if value < alpha:
                    break
            if comparisonFunction == max:
                alpha = max(alpha,value)
                if value > beta:
                    break

        # Initialize an empty list to store the values

        # Iterate over each state in the list
        # Call self.minimax to get the value for the current next_state
        # If beta is less than or equal to alpha, prune the rest of the states


        bestValue = comparisonFunction(storeValues)
        # Determine the corresponding action for the best value
        bestAction = nextLegalActions[storeValues.index(bestValue)]

        return bestAction, bestValue  # Return the best action and its value 

        util.raiseNotDefined()

# Brittney's Part
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
        "*** YOUR CODE HERE ***"
        # Note-to-self: Can use underscore '_' to ignore specific values

        bestAction, _ = self.expectimax(gameState, 0, 0)
        return bestAction
    
    def expectimax(self, gameState, agentIndex, depth):
        # Check if the game is in a terminal state or the maximum depth is reached
        if gameState.isWin() or gameState.isLose() or depth >= self.depth * gameState.getNumAgents():
            return 'Stop', self.evaluationFunction(gameState)
        
        # If it's the maximizing player's turn (agentIndex == 0)
        if agentIndex == 0:
            # Call maxValue function to find the best action and value
            return self.maxValue(gameState, 0, depth)
        
        # If it's a chance node (expecting chance nodes in this case)
        else:
            return self.expectimaxValue(gameState, agentIndex, depth)
        
    # Determines the best action and associated value for the maximizing player
    def maxValue(self, gameState, agentIndex, depth):
        # Get legal actions for the current agent
        nextLegalActions = gameState.getLegalActions(agentIndex)
        nextStates = []  # Initialize a list to store the successor states

        # Generate successor states for each legal action
        for action in nextLegalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextStates.append(successorState) 
        
        # Calculate the index of the next agent
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        storeValues = []  # Initialize an empty list to store the values

        for state in nextStates:
            # Call self.expectimax to get the value for the current successor state
            # Assume that the expectimax function returns a tuple where the second element is the value we want
            value = self.expectimax(state, nextAgentIndex, depth + 1)[1]

            storeValues.append(value)
        
        # Find the maximum value among the successor states
        bestValue = max(storeValues)

        # Find the action corresponding to the best value
        bestAction = nextLegalActions[storeValues.index(bestValue)]
        return bestAction, bestValue
    
    # Evaluates the expected value for the current player 
    def expectimaxValue(self, gameState, agentIndex, depth):
        # Get legal actions for the current agent
        nextLegalActions = gameState.getLegalActions(agentIndex)
        nextStates = []  # Initialize a list to store the successor states

        # Generate successor states for each legal action
        for action in nextLegalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            nextStates.append(successorState) 
        
        # Calculate the index of the next agent
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        storeValues = []  # Initialize an empty list to store the values

        for state in nextStates:
            # Call self.expectimax to get the value for the current successor state
            # Assume that the expectimax function returns a tuple where the second element is the value we want
            value = self.expectimax(state, nextAgentIndex, depth + 1)[1]

            storeValues.append(value)
        
        # Calculate the mean value of the values of all successor states
        meanValue = sum(storeValues) / len(storeValues)

        # Return None as the action (no specific action is chosen) and the mean value
        return None, meanValue

# Dhriti's & Brittney's Part (Worked Together)
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Retrieves positions for pacman and ghosts
    newFood = list(currentGameState.getFood())
    distance = float("inf")
    for food in newFood:
        for i in food:
            if i == True:
                distance = min(distance, manhattanDistance(currentGameState.getPacmanPosition(), (newFood.index(food), food.index(i))))

    AgentIndexCounter = 1
    GhostDistance = 0
    distancetoScaredGhost = 0
    for i in currentGameState.getGhostStates():
        if manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(AgentIndexCounter)) < 2 and not i.scaredTimer:
            GhostDistance += manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(AgentIndexCounter))
            return -float("inf")
        if i.scaredTimer:
            distancetoScaredGhost += manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(AgentIndexCounter))
        AgentIndexCounter += 1

    capsuledistance = float("inf")
    for i in currentGameState.getCapsules():
        capsuledistance = min(capsuledistance, manhattanDistance(currentGameState.getPacmanPosition(), i))

    foodLeft = len(currentGameState.getFood().asList())
    capsuleLeft = len(currentGameState.getCapsules())
    score = currentGameState.getScore()

    return score + 1/(distance + 1) + 1/(capsuledistance + 1) + 1/(distancetoScaredGhost + 1) + GhostDistance - foodLeft - capsuleLeft

    ""
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPosition = currentGameState.getGhostPositions()

    # Retrieves list of food
    remainingFood = currentGameState.getFood().asList()
    foodCount = len(remainingFood)

    capsule = currentGameState.getCapsules()
    capsuleCount = len(capsule)
    closestFood = 1

    # Retrieve if game state is win or lose
    win = currentGameState.isWin()
    lose = currentGameState.isLose()

    score = currentGameState.getScore()

    # List to store the distances
    foodDistances = []

    # Iterate over each food position in the food list
    for position in remainingFood:
        # Calculate the Manhattan distance between Pacman's position and the current food position
        distance = manhattanDistance(pacmanPosition, position)

        # Add the distance to the list
        foodDistances.append(distance)

    if foodCount > 0:
        closestFood = min(foodDistances)

    for position in ghostPosition:
        ghostDistance = manhattanDistance(pacmanPosition, position)

        if ghostDistance < 2:
            closestFood = float('inf')

    features = [1.0 / closestFood, score, foodCount, capsuleCount, win, lose]

    weights = [50, 300, 100, 50, 500, -700]

    # Initialize a variable to store the total evaluation score
    totalScore = 0

    # Iterate through the indices of features
    for i in range(len(features)):
        # Get the feature and its corresponding weight
        feature = features[i]
        weight = weights[i]

        # Calculate the contribution of this feature to the total score
        featureContribution = feature * weight

        # Add this contribution to the total score
        totalScore += featureContribution

    return totalScore
    ""

# Abbreviation
better = betterEvaluationFunction
