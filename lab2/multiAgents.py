# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # use the idea of what I did for project 1
    foods = newFood.asList()
    tempFoods = []
    currentPos = newPos
    foodAllMinDistance = 0
    for food in foods:
        tempFoods.append(food)
    while (len(tempFoods) > 0):
        minDistance, minPos = findMinDistance(tempFoods, currentPos)
        foodAllMinDistance += minDistance
        currentPos = minPos
        tempFoods.remove(minPos)

    ghostPositions = []
    ghostDistances = []
    currentPos = newPos
    ghostAllMinDistance = 0
    for index in range(1, len(newGhostStates)+1):
        ghostPosition = successorGameState.getGhostPosition(index)
        ghostPositions.append(ghostPosition)
        distance = util.manhattanDistance(ghostPosition, newPos)
        ghostDistances.append(distance)

    while (len(ghostPositions) > 0):
        minDistance, minPos = findMinDistance(ghostPositions, currentPos)
        ghostAllMaxDistance = minDistance
        currentPos = minPos
        ghostPositions.remove(minPos)

    allDistance = foodAllMinDistance + ghostAllMinDistance

    scares = []
    for scare in newScaredTimes:
        if scare > 0:
            scares.append(scare)

    if allDistance == 0:
        return successorGameState.getScore()
    if len(scares) > 0:
        return successorGameState.getScore() + allDistance
    if min(ghostDistances) < 2:
        return -allDistance
    return successorGameState.getScore() - allDistance

# a helper method
def findMinDistance(ls, position):
    if len(ls) == 0:
        return 0
    minDis = util.manhattanDistance(ls[0], position)
    minPos = ls[0]
    for point in ls:
        distance = util.manhattanDistance(point, position)
        if (distance < min):
            minDis = distance
            minPos = point
    return minDis, minPos

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    currentMax = -10000
    legalActions = gameState.getLegalActions(0)
    currentAction = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.minValue(successor, 1, 0)
      if (score > currentMax and action != Directions.STOP):
        currentMax = score
        currentAction = action
    return currentAction

  def isEnd(self, gameState, depth):
    if (depth == self.depth or gameState.isWin() or gameState.isLose()):
      return True

  def minValue(self, gameState, agentID, depth):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    currentMin = 10000

    if agentID >= gameState.getNumAgents():
      return self.maxValue(gameState, 0, depth+1)

    legalActions = gameState.getLegalActions(agentID)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      currentMin = min(currentMin, self.maxValue(newGameState, agentID+1, depth))
    return currentMin

  def maxValue(self, gameState, agentID, depth):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID != 0:
      return self.minValue(gameState, agentID, depth)

    currentMax = -10000

    # since we always want to maximum the value of the pacman
    legalActions = gameState.getLegalActions(0)
    for action in legalActions:
      if action != Directions.STOP:
        newGameState = gameState.generateSuccessor(agentID, action)
        currentMax = max(currentMax, self.minValue(newGameState, 1, depth))
    return currentMax

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    currentMax = -10000
    currentMin = 10000
    legalActions = gameState.getLegalActions(0)
    currentAction = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.minValue(successor, 1, 0, currentMax, currentMin)
      if (score > currentMax and action != Directions.STOP):
        currentMax = score
        currentAction = action
    return currentAction

  def isEnd(self, gameState, depth):
    if (depth == self.depth or gameState.isWin() or gameState.isLose()):
      return True

  def minValue(self, gameState, agentID, depth, alpha, beta):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    value = 10000

    if agentID >= gameState.getNumAgents():
      agentID = 0
      return self.maxValue(gameState, agentID, depth+1, alpha, beta)

    legalActions = gameState.getLegalActions(agentID)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      value = min(value, self.maxValue(newGameState, agentID+1, depth, alpha, beta))
      beta = min(beta, value)
      if (alpha >= beta):
        return value
    return value

  def maxValue(self, gameState, agentID, depth, alpha, beta):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID != 0:
      return self.minValue(gameState, agentID, depth, alpha, beta)

    value = -10000

    # since we always want to maximum the value of the pacman
    legalActions = gameState.getLegalActions(0)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      value = max(value, self.minValue(newGameState, 1, depth, alpha, beta))
      alpha = max(alpha, value)
      if (alpha >= beta):
        return value
    return value

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
    currentMax = -10000
    legalActions = gameState.getLegalActions(0)
    currentAction = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.expectValue(successor, 1, 0)
      if (score > currentMax and action != Directions.STOP):
        currentMax = score
        currentAction = action

    return currentAction
  
  def isEnd(self, gameState, depth):
    if (depth == self.depth or gameState.isWin() or gameState.isLose()):
      return True

  def expectValue(self, gameState, agentID, depth):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    value = 0

    if agentID >= gameState.getNumAgents():
      agentID = 0
      return self.maxValue(gameState, agentID, depth+1)

    legalActions = gameState.getLegalActions(agentID)
    scores = []
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      scores.append(self.maxValue(newGameState, agentID+1, depth))
    if (len(scores) == 0):
      return value
    value = sum(scores) / len(scores)
    return value

  def maxValue(self, gameState, agentID, depth):
    if (self.isEnd(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID != 0:
      return self.expectValue(gameState, agentID, depth)

    value = -10000
    # since we always want to maximum the value of the pacman
    legalActions = gameState.getLegalActions(0)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      value = max(value, self.expectValue(newGameState, 1, depth))
    return value


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    first calculating the closest food position, and find the distance from the current pacman position
    to that food. Then, assume pacman is already in that position, and finding the cloest food again. Do it over and over again
    until there is no more food. Sum up all distances to get the foodAllMinDistance.
    Do the same thing for the ghost, then we can get ghostAllMinDistance.
    Sum up foodAllMinDistances and ghostAllMinDistances to get the allDistance.

    Then, if there is the closest ghost is only one unit away from the pacman, return negative allDistance. If the second
    cloest ghost is also less than two units away from the pacman, return negative allDistance times 2.

    Since we want to get to the food as close as possible and stay away from the ghost as far as possible.
    Then, if there is only one more food left, return current score + ghostAllMinDistance - foodAllMinDistance.
    If the farest food is less than 2 units from current pacman, return current score - max food distance.
    If scares exist, return current score - max ghost distance * 2
    Otherwise, return current score + ghostAllMinDistance - foodAllMinDistance.
  """
  "*** YOUR CODE HERE ***"
  pos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  foods = newFood.asList()

  tempFoods = []
  foodDistances = []
  currentPos = pos
  foodAllMinDistance = 0
  for food in foods:
    tempFoods.append(food)
    distance = util.manhattanDistance(food, pos)
    foodDistances.append(distance)

  while (len(tempFoods) > 0):
    minDistance, minPos = findMinDistance(tempFoods, currentPos)
    foodAllMinDistance += minDistance
    currentPos = minPos
    tempFoods.remove(minPos)

  ghostPositions = []
  ghostDistances = []
  currentPos = pos
  ghostAllMinDistance = 0
  for index in range(1, len(ghostStates)+1):
    ghostPosition = currentGameState.getGhostPosition(index)
    ghostPositions.append(ghostPosition)
    distance = util.manhattanDistance(ghostPosition, pos)
    ghostDistances.append(distance)

  # while (len(ghostPositions) > 0):
  #   minDistance, minPos = findMinDistance(ghostPositions, currentPos)
  #   currentPos = minPos
  #   ghostPositions.remove(minPos)

  allDistance = foodAllMinDistance + ghostAllMinDistance

  scares = []
  for scare in scaredTimes:
    if scare > 0:
      scares.append(scare)

  if min(ghostDistances) <= 1:
    if findSecondSmallest(ghostDistances) <= 2:
      return -allDistance * 2
    return -allDistance

  if len(foodDistances) <= 1:
    return currentGameState.getScore() + ghostAllMinDistance - foodAllMinDistance

  if len(scares) > 0:
    return currentGameState.getScore() - min(foodDistances) - min(ghostDistances) * len(scares) * 2

  if max(foodDistances) < 2:
    return currentGameState.getScore() - max(foodDistances)

  return currentGameState.getScore() + ghostAllMinDistance - foodAllMinDistance

# Abbreviation
better = betterEvaluationFunction

def findSecondSmallest(ls):
  tempList = ls
  tempList.remove(min(tempList))
  return min(tempList)

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    currentMax = -10000
    currentMin = 10000
    legalActions = gameState.getLegalActions(0)
    currentAction = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.minValue(successor, 1, 0, currentMax, currentMin)
      if (score > currentMax and action != Directions.STOP):
        currentMax = score
        currentAction = action
    return currentAction

  def isEnd(self, gameState, depth):
    if (depth == self.depth or gameState.isWin() or gameState.isLose()):
      return True

  def minValue(self, gameState, agentID, depth, alpha, beta):
    if (self.isEnd(gameState, depth)):
      return betterEvaluationFunction2(gameState)

    value = 10000

    if agentID >= gameState.getNumAgents():
      agentID = 0
      return self.maxValue(gameState, agentID, depth+1, alpha, beta)

    legalActions = gameState.getLegalActions(agentID)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      value = min(value, self.maxValue(newGameState, agentID+1, depth, alpha, beta))
      beta = min(beta, value)
      if (alpha >= beta):
        return value
    return value

  def maxValue(self, gameState, agentID, depth, alpha, beta):
    if (self.isEnd(gameState, depth)):
      return betterEvaluationFunction2(gameState)

    if agentID != 0:
      return self.minValue(gameState, agentID, depth, alpha, beta)

    value = -10000

    # since we always want to maximum the value of the pacman
    legalActions = gameState.getLegalActions(0)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      value = max(value, self.minValue(newGameState, 1, depth, alpha, beta))
      alpha = max(alpha, value)
      if (alpha >= beta):
        return value
    return value


def betterEvaluationFunction2(currentGameState):
  pos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  foods = newFood.asList()

  tempFoods = []
  foodDistances = []
  currentPos = pos
  foodAllMinDistance = 0
  for food in foods:
    tempFoods.append(food)
    distance = util.manhattanDistance(food, pos)
    foodDistances.append(distance)

  while (len(tempFoods) > 0):
    minDistance, minPos = findMinDistance(tempFoods, currentPos)
    foodAllMinDistance += minDistance
    currentPos = minPos
    tempFoods.remove(minPos)

  ghostPositions = []
  ghostDistances = []
  currentPos = pos
  ghostAllMinDistance = 0
  ghostScares = 0
  for index in range(1, len(ghostStates)+1):
    ghostPosition = currentGameState.getGhostPosition(index)     
    ghostPositions.append(ghostPosition)
    distance = util.manhattanDistance(ghostPosition, pos)
    ghostDistances.append(distance)
    ghostScare = ghostStates[index-1].scaredTimer
    coef = 0
    if (distance > 0):
      coef = 1 / distance
    if ghostScare <= 0:
      if (min(ghostDistances) <= 1):
        ghostScares -= 10
      ghostScares -= 5 * coef
    else:
      ghostScares += 50 * coef

  if (len(foods) > 0):
    return currentGameState.getScore() / 2 + ghostScares - 1.1 * min(foodDistances)
  else:
    return currentGameState.getScore() / 2 + ghostScares

  # while (len(ghostPositions) > 0):
  #   minDistance, minPos = findMinDistance(ghostPositions, currentPos)
  #   currentPos = minPos
  #   ghostPositions.remove(minPos)

  # allDistance = foodAllMinDistance + ghostAllMinDistance

  # scares = []
  # for scare in scaredTimes:
  #   if scare > 0:
  #     scares.append(scare)

  # if min(ghostDistances) <= 1:
  #   if findSecondSmallest(ghostDistances) <= 1:
  #     return -allDistance * 2.5
  #   return -allDistance

  # if len(foodDistances) <= 5:
  #   return currentGameState.getScore() + ghostAllMinDistance - foodAllMinDistance

  # if len(scares) > 0:
  #   return currentGameState.getScore() * 3 - min(foodDistances) - min(ghostDistances) * len(scares) * 1.3

  # if max(foodDistances) < 2:
  #   return currentGameState.getScore() - max(foodDistances)

  # return currentGameState.getScore() + ghostAllMinDistance - foodAllMinDistance

  # pos = currentGameState.getPacmanPosition()
  # newFood = currentGameState.getFood()
  # ghostStates = currentGameState.getGhostStates()
  # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  # foods = newFood.asList()
  # walls = currentGameState.getWalls()
  # top = walls.height - 2
  # width = walls.width - 2
  # upperLeftDistance = util.manhattanDistance(pos, (1, top))
  # upperRightDistance = util.manhattanDistance(pos, (width, top))
  # lowerLeftDistance = util.manhattanDistance(pos, (1, 1))
  # upperLeftHasFood = newFood[1][2]
  # upperRightHasFood = newFood[width][top-1]
  # lowerLeftHasFood = newFood[1][7]
  # tempFoods = []
  # foodDistances = []
  # currentPos = pos
  # foodAllMinDistance = 0
  # for food in foods:
  #   tempFoods.append(food)
  #   distance = util.manhattanDistance(food, pos)
  #   foodDistances.append(distance)

  # while (len(tempFoods) > 0):
  #   minDistance, minPos = findMinDistance(tempFoods, currentPos)
  #   foodAllMinDistance += minDistance
  #   currentPos = minPos
  #   tempFoods.remove(minPos)

  # ghostPositions = []
  # ghostDistances = []
  # currentPos = pos
  # ghostAllMinDistance = 0
  # ghostScares = 0
  # for index in range(1, len(ghostStates)+1):
  #   ghostPosition = currentGameState.getGhostPosition(index)
  #   ghostPositions.append(ghostPosition)
  #   distance = util.manhattanDistance(ghostPosition, pos)
  #   ghostDistances.append(distance)
  #   ghostScare = ghostStates[index-1].scaredTimer
  #   if ghostScare > 0:
  #     ghostScares += ghostScare / (distance + 1)


  # allDistance = foodAllMinDistance + ghostAllMinDistance

  # leftLower = currentGameState
  # scares = []
  # for scare in scaredTimes:
  #   if scare > 0:
  #     scares.append(scare)
  
  # score = 100
  # if min(ghostDistances) <= 1:
  #     if findSecondSmallest(ghostDistances) <= 1:
  #       return -allDistance * 2
  #     return -allDistance

  # if len(foodDistances) <= 5 and len(scares) == 0:
  #   return currentGameState.getScore() + ghostAllMinDistance - foodAllMinDistance

  # if len(scares) > 0:
  #   #return currentGameState.getScore() - foodAllMinDistance + max(scares) * len(scares) 
  #   return currentGameState.getScore() - min(foodDistances) - min(ghostDistances) * len(scares)

  # if upperLeftHasFood:
  #   score = score - upperLeftDistance + lowerLeftDistance + upperRightDistance
  # else:
  #   if lowerLeftHasFood:
  #     score = score - lowerLeftDistance + upperRightDistance
  #   else:
  #     if upperRightHasFood:
  #       score -= upperRightDistance
