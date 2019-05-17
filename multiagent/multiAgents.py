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
from __future__ import division

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
		import math
		if action == Directions.STOP:
			return float("-inf")
		for i,GS in enumerate(currentGameState.getGhostStates()):
			if util.manhattanDistance(newGhostStates[i].getPosition(), newPos) < 2 and newScaredTimes[i]==0:
				return float("-inf")
		foodDists = []
		score = 0
		for pos in newFood.asList():
			foodDists.append(util.manhattanDistance(newPos,pos))
		if currentGameState.hasFood(newPos[0],newPos[1]):
			score += 1
		if foodDists != []:
			minDist = min(foodDists)
			score += 1. /minDist
		#print(int(score))
		return score

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

	"""
	  https://fr.wikipedia.org/wiki/Algorithme_minimax#Pseudocode
	"""
	def minimax(self,state,depth,agent):

		# making sure we don't get out of bounds when calling with agent+1
		agent %= self.agentCount

		# if terminal state
		if depth == 0 or state.isWin() or state.isLose():
			return (Directions.STOP, self.evaluationFunction(state))

		# if maximizing player
		elif agent == self.index:
			value = float("-inf")
			ret = Directions.STOP
			for action in state.getLegalActions(agent):
				tmp = self.minimax(state.generateSuccessor(agent,action),depth-1,agent+1)[1]
				if tmp > value:
					ret = action
					value = tmp
			return (ret, value)

		# if minimizing player
		else:
			value = float("inf")
			ret = Directions.STOP
			for action in state.getLegalActions(agent):
				tmp = self.minimax(state.generateSuccessor(agent,action),depth-1,agent+1)[1]
				if tmp < value:
					ret = action
					value = tmp
			return (ret, value)
	
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
		"*** YOUR CODE HERE ***"
		self.agentCount = gameState.getNumAgents()

		return self.minimax(gameState,self.depth*self.agentCount,self.index)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	"""
	  https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
	"""
	def alphabetasearch(self,state,depth,alpha,beta,agent):

		# making sure we don't get out of bounds when calling with agent+1
		agent %= self.agentCount

		# if terminal state
		if depth == 0 or state.isWin() or state.isLose():
			return (Directions.STOP, self.evaluationFunction(state))

		# if maximizing player
		elif agent == self.index:
			value = float("-inf")
			ret = Directions.STOP
			for action in state.getLegalActions(agent):
				tmp = self.alphabetasearch(state.generateSuccessor(agent,action),depth-1,alpha,beta,agent+1)[1]
				if tmp > value:
					ret = action
					value = tmp
				if value > beta:
					return (action, value)
				alpha = max(alpha, value)
			return (ret, value)

		# if minimizing player
		else:
			value = float("inf")
			ret = Directions.STOP
			for action in state.getLegalActions(agent):
				tmp = self.alphabetasearch(state.generateSuccessor(agent,action),depth-1,alpha,beta,agent+1)[1]
				if tmp < value:
					ret = action
					value = tmp
				if value < alpha:
					return (action, value)
				beta = min(beta, value)
			return (ret, value)

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		self.agentCount=gameState.getNumAgents()
		alpha=float("-inf")
		beta=float("inf")

		return self.alphabetasearch(gameState,self.depth*self.agentCount,alpha,beta,self.index)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def expectimax(self,state,depth,agent):

		# making sure we don't get out of bounds when calling with agent+1
		agent %= self.agentCount

		# if terminal state
		if depth == 0 or state.isWin() or state.isLose():
			return self.evaluationFunction(state)

		# if maximizing player
		elif agent == self.index:
			value = float("-inf")
			for action in state.getLegalActions(agent):
				value = max(value,self.expectimax(state.generateSuccessor(agent,action),depth-1,agent+1))
			return value

		# if random player
		else:
			alpha = 0
			actions = state.getLegalActions(agent)
			n = len(actions)
			for action in actions:
				alpha += ((1/n) * self.expectimax(state.generateSuccessor(agent, action),depth-1,agent+1))    
			return alpha
			#return (ret, value)

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		self.agentCount = gameState.getNumAgents()
		depth=self.depth*self.agentCount
		agent = self.index

		actionsValues = {}

		for action in gameState.getLegalActions(agent):
			value = self.expectimax(gameState.generateSuccessor(self.index, action),depth-1, agent+1)
			actionsValues[value]=action

		return actionsValues[max(actionsValues)]

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"

	# We obviously check for end-game state
	if currentGameState.isLose():
		return float("-inf")
	elif currentGameState.isWin():
		return float("inf")

	# Now we retrieve all the data we want to use

	#pacman position
	pos = currentGameState.getPacmanPosition()

	#current score
	score=scoreEvaluationFunction(currentGameState)

	#food
	food = currentGameState.getFood().asList()
	#number of food left
	nfood = len(food)

	#capsules
	capsules = currentGameState.getCapsules()
	#number of capsules left
	ncapsules = len(capsules)

	#ghosts
	ghosts = currentGameState.getGhostStates()

	# Parametersthat will matter
	closestFood = 0
	closestCapsule = 0
	closestGhost = 0
	closestScaredGhost = 0

	# closest food
	closestFood = closestFromList(pos, food)

	# closest capsule
	closestCapsule = closestFromList(pos, capsules)

	# split ghosts
	aggressiveGhosts = []
	scaredGhosts = []

	for ghost in ghosts:
		if ghost.scaredTimer:
			scaredGhosts.append(ghost)
		else:
			aggressiveGhosts.append(ghost)

	# closest aggressive ghost
	closestGhost = closestFromList(pos, [ghost.getPosition() for ghost in aggressiveGhosts])

	# closest scared ghost
	closestScaredGhost = closestFromList(pos, [ghost.getPosition() for ghost in scaredGhosts])

	evaluation = 0
	evaluation += score # Takes too long without accounting score
	evaluation += -closestFood # Best coef seems to be 1
	#evaluation += -closestCapsule # This ruins everything
	evaluation += closestGhost # Helps with speed
	#evaluation += closestScaredGhost # Seems to be better ignoring em
	evaluation += -nfood*3
	evaluation += -ncapsules*29 # This with high coef really helps.

	return evaluation




def closestFromList(agentPos, targetList):
	if targetList:
		return min(map(lambda t: util.manhattanDistance(agentPos, t), targetList))
	else:
		return 0

# Abbreviation
better = betterEvaluationFunction
