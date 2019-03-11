# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    stack.push(problem.getStartState())
    explored = util.Stack()
    movements = []
    parent = stack.pop()
    successors = problem.getSuccessors(parent)
    explored.push(parent)
    for s in successors:
        node = Node(state = s[0],direction=s[1])
        stack.push(node)
    while not stack.isEmpty():
        parent = stack.pop()
        state = parent.state
        if problem.isGoalState(state):
            break
        explored.push(state)
        next = problem.getSuccessors(state)
        for s in next:
            if (s[0] not in stack.list) and (s[0] not in explored.list):
                node = Node(s[0],s[1],parent)
                stack.push(node)
    moves = util.Stack()
    while not parent == None:
        moves.push(parent.direction)
        parent = parent.parent
    while not moves.isEmpty():
        movements.append(moves.pop())
    return movements

class Node:
    def __init__(self, state, direction,parent = None):
        self.state = state
        self.direction = direction
        self.parent = parent

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    queue.push(problem.getStartState())
    states = util.Queue()
    movements = []
    parent = queue.pop()
    successors = problem.getSuccessors(parent)
    explored = util.Queue()
    explored.push(parent)
    for s in successors:
        node = Node(s[0],s[1])
        states.push(s[0])
        queue.push(node)
    while not queue.isEmpty():
        parent = queue.pop()
        state = parent.state
        if problem.isGoalState(state):
            break
        successors = problem.getSuccessors(state)
        explored.push(state)
        for s in successors:
            node = Node(s[0],s[1],parent)
            if (not s[0] in states.list) and (not s[0] in explored.list):
                states.push(s[0])
                queue.push(node)
    moves = util.Stack()
    while not parent == None:
        moves.push(parent.direction)
        parent = parent.parent
    while not moves.isEmpty():
        movements.append(moves.pop())
    return movements
    util.raiseNotDefined()

class WeightedNode(Node):
    def __init__(self, state, direction, parent = None,stepCost = 0):
        self.state = state
        self.direction = direction
        self.parent = parent
        self.weight = parent.weight + stepCost

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    node = Node(problem.getStartState(),None,None)
    node.weight = 0
    movements = []
    states = []
    explored = []
    start = Node(None,None)
    parent = node
    explored.append(parent.state)
    states.append(parent.state)
    for s in problem.getSuccessors(parent.state):
        node = WeightedNode(s[0],s[1],parent,s[2])
        states.append(s[0])
        frontier.update(node,s[2])
    while not frontier.isEmpty():
        parent = frontier.pop()
        explored.append(parent.state)
        if problem.isGoalState(parent.state):
            start = parent
            break
        for s in problem.getSuccessors(parent.state):
            child = WeightedNode(s[0],s[1],parent,s[2])
            if (not child.state in explored) and (not child.state in states):
                frontier.update(child,child.weight)
                states.append(s[0])
    moves = util.Stack()
    while start.direction != None:
        moves.push(start.direction)
        start = start.parent
    while not moves.isEmpty():
        movements.append(moves.pop())
    return movements
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

class aStarNode(Node):
    """docstring for aStarNode."""
    def __init__(self, state, direction, parent = None, stepCost = 0, finalCost = 0):
        self.state = state
        self.direction = direction
        self.parent = parent
        self.weight = parent.weight + stepCost
        self.cost = finalCost + self.weight

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    node = Node(problem.getStartState(),None,None)
    explored = []
    states = []
    movements = []
    start = Node(None,None)
    node.weight = 0
    parent = node
    explored.append(parent.state)
    states.append(parent.state)
    for s in problem.getSuccessors(parent.state):
        w = heuristic(s[0],problem)
        node = aStarNode(s[0],s[1],parent,s[2],w)
        frontier.update(node,w)
        states.append(s[0])
    while not frontier.isEmpty():
        parent = frontier.pop()
        explored.append(parent.state)
        if problem.isGoalState(parent.state):
            start = parent
            break
        for s in problem.getSuccessors(parent.state):
            w = heuristic(s[0],problem)
            child = aStarNode(s[0],s[1],parent,s[2],w)
            if not child.state in explored and not child.state in states:
                frontier.update(child,child.cost)
                states.append(s[0])
    moves = util.Stack()
    while start.direction != None:
        moves.push(start.direction)
        start = start.parent
    while not moves.isEmpty():
        movements.append(moves.pop())
    return movements
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
