from functools import partial

import util


class PathNode(object):

    __slots__ = ["state", "action", "cost", "parent"]

    @staticmethod
    def get_path(child, path=None):
        if path is None:
            path = []
        if child.parent is None:
            return list(reversed(path))
        path.append(child)
        return PathNode.get_path(child.parent, path)

    def __init__(self, pos, action=None, cost=None, parent=None):
        self.state = pos
        self.parent = parent
        self.cost = cost
        self.action = action

    def __repr__(self):
        return str((self.state, self.action))

    @property
    def path(self):
        return self.get_path(self)

    @property
    def path_cost(self):
        return sum([c.cost for c in self.path])

    @property
    def path_states(self):
        return [c.state for c in self.path]


def as_list(frontier):
    if hasattr(frontier, "list"):
        return [c.state for c in frontier.list]
    elif hasattr(frontier, "heap"):
        return [(p, c.state) for p, _, c in frontier.heap]


def check_successor(succ, depth, explored, frontier, path_cost):
    frontier_list = as_list(frontier)
    if depth is True and not succ in explored:
        return True, None
    elif depth is False and not succ in explored:
        if isinstance(frontier, util.Queue):
            if not succ in frontier_list:
                return True, None
        else:
            if not succ in [f[1] for f in frontier_list]:
                return True, None
            else:
                for i, f in enumerate(frontier_list):
                    if f[1] == succ and f[0] > path_cost:
                        return True, i
    return False, None


def do_search(problem, struct, depth=False):
    frontier = struct()
    frontier.push(PathNode(problem.getStartState()))
    explored = set(problem.getStartState())
    if problem.isGoalState(problem.getStartState()):
        return [problem.getStartState()]
    while not frontier.isEmpty():
        node = frontier.pop()
        explored.add(node.state)
        if problem.isGoalState(node.state):
            return [c.action for c in node.path]
        for successor, action, cost in problem.getSuccessors(node.state):
            succ_node = PathNode(successor, action, cost, node)
            if isinstance(frontier, PriorityQueueWithFunctionAndReplace):
                check_cost = frontier.priorityFunction(succ_node)
            else:
                check_cost = succ_node.path_cost
            check, replace = check_successor(successor, depth, explored, frontier, check_cost)
            if check is True:
                if replace is None:
                    frontier.push(succ_node)
                else:
                    frontier.replace(replace, succ_node)
    return []



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return do_search(problem, util.Stack, depth=True)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    return do_search(problem, util.Queue)


class PriorityQueueWithFunctionAndReplace(util.PriorityQueueWithFunction):

    def replace(self, i, child):
        del self.heap[i]
        self.push(child)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    def cost_func(x):
        if hasattr(problem, "costFn"):
            return sum([problem.costFn(state) for state in x.path_states])
        else:
            return x.path_cost

    return do_search(problem, partial(PriorityQueueWithFunctionAndReplace, cost_func))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    problem.heuristic = heuristic

    def heuristic_func(x):
        if hasattr(problem, "costFn"):
            path_cost = sum([problem.costFn(state) for state in x.path_states])
        else:
            path_cost = x.path_cost
        return path_cost + heuristic(x.state, problem)

    return do_search(problem, partial(PriorityQueueWithFunctionAndReplace, heuristic_func))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
