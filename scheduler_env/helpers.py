import json
from scheduler_env import Resource, Order, Task

# Opens JSON file passed as argument with order data and populates global variables orders and tasks
def load_orders_new_version(file):
    # Just in case we are reloading tasks
    tasks = []
    orders = []
    f = open(file)

    # returns JSON object as  a dictionary
    data = json.load(f)
    f.close()
    orders = data['orders']

    # General order of index
    stepIndex = 0

    for order in orders:
        # Initial index of steps within order
        orderIndex = stepIndex
        name = order['name']
        color = order['color']
        earliestStart = order['earliest_start']

        for step in order['tasks']:
            stepIndex += 1
            duration = step['duration']
            predecessor = step['predecessor']
            task_type = step['type']

            if not (predecessor is None):
                absPredecessor = predecessor + orderIndex

            task = {}
            # Sequence is the scheduling order, the series of which defines a State or Node.
            task['sequence'] = None
            task['index'] = stepIndex
            task['order'] = name
            task['color'] = color
            task['type'] = task_type

            if predecessor is None:
                task['predecessor'] = None
                task['earliest_start'] = earliestStart
            else:
                task['predecessor'] = absPredecessor
                task['earliest_start'] = None

            task['duration'] = duration
            task['start'] = None
            task['finish'] = None

            tasks.append(Task(task))

        orders.append(Order())
    return tasks
    



def load_orders(file):

    # Just in case we are reloading tasks
    tasks = []
    orders = []
    f = open(file)

    # returns JSON object as  a dictionary
    data = json.load(f)
    f.close()
    orders = data['orders']

    # General order of index
    stepIndex = 0

    for order in orders:
        # Initial index of steps within order
        orderIndex = stepIndex
        name = order['name']
        color = order['color']
        earliestStart = order['earliest_start']

        for step in order['steps']:
            stepIndex += 1
            # orderStep = step['step']
            resource = step['resource']
            duration = step['duration']
            predecessor = step['predecessor']

            if not (predecessor is None):
                absPredecessor = predecessor + orderIndex

            task = {}
            # Sequence is the scheduling order, the series of which defines a State or Node.
            task['sequence'] = None
            task['index'] = stepIndex
            task['order'] = name
            task['color'] = color
            task['resource'] = resource

            if predecessor is None:
                task['predecessor'] = None
                task['earliest_start'] = earliestStart
            else:
                task['predecessor'] = absPredecessor
                task['earliest_start'] = None

            task['duration'] = duration
            task['start'] = None
            task['finish'] = None

            tasks.append(task)
    return tasks
