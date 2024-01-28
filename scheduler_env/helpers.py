import json

def load_resources(file_path):
    resources = []

    with open(file_path, 'r') as file:
        data = json.load(file)

    for resource_data in data["resources"]:
        resource = {}
        resource['name'] = resource_data["name"]
        resource['ability'] = resource_data["type"].split(', ')
        resources.append(resource)

    return resources

# Opens JSON file passed as argument with order data and populates global variables orders and tasks
def load_orders_new_version(file):
    # Just in case we are reloading tasks
    
    orders = [] # 리턴할 용도
    orders_new_version = [] # 파일 읽고 저장할 때 쓰는 용도
    f = open(file)

    # returns JSON object as  a dictionary
    data = json.load(f)
    f.close()
    orders_new_version = data['orders']

    for order in orders_new_version:
        order_dictonary = {}
        # Initial index of steps within order
        order_dictonary['name'] = order['name']
        order_dictonary['color'] = order['color']
        earliestStart = order['earliest_start']

        tasks = []
        for task in order['tasks']:
            predecessor = task['predecessor']
            task_dictionary = {}
            # Sequence is the scheduling order, the series of which defines a State or Node.
            task_dictionary['sequence'] = None
            task_dictionary['step'] = task['step']
            task_dictionary['type'] = task['type']
            if predecessor is None:
                task_dictionary['predecessor'] = None
                task_dictionary['earliest_start'] = earliestStart
            else:
                task_dictionary['predecessor'] = predecessor
                task_dictionary['earliest_start'] = None
            task_dictionary['duration'] = task['duration']
            task_dictionary['start'] = None
            task_dictionary['finish'] = None

            tasks.append(task_dictionary)
        
        order_dictonary['tasks'] = tasks
        orders.append(order_dictonary)

    return orders



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
