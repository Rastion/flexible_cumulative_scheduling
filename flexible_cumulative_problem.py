from qubots.base_problem import BaseProblem
import random

class FlexibleCumulativeProblem(BaseProblem):
    """
    Flexible Cumulative Scheduling Problem for Qubots.

    A project consists of a set of tasks that must be scheduled on renewable resources.
    Each task may be processed by one (or more) compatible resources.
    For each task and resource, a processing time and a resource consumption (weight) are given.
    If both values are zero for a resource, that resource cannot process the task.
    There are precedence constraints (a task must finish before its successors start)
    and cumulative constraints on each resource (the sum of weights of concurrently running tasks
    cannot exceed the resource capacity).

    The objective is to minimize the makespan—the time by which all tasks are finished.

    **Candidate Solution Format:**
      A tuple: (task_resources, start_times)
      - task_resources: List of length nb_tasks where each element is the index (0-based) of the selected resource.
      - start_times: List of length nb_tasks where each element is the start time for that task.
    """

    def __init__(self, instance_file: str):
        """
        Initializes the problem by reading the instance file.
        
        Instance file format:
          - First line: <nb_tasks> <nb_resources>
          - Second line: <capacity_0> <capacity_1> ... <capacity_{nb_resources-1}>
          - Next nb_tasks lines: For each task, 2*nb_resources integers representing:
              processing time and weight for each resource (in that order).
          - Next nb_tasks lines: For each task, first integer is the number of successors,
              followed by the successor task IDs.
        """
        (self.nb_tasks, self.nb_resources, self.capacity, 
         self.task_processing_time_data, self.weights, 
         self.nb_successors, self.successors, self.horizon) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, 'r') as f:
            lines = f.readlines()
        # Remove empty lines and strip whitespace.
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line: number of tasks and number of resources.
        first_line = lines[0].split()
        nb_tasks = int(first_line[0])
        nb_resources = int(first_line[1])
        
        # Second line: resource capacities.
        capacity = list(map(int, lines[1].split()))
        
        # Next nb_tasks lines: processing times and weights.
        task_processing_time_data = []
        # Initialize weights as a list per resource.
        weights = [[] for _ in range(nb_resources)]
        for i in range(nb_tasks):
            parts = lines[2 + i].split()
            durations = []
            for r in range(nb_resources):
                duration = int(parts[2 * r])
                durations.append(duration)
                weight_val = int(parts[2 * r + 1])
                weights[r].append(weight_val)
            task_processing_time_data.append(durations)
        
        # Next nb_tasks lines: successors.
        nb_successors = []
        successors = []
        for i in range(nb_tasks):
            parts = lines[2 + nb_tasks + i].split()
            num_succ = int(parts[0])
            nb_successors.append(num_succ)
            succ_list = list(map(int, parts[1:])) if num_succ > 0 else []
            successors.append(succ_list)
        
        # A trivial horizon: sum of maximum processing times over tasks.
        horizon = sum(max(task_processing_time_data[i]) for i in range(nb_tasks))
        return nb_tasks, nb_resources, capacity, task_processing_time_data, weights, nb_successors, successors, horizon

    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution = (task_resources, start_times)
          - task_resources: list with the resource selected for each task.
          - start_times: list with the start time for each task.
        
        Returns:
          - The makespan (maximum finish time) if the solution is feasible.
          - A high penalty value (1e9) if any constraint is violated.
        """
        penalty = 1e9
        # Validate the solution format.
        if not (isinstance(solution, (list, tuple)) and len(solution) == 2):
            return penalty
        task_resources, start_times = solution
        if len(task_resources) != self.nb_tasks or len(start_times) != self.nb_tasks:
            return penalty

        finish_times = [0] * self.nb_tasks
        
        # Compute finish times and check resource compatibility.
        for i in range(self.nb_tasks):
            r = task_resources[i]
            st = start_times[i]
            # Check that the resource index is valid.
            if r < 0 or r >= self.nb_resources:
                return penalty
            proc_time = self.task_processing_time_data[i][r]
            weight = self.weights[r][i]
            # Incompatibility: if both processing time and weight are zero, the resource cannot process the task.
            if proc_time == 0 and weight == 0:
                return penalty
            finish_times[i] = st + proc_time
        
        # Check precedence constraints: each task must finish before its successors start.
        for i in range(self.nb_tasks):
            for succ in self.successors[i]:
                if finish_times[i] > start_times[succ]:
                    return penalty
        
        # Check cumulative resource constraints for each resource.
        for r in range(self.nb_resources):
            events = []
            # For tasks assigned to resource r, record start and finish events.
            for i in range(self.nb_tasks):
                if task_resources[i] == r:
                    st = start_times[i]
                    ft = finish_times[i]
                    w = self.weights[r][i]
                    events.append((st, w))
                    events.append((ft, -w))
            events.sort(key=lambda x: x[0])
            current_usage = 0
            for time, delta in events:
                current_usage += delta
                if current_usage > self.capacity[r]:
                    return penalty
        
        # Feasible solution: return the makespan.
        makespan = max(finish_times)
        return makespan

    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each task, it randomly selects one of the compatible resources
        (i.e. those for which processing time or weight is nonzero)
        and assigns a random start time between 0 and the horizon.
        
        Note: The generated solution is not guaranteed to satisfy all constraints.
        """
        task_resources = []
        start_times = []
        for i in range(self.nb_tasks):
            # Find all resources that can process task i.
            compatible_resources = [
                r for r in range(self.nb_resources)
                if not (self.task_processing_time_data[i][r] == 0 and self.weights[r][i] == 0)
            ]
            if not compatible_resources:
                task_resources.append(0)
            else:
                task_resources.append(random.choice(compatible_resources))
            start_times.append(random.randint(0, self.horizon))
        return (task_resources, start_times)
