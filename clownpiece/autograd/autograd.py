from typing import Dict, Iterable, List, Optional, Union, Any

from clownpiece.tensor import Tensor, ones_like, zeros_like
from clownpiece.utils_ import wrap_tuple

"""
    Autograd Module
"""

# autograd/autograd.py
class Node():
    node_id: int
    next_edges: List["Edge"]

    def __init__(self):
        self.node_id = None
        self.next_edges = []
        
    def run(self, *args, **kargs):
        raise NotImplementedError("run method not implemented for abstract Node instance")
    
    # define __hash__ and __eq__ to use Node as dict's key
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

class Edge():

    input_nr: int # the Edge points to the i-th input of target Node
    node: Optional[Node] # target node the Edge points to

    def __init__(self, input_nr: int, node: Optional[Node]):
        self.input_nr = input_nr
        self.node = node
    
    @staticmethod
    def gradient_edge(tensor: Tensor) -> "Edge":
        if tensor.grad_fn is not None:
            return Edge(tensor.output_nr, tensor.grad_fn)
        if tensor.requires_grad:
            from clownpiece.autograd.function import AccumulateGrad
            return Edge(0, AccumulateGrad(tensor))
        return Edge(0, None)

class GraphRoot(Node):
    """
    Root node in the computation graph.
    """

    def __init__(self, tensor: Tensor, grad: Tensor):
        super().__init__()
        self.grad = grad
        self.next_edges.append(Edge.gradient_edge(tensor))
    
    def run(self, *args, **kargs):
        return self.grad


class NodeTask():
    """
    NodeTask wraps a Node and all its input. 
    It's a ready-to-run Node in GraphTask.
    """

    base: "GraphTask"
    node: Node
    inputs: List[Tensor]
    
    def __init__(self, node: Node, inputs: List[Tensor], base: "GraphTask"):
        self.base = base
        self.node = node
        self.inputs = inputs
        
    def run(self):
        input_grads = wrap_tuple(self.node.run(*self.inputs))
        for edge, input_grad in zip(self.node.next_edges, input_grads):
            if edge.node is not None and input_grad is not None:
                self.base.fill_input(edge.node, input_grad, edge.input_nr)



class GraphTask():
    
    """
    GraphTask wraps the execution of a computation graph.
    """
    
    roots: List[Node] # GraphRoots instances
    nodes: List[Node] # all nodes in the computation graph
    dependencies: Dict[Node, int] # count of inbound degree for topological sort
    inputs_buffer: Dict[Node, List[Tensor]] # inputs_buffer to accumulate intermediate results.
    
    def __init__(self, roots: List[Node]):
        roots = wrap_tuple(roots)
        roots = [root for root in roots if root is not None]
        
        if not roots:
            raise ValueError("roots is empty")
    
        self.roots = roots
        self.nodes = []
        self.dependencies = {}
        self.inputs_buffer = {}
        self._construct_graph()
        
    # helper function to assign node_id and initialize self.nodes, dependencies and inputs_buffer
    def _construct_graph(self):
        i = 0
        visited = set()
        def dfs(start: Node):
            nonlocal i
            if start is None or id(start) in visited:
                return
            visited.add(id(start))
            start.node_id = i
            i += 1
            self.nodes.append(start)
            self.dependencies[start] = 0
            self.inputs_buffer[start] = []
            for edge in start.next_edges:
                target = edge.node
                if target is None:
                    continue
                dfs(target)
                self.dependencies[target] += 1
        for root in self.roots:
            dfs(root)
        
    # execute
    def run(self):
        self._run_multi_thread()

    # for debug
    def _run_single_thread(self):
        from collections import deque
        queue = deque()
        for root in self.roots:
            queue.append(NodeTask(root, [], self))
        while queue:
            node_task = queue.pop()
            node_task.run()
            for edge in node_task.node.next_edges:
                if edge.node is not None:
                    self.dependencies[edge.node] -= 1
                    if self.dependencies[edge.node] == 0:
                        queue.append(NodeTask(edge.node, self.inputs_buffer[edge.node], self))
            self.inputs_buffer.pop(node_task.node)


    # for production
    def _run_multi_thread(self):
        import os
        import threading
        from queue import Queue
        ready_queue = Queue()
        state_lock = threading.Lock()
        accumulate_lock = threading.Lock()
        stop = object()
        errors = []
        for root in self.roots:
            ready_queue.put(NodeTask(root, [], self))
        def worker():
            while True:
                node_task = ready_queue.get()
                try:
                    if node_task is stop:
                        return
                    if node_task.node.next_edges:
                        input_grads = wrap_tuple(node_task.node.run(*node_task.inputs))
                    else:
                        with accumulate_lock:
                            input_grads = wrap_tuple(node_task.node.run(*node_task.inputs))
                    with state_lock:
                        for edge, input_grad in zip(node_task.node.next_edges, input_grads):
                            if edge.node is not None and input_grad is not None:
                                self.fill_input(edge.node, input_grad, edge.input_nr)
                        for edge in node_task.node.next_edges:
                            target = edge.node
                            if target is None:
                                continue
                            self.dependencies[target] -= 1
                            if self.dependencies[target] == 0:
                                ready_queue.put(NodeTask(target, self.inputs_buffer[target], self,))
                        self.inputs_buffer.pop(node_task.node, None)
                except Exception as error:
                    with state_lock:
                        if not errors:
                            errors.append(error)
                finally:
                    ready_queue.task_done()

        worker_count = min(len(self.nodes), os.cpu_count() or 1)
        workers = [threading.Thread(target=worker) for _ in range(worker_count)]
        for thread in workers:
            thread.start()
        ready_queue.join()
        for _ in workers:
            ready_queue.put(stop)
        for thread in workers:
            thread.join()
        if errors:
            raise errors[0]
                    
    # accumulate input_grad to self.inputs_buffer[node][input_nr]
    def fill_input(self, node: Node, input_grad: Tensor, input_nr: int):
        buffer = self.inputs_buffer[node]
        while len(buffer) <= input_nr:
            buffer.append(None)
        if buffer[input_nr] is None:
            buffer[input_nr] = input_grad
        else:
            buffer[input_nr] += input_grad


"""
    Execute backward pass.    
"""
def backward(tensors: Union[Tensor, List[Tensor]], grads: Optional[Union[Tensor, List[Tensor]]] = None):
    tensors = wrap_tuple(tensors)

    if grads is None:
        grads = [ones_like(tensor) for tensor in tensors]
    grads = wrap_tuple(grads)
    
    # wrap with GraphRoots
    graph_roots = [
        GraphRoot(tensor, grad) for tensor, grad in zip(tensors, grads) if tensor.requires_grad
    ]

    # execute with GraphTask
    gt = GraphTask(graph_roots)
    gt.run()
