from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl, base
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.communication import new_group
import bagua.torch_api as bagua
import torch
import math
from typing import Dict, List, Tuple

class AdasumTensorBuffers:
    def __init__(self, rec_levels, param) -> None:
        # torch.cuda.empty_cache()
        self.buffer = []
        self.rank = bagua.get_rank()

        start_dim = param.size()
        for i in range(rec_levels):
            dim_left, dim_right = self.get_dimensions(i + 1, start_dim)  # TODO: check i+1 correct
            if self.is_left(i + 1):
                self.buffer.append(BufferHelp(send_dim=dim_right, recv_dim=dim_left, end_dim=dim_right))
            else:
                self.buffer.append(BufferHelp(send_dim=dim_left, recv_dim=dim_right, end_dim=dim_left))

    def get_buffers_of_level(self, level):
        send_buf = self.buffer[level - 1].send_buf
        recv_buf = self.buffer[level - 1].recv_buf  # equals dim_right
        end_dim = self.buffer[level - 1].end_buf
        return send_buf, recv_buf, end_dim

    def is_left(self, rec_level):
        # Checks if dimension of current recursion level is left or right
        return math.floor(self.rank / math.pow(2, rec_level - 1)) % 2 == 0

    def get_dimensions(self, rec_level, start_dim):
        # Need to create a list becaue can't modify a torch.Size object directly. Convert later to tuple
        dim = []
        for i in range(len(start_dim)):
            dim.append(start_dim[i])
        other_dim = dim.copy()
        for i in range (rec_level):
            curr_dim = dim[0]
            if self.is_left(i + 1):
                # Get dimension of left part
                dim[0] = math.floor(curr_dim / 2)
                other_dim[0] = math.ceil(curr_dim / 2)
            else:
                # Get dimension of right part
                dim[0] = math.ceil(curr_dim / 2)
                other_dim[0] = math.floor(curr_dim / 2)
        
        if self.is_left(rec_level):  
            dim_left = dim
            dim_right = other_dim
        else:
            dim_left = other_dim
            dim_right = dim

        return tuple(dim_left), tuple(dim_right)

class BufferHelp:
    def __init__(self, send_dim, recv_dim, end_dim) -> None:
        self.send_buf = torch.zeros(size=send_dim).cuda()
        self.recv_buf = torch.zeros(size=recv_dim).cuda()
        self.end_buf = torch.zeros(size=end_dim).cuda()  # Opposite dimension compared to recv buffers? Bc we processed dim of recv buffer

class AdasumAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self, 
        process_group: BaguaProcessGroup,
        model_params
    ):
        super(AdasumAlgorithmImpl, self).__init__(process_group)
        self.buffers = []
        self.partial_dots_buffer = torch.zeros(3).cuda()
        self.rank = bagua.get_rank()
        self.world_size = bagua.get_world_size()
        self.comm_groups = self.init_comm_groups()

        rec_levels = int(math.log2(self.world_size))
        for param in model_params:  
                self.buffers.append(AdasumTensorBuffers(rec_levels, param))

    def get_buffers(self, param_idx, rec_level):  # TODO: make individually?
        return self.buffers[param_idx].get_buffers_of_level(rec_level)

    def init_comm_groups(self):
        comm_groups = {}
        rec_levels = int(math.log2(self.world_size))
        for i in range(rec_levels):
            comm_groups[i+1] = self.create_groups(i+1)
        return comm_groups

    def create_groups(self, rec_level) -> List[BaguaProcessGroup]:  # TODO: Need to check if they are the same across processes
        groups = []
        grp_size = 2 ** rec_level
        counter = 0
        curr_grp = []
        # Outer loop controls how many comm groups are created per recursion level
        for i in range(self.world_size):  # Should be divisible by grp_size, i.e. a power of 2
            if counter < grp_size:
                curr_grp.append(i)
                counter += 1
                if counter == grp_size:
                    group = new_group(ranks=curr_grp)
                    groups.append(group)
                    curr_grp = []
                    counter = 0
        return groups

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_adasum():
            """
            Still need to write
            """
            def adasum(gradient: torch.Tensor, distance, param_idx):
                rank = bagua.get_rank()
                mid = math.floor(gradient.size()[0]/2)  # split current gradient across first dimension
                rec_level = int(math.log2(distance) + 1)
                send_buf, recv_buf, end_buf = self.get_buffers(param_idx, rec_level)  # TODO: Check "send_buf is self.buffer. etc."
                # send_buf2 = self.buffers[param_idx].buffer[rec_level - 1].send_buf // is the same
                is_left = True

                if math.floor(rank/distance) % 2 == 0:              # process right half
                    neighbor = rank + distance
                    grad_b = recv_buf
                    # print(f"rank: {rank} :: recv: {grad_b.size()}, send: {gradient[mid:,].size()}, neighbor: {neighbor}")
                    send_buf.copy_(gradient[mid:,])
                    bagua.send(tensor=send_buf, dst=neighbor)
                    bagua.recv(tensor=grad_b, src=neighbor)         # override the half of input which we do not use, could also use a separate buffer
                    grad_a = gradient[0:mid]                   # TODO: compare to grad_a = gradient[0:mid
                else:                                               # process left half
                    neighbor = rank - distance
                    grad_a = recv_buf
                    # print(f"rank: {rank} :: recv: {grad_a.size()}, send: {gradient[0:mid].size()}, neighbor: {neighbor}")
                    send_buf.copy_(gradient[0:mid])
                    bagua.recv(tensor=grad_a, src=neighbor)
                    bagua.send(tensor=send_buf, dst=neighbor)
                    grad_b = gradient[mid:,]
                    is_left = False
                
                new_distance = 2 * distance
                dim_cnt = len(gradient.size())  
                self.partial_dots_buffer[0].copy_(torch.tensordot(grad_a, grad_b, dims=dim_cnt))
                self.partial_dots_buffer[1].copy_(torch.tensordot(grad_a, grad_a, dims=dim_cnt))
                self.partial_dots_buffer[2].copy_(torch.tensordot(grad_b, grad_b, dims=dim_cnt))

                group = self.comm_groups[rec_level][math.floor(rank / new_distance)]
                bagua.allreduce_inplace(tensor=self.partial_dots_buffer,comm=group.get_global_communicator())
                # Inplace version of Adasum operation
                grad_a.mul_(1 - self.partial_dots_buffer[0] / (2 * self.partial_dots_buffer[1])).add_(
                    grad_b.mul_(1 - (self.partial_dots_buffer[0] / (2 * self.partial_dots_buffer[2]))))

                if new_distance < self.world_size:
                    grad_a.copy_(adasum(grad_a, new_distance, param_idx))

                if is_left:
                    bagua.send(tensor=grad_a, dst=neighbor)
                    bagua.recv(tensor=end_buf, src=neighbor)
                    return torch.cat(tensors=(grad_a, end_buf), dim=0)  

                else:
                    bagua.recv(tensor=end_buf, src=neighbor)
                    bagua.send(tensor=grad_a, dst=neighbor)
                    return torch.cat(tensors=(end_buf, grad_a), dim=0)

            optimizer = bagua_ddp.bagua_optimizers[0]
            for group in optimizer.param_groups:
                for idx, param in enumerate(group["params"]):
                    # param.grad.data = adasum(param.grad.data, 1, idx) 
                    param.grad.data.copy_(adasum(param.grad.data, 1, idx))

        return hook_adasum


class AdasumAlgorithm (Algorithm):
    def __init__(self, model_params, hierarchical: bool = False):
        """
        Create an instance of the
        `Adasum Algorithm <https://arxiv.org/abs/2006.02924>`
        .
        Args:
            sgd_optimizer: A regular SGD optimizer from PyTorch initialized with model parameters.
            hierarchical: Enable hierarchical communication. (not implemented for now)
        """
        self.hierarchical = hierarchical
        self.model_params = model_params

    def reify(self, process_group: BaguaProcessGroup) -> AdasumAlgorithmImpl:
        return AdasumAlgorithmImpl(
            process_group,
            self.model_params
        )
