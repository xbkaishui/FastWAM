import torch
from torch import nn

D_in = 32
D_out = 32

class simple_model(nn.Module):
    def __init__(self):
        super().__init__()
        num_layer = 10000
        
        self.blocks =  torch.nn.ModuleList([nn.Linear(D_in, D_out) for _ in range (num_layer)])
    
    def forward(self, x, y, z):
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        c = torch.add(a, b)
        for block in self.blocks:
            c = block(c)
        return c


class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}
    
    def capture(self, x, y, z):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x,y,z)
        torch.cuda.synchronize()
        self.cuda_graph.debug_dump(r"e:/dev/ai/my/FastWAM/test/graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x 
        self.graph_input['y'] = y
        self.graph_input['z'] = z
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out 
        
    def forward(self, x, y, z):
        self.graph_input['x'].copy_(x)
        self.graph_input['y'].copy_(y)
        self.graph_input['z'].copy_(z)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
def test_run():
    model = simple_model().cuda()
    inp = torch.randn(32, D_in).cuda()
    model.eval()
    model(x=inp, y=inp, z=inp) # warm up, 触发一些 gpu 资源的初始化

    graph_runner = CUDAGraphRunner(model)
    inputs = {"x":inp, "y":inp, "z":inp}
    graph_runner.capture(**inputs)
    graph_runner(**inputs) # cuda_graph_runner run
    
    
if __name__ == "__main__":
    test_run()
