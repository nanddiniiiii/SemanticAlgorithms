"""Aggregate native VQ EMA observations across DataParallel replicas.

DataParallel discards non-primary replicas after forward; allowing each replica
to update its own EMA would silently retain only GPU0's codebook observations.
"""
import queue


class ReplicaEMA:
    def __init__(self,modules):
        self.modules=modules;self.pending=queue.Queue();self.hooks=[];self.original={}
        for name,module in modules.items():
            self.original[name]=module.ema
            if not module.ema:raise ValueError('Expected native EMA VQ codebook: '+name)
            module.ema=False
            def collect(native,args,output,key=name):
                if native.training:
                    self.pending.put((key,args[0].detach(),output[1].detach()))
            self.hooks.append(module.register_forward_hook(collect))

    def update(self):
        import torch
        gathered={}
        while not self.pending.empty():
            name,values,indices=self.pending.get()
            gathered.setdefault(name,[]).append((values,indices))
        with torch.no_grad():
            for name,items in gathered.items():
                module=self.modules[name];device=module.weight.device
                dimension=items[0][0].shape[-1]
                values=torch.cat([value.reshape(-1,dimension).to(device) for value,_ in items],dim=0)
                indices=torch.cat([ids.reshape(-1).to(device) for _,ids in items],dim=0)
                module._update_buffers(values,indices)
                module._update_embedding()
        return {name:sum(value.numel()//value.shape[-1] for value,_ in items) for name,items in gathered.items()}

    def close(self):
        for hook in self.hooks:hook.remove()
        for name,module in self.modules.items():module.ema=self.original[name]
