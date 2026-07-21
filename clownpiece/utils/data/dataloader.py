from clownpiece.utils.data.dataset import Dataset
from clownpiece import Tensor
import random


class DefaultSampler:
  def __init__(self, length, shuffle):
    self.length = length
    self.shuffle = shuffle

  def __iter__(self):
    indices = list(range(self.length))
    if self.shuffle:
      random.shuffle(indices)
    yield from indices
    
  def __len__(self):
    return self.length
  
def default_collate_fn(batch):
  def stack_items(items):
    items = list(items)
    if isinstance(items[0], Tensor):
        return Tensor.stack(items)
    return Tensor.stack([Tensor(item) for item in items])

  if isinstance(batch[0], tuple):
    return tuple(stack_items(items) for items in zip(*batch))

  return stack_items(batch)

class Dataloader:
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 drop_last=False, 
                 sampler=None, 
                 collate_fn=None):
      self.dataset = dataset
      self.batch_size = batch_size
      self.drop_last = drop_last
      if sampler is None:
        self.sampler = DefaultSampler(len(dataset), shuffle)
      else:
        self.sampler = sampler
      if collate_fn is None:
        self.collate_fn = default_collate_fn
      else:
        self.collate_fn = collate_fn


    def __iter__(self):
        batch = []
        for index in iter(self.sampler):
          batch.append(self.dataset[index])
          if len(batch) == self.batch_size:
            yield self.collate_fn(batch)
            batch = []
        if batch and not self.drop_last:
          yield self.collate_fn(batch)

    def __len__(self):
      if self.drop_last:
        return len(self.sampler) // self.batch_size
      return (len(self.sampler) + self.batch_size - 1) // self.batch_size