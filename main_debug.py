from model import *
from trainer import Trainer

#
# Settings.
#

import ptvsd
ptvsd.enable_attach('pytorch',address=('0.0.0.0',5022))
ptvsd.wait_for_attach()
ptvsd.break_into_debugger()

fe = FrontEnd()
d = D()
q = Q()
g = G()

print(d)
print(q)
print(g)

for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
trainer.train()
