#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import Orange


data = Orange.data.Table("bridges")

#print 3 example instances:
print("===Example instances===")
for i in [1,25,82]:
    if i < len(data):
        print("%03d ::" %i, data[i])

#histogram
print("\===Histogram generation")
print(data.domain.features)
#attribute list and types

discAttr=[attr.name for attr in data.domain.features if attr.var_type==Orange.feature.Type.Discrete]
contAttr=[attr.name for attr in data.domain.features if attr.var_type==Orange.feature.Type.Continuous]
attrCount= len(data.domain.features)-1

print("\n===Attributes===")

print("Attributes count: ", attrCount)

print("There are %d discrete attributes: " %len(discAttr), discAttr)
print("There are %d continuous attributes: " %len(contAttr), contAttr)

print ("%-15s %-10s %s" % ("Attribute", "Mean/Mode", "Value"))

print("\n===Attribute Mean and Modal values===")
contDistr = Orange.statistics.basic.Domain(data)
for x in data.domain.features[:-1]:
    if x.var_type==Orange.feature.Type.Continuous:
        print("%-15s %-10s %f" % (x.name, "Mean", contDistr[x.name].avg))
    else:
        print("%-15s %-10s %s" % (x.name, "Mode", Orange.statistics.distribution.Discrete(data.domain[x.name]).modus()))

print("\n===Missing Values for attributes===")
for x in data.domain.features:
    n_miss = sum(1 for d in data if d[x].is_special())
    print("%-15s %d" %(x.name, n_miss))

print("\n===Random Sample===")
import random
indices2 = Orange.data.sample.SubsetIndices2(p0=0.1)
indices2.random_generator = Orange.misc.Random(random.randint(0, 50000))  
ind = indices2(data)
sample = data.select(ind, 0)
print("Sample of 10% of the instances in data set")
for x in sample:
    print(x)
