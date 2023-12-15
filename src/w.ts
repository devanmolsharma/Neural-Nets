/// <reference path="TensorOperationsList.ts" />
/// <reference path="Linear.ts" />
/// <reference path="Sequential.ts" />

let s = new Sequential();
let s2 = new Sequential();
s.add(new Linear(1,4))
s.add(new Linear(4,1))
s.add(new Linear(1,1))
console.log(s(new Tensor([[2]])));