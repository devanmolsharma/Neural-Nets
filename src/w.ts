/// <reference path="TensorOperationsList.ts" />

const [t1, t2, t3] = [new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4])];
let j = new Sum()([t1, t2, t3])
j = new Mean()([j])
j.backward()
console.log(j.toString());